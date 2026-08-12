"""
Step 5 — Direction (rule-based, no ML)
--------------------------------------------
Depends on Step 4's AddressRegistry — direction logic is only as good as the
entity labels underneath it, so it's built to degrade honestly, not guess.

  Wallet→Exchange vs Exchange→Wallet misclassified
      -> classify_direction() looks up BOTH sides via the Step 4 registry
         and only commits to a direction label when confidence supports it;
         low-confidence entity labels propagate as low-confidence direction,
         they're never silently treated as certain.

  Internal exchange transfers mistaken for deposit/withdrawal
      -> if both addresses resolve to the SAME entity_name (e.g. both are
         "Binance" wallets, hot or cold), tagged INTERNAL_TRANSFER explicitly
         — never DEPOSIT/WITHDRAWAL, which would double-count real flow.

  Wallet→Wallet meaning unclear
      -> tagged WALLET_TO_WALLET, no invented "accumulation/dump" story
         layered on top without exchange context to support it.

  Bridge transfers
      -> checked as a distinct EntityType.BRIDGE before falling through to
         wallet/exchange logic, tagged BRIDGE_TRANSFER, not miscounted as
         a wallet-to-wallet or exchange move.

  Smart-contract interactions
      -> checked before falling through to WALLET_TO_WALLET; a non-exchange
         contract on either side is CONTRACT_INTERACTION, not silently
         treated like a simple transfer.

  Multi-hop transactions
      -> multi-hop can't be proven from one transaction in isolation. This
         module explicitly does NOT claim to detect it from a single row.
         `flag_possible_multihop()` is a separate, honestly-scoped heuristic
         that needs a transaction *sequence* per wallet (see docstring there)
         and returns a suspicion flag, never a direction label.

  Exchange address list incomplete
      -> when either side resolves to UNKNOWN, direction confidence is
         capped at LOW and the tag says "_unconfirmed" — the gap in your
         label coverage is visible in the output, not hidden by a guess.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from step4_wallet_exchange_identify import (
    AddressRegistry, AddressLabel, EntityType, Confidence, classify
)


class DirectionTag(str, Enum):
    DEPOSIT = "deposit_to_exchange"                # wallet -> exchange
    WITHDRAWAL = "withdrawal_from_exchange"          # exchange -> wallet
    INTERNAL_TRANSFER = "internal_exchange_transfer"  # exchange -> same exchange
    WALLET_TO_WALLET = "wallet_to_wallet"
    BRIDGE_TRANSFER = "bridge_transfer"
    CONTRACT_INTERACTION = "contract_interaction"
    UNCONFIRMED = "unconfirmed"  # one or both sides unresolvable — say so, don't guess


@dataclass
class DirectionResult:
    tag: DirectionTag
    from_label: AddressLabel
    to_label: AddressLabel
    confidence: Confidence
    note: str = ""


def classify_direction(
    from_address: str, to_address: str, registry: AddressRegistry,
    from_is_contract: Optional[bool] = None, to_is_contract: Optional[bool] = None,
) -> DirectionResult:

    from_label = classify(from_address, registry, is_contract=from_is_contract)
    to_label = classify(to_address, registry, is_contract=to_is_contract)

    from_t, to_t = from_label.entity_type, to_label.entity_type

    # bridge check first — a bridge contract on either side overrides
    # everything else, it's not a "wallet" move even if the other side is one
    if from_t == EntityType.BRIDGE or to_t == EntityType.BRIDGE:
        return DirectionResult(
            tag=DirectionTag.BRIDGE_TRANSFER, from_label=from_label, to_label=to_label,
            confidence=min(from_label.confidence, to_label.confidence, key=_confidence_rank),
            note="cross-chain bridge involved — treat as leaving/entering this chain's accounting, not a normal transfer",
        )

    # internal exchange transfer — same entity_name on both sides
    if (from_label.entity_name and to_label.entity_name
            and from_label.entity_name == to_label.entity_name
            and from_t in (EntityType.EXCHANGE_HOT, EntityType.EXCHANGE_COLD)
            and to_t in (EntityType.EXCHANGE_HOT, EntityType.EXCHANGE_COLD)):
        return DirectionResult(
            tag=DirectionTag.INTERNAL_TRANSFER, from_label=from_label, to_label=to_label,
            confidence=min(from_label.confidence, to_label.confidence, key=_confidence_rank),
            note=f"internal move within {from_label.entity_name} (e.g. hot<->cold rebalance) — not a real deposit/withdrawal",
        )

    # unresolved on either side — do not guess a direction
    if from_t == EntityType.UNKNOWN or to_t == EntityType.UNKNOWN:
        return DirectionResult(
            tag=DirectionTag.UNCONFIRMED, from_label=from_label, to_label=to_label,
            confidence=Confidence.UNVERIFIED,
            note="one or both addresses unresolved — exchange address list may be incomplete, do not treat as wallet-to-wallet",
        )

    # deposit: wallet -> exchange
    if from_t == EntityType.WALLET and to_t in (EntityType.EXCHANGE_HOT, EntityType.EXCHANGE_COLD):
        return DirectionResult(
            tag=DirectionTag.DEPOSIT, from_label=from_label, to_label=to_label,
            confidence=min(from_label.confidence, to_label.confidence, key=_confidence_rank),
        )

    # withdrawal: exchange -> wallet
    if from_t in (EntityType.EXCHANGE_HOT, EntityType.EXCHANGE_COLD) and to_t == EntityType.WALLET:
        return DirectionResult(
            tag=DirectionTag.WITHDRAWAL, from_label=from_label, to_label=to_label,
            confidence=min(from_label.confidence, to_label.confidence, key=_confidence_rank),
        )

    # contract interaction: either side is a non-exchange, non-bridge contract
    if from_t == EntityType.CONTRACT or to_t == EntityType.CONTRACT:
        return DirectionResult(
            tag=DirectionTag.CONTRACT_INTERACTION, from_label=from_label, to_label=to_label,
            confidence=min(from_label.confidence, to_label.confidence, key=_confidence_rank),
            note="DeFi/contract call, not a plain transfer — do not interpret as accumulation/dump",
        )

    # both plain wallets
    return DirectionResult(
        tag=DirectionTag.WALLET_TO_WALLET, from_label=from_label, to_label=to_label,
        confidence=min(from_label.confidence, to_label.confidence, key=_confidence_rank),
        note="no exchange context on either side — meaning genuinely unclear, don't invent one",
    )


_CONF_ORDER = [Confidence.UNVERIFIED, Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH, Confidence.VERIFIED]

def _confidence_rank(c: Confidence) -> int:
    return _CONF_ORDER.index(c)


def flag_possible_multihop(tx_sequence: List[dict], time_window_minutes: int = 15,
                            amount_tolerance_pct: float = 0.02) -> List[dict]:
    """
    Multi-hop detection needs a SEQUENCE of transactions, not one row — this
    is intentionally a separate function so classify_direction() never
    silently claims multi-hop knowledge it doesn't have from a single tx.

    Heuristic: wallet A -> B, then B -> C within `time_window_minutes`, with
    C's outgoing amount within `amount_tolerance_pct` of B's incoming amount
    (accounting for gas) suggests B is a pass-through hop, not a destination.

    tx_sequence: list of dicts with keys: from_address, to_address, amount_usd, timestamp
    Sorted by timestamp is assumed. Returns the subset of transactions flagged
    as likely hops, each tagged with 'possible_multihop': True — this is a
    SUSPICION flag for human review, not a confirmed direction.
    """
    flagged = []
    for i, tx in enumerate(tx_sequence):
        for j in range(i + 1, len(tx_sequence)):
            nxt = tx_sequence[j]
            if nxt["from_address"] != tx["to_address"]:
                continue
            minutes_apart = (nxt["timestamp"] - tx["timestamp"]).total_seconds() / 60
            if minutes_apart > time_window_minutes:
                break  # sequence assumed time-sorted; nothing further will be closer
            amount_diff_pct = abs(nxt["amount_usd"] - tx["amount_usd"]) / max(tx["amount_usd"], 1e-9)
            if amount_diff_pct <= amount_tolerance_pct:
                flagged.append({**tx, "possible_multihop": True,
                                 "hop_to": nxt["to_address"], "minutes_to_next_hop": round(minutes_apart, 1)})
    return flagged


if __name__ == "__main__":
    import pandas as pd
    registry = AddressRegistry()
    registry.upsert(AddressLabel("0xBinanceHot1", EntityType.EXCHANGE_HOT, "Binance", Confidence.VERIFIED, "official"))
    registry.upsert(AddressLabel("0xBinanceCold1", EntityType.EXCHANGE_COLD, "Binance", Confidence.VERIFIED, "official"))
    registry.upsert(AddressLabel("0xPolygonBridge", EntityType.BRIDGE, "Polygon Bridge", Confidence.VERIFIED, "official"))
    registry.upsert(AddressLabel("0xUniswapRouter", EntityType.CONTRACT, "Uniswap V3", Confidence.VERIFIED, "official", is_contract=True))

    cases = [
        ("0xUserWallet1", "0xBinanceHot1", False, False, "expect DEPOSIT"),
        ("0xBinanceHot1", "0xUserWallet1", False, False, "expect WITHDRAWAL"),
        ("0xBinanceHot1", "0xBinanceCold1", False, False, "expect INTERNAL_TRANSFER"),
        ("0xUserWallet1", "0xUserWallet2", False, False, "expect WALLET_TO_WALLET"),
        ("0xUserWallet1", "0xPolygonBridge", False, False, "expect BRIDGE_TRANSFER"),
        ("0xUserWallet1", "0xUniswapRouter", False, True, "expect CONTRACT_INTERACTION"),
        ("0xUserWallet1", "0xNeverSeenBefore", False, None, "expect UNCONFIRMED (is_contract unknown)"),
    ]
    for frm, to, frm_ic, to_ic, expectation in cases:
        result = classify_direction(frm, to, registry, from_is_contract=frm_ic, to_is_contract=to_ic)
        print(f"{frm} -> {to}: {result.tag.value:30s} conf={result.confidence.value:10s} ({expectation})")

    print("\n--- multi-hop check ---")
    seq = [
        {"from_address": "0xA", "to_address": "0xB", "amount_usd": 100_000, "timestamp": pd.Timestamp("2026-08-01 10:00")},
        {"from_address": "0xB", "to_address": "0xC", "amount_usd": 99_500, "timestamp": pd.Timestamp("2026-08-01 10:05")},
        {"from_address": "0xX", "to_address": "0xY", "amount_usd": 5_000, "timestamp": pd.Timestamp("2026-08-01 11:00")},
    ]
    hops = flag_possible_multihop(seq)
    for h in hops:
        print(f"possible hop: {h['from_address']} -> {h['to_address']} -> {h['hop_to']} ({h['minutes_to_next_hop']} min)")
