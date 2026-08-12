"""
Step 6 — Gas Source Track (rule-based, no ML)
------------------------------------------------
Every problem you listed maps to a specific design decision:

  Gas payer != transaction sender
      -> GasRecord has SEPARATE `tx_signer` and `gas_payer` fields, never
         one field doing double duty. Default assumption is gas_payer ==
         tx_signer ONLY when nothing says otherwise (see relayer check).

  Relayer / meta-transactions
      -> detect_gas_payer() checks the `to_address` against a known
         relayer/forwarder registry (EIP-2771 style) BEFORE assuming
         signer == payer. If matched, gas_payer is tagged as the relayer,
         signer is preserved separately, both visible in the output.

  Smart-contract transactions
      -> `is_contract` on the interaction target is carried through from
         Step 4/5's classification, not re-guessed here — contract calls
         often have different gas profiles (gas_used varies a lot more
         than a plain transfer), so this is exposed as a flag, not hidden.

  Gas sponsorship
      -> same mechanism as relayer detection: a known sponsor-contract
         registry, same `gas_payer != tx_signer` output shape.

  Different chains' gas models
      -> GasFeeModel enum (LEGACY / EIP1559 / FIXED_FEE) + normalize_fee()
         converts each chain's native fee structure into one common
         `gas_fee_usd` output, so downstream code never needs to know
         whether it's looking at Ethereum gwei math or Solana's flat
         lamports fee.

  Gas price / gas-used data missing
      -> normalize_fee() NEVER silently treats missing as 0. Returns
         data_quality="missing" and fee=None instead of a fake zero that
         would quietly bias any aggregate/average built on top of it.

  Funding wallet not identified
      -> `infer_funding_wallet()` is explicitly a SEPARATE, best-effort
         lookup (first known inbound transfer to the gas payer), always
         labeled `inferred=True` — never presented as a confirmed fact.

  Gas source wrongly associated with the wrong wallet
      -> every GasRecord keeps tx_signer, gas_payer, and (if inferred)
         funding_wallet as three DISTINCT fields with their own confidence
         — nothing here ever collapses them into one "source" silently.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class GasFeeModel(str, Enum):
    EIP1559 = "eip1559"      # Ethereum/Polygon/etc post-London: base_fee + priority_fee
    LEGACY = "legacy"         # pre-EIP1559: flat gas_price * gas_used
    FIXED_FEE = "fixed_fee"   # e.g. Solana: fixed lamports per signature, not usage-based


# Known relayer / meta-transaction forwarder contracts (EIP-2771 style) and
# gas-sponsorship contracts. In production this is a maintained registry,
# same staleness concerns as Step 4's AddressRegistry apply here too.
KNOWN_RELAYERS = {
    "0xtrustedforwarder1": "OpenGSN Forwarder",
    "0xbiconomyforwarder1": "Biconomy Forwarder",
}
KNOWN_GAS_SPONSORS = {
    "0xsponsorcontract1": "Coinbase Paymaster",
}


@dataclass
class GasRecord:
    tx_hash: str
    tx_signer: str                       # who signed/authorized the transaction
    gas_payer: str                       # who actually paid the gas fee (may differ)
    gas_payer_confidence: str            # "verified" | "inferred" | "assumed"
    gas_payer_source: str
    is_contract_interaction: Optional[bool] = None
    fee_model: Optional[GasFeeModel] = None
    gas_fee_native: Optional[float] = None
    gas_fee_usd: Optional[float] = None
    data_quality: str = "ok"             # "ok" | "missing" | "partial"
    funding_wallet: Optional[str] = None
    funding_wallet_inferred: bool = False


def detect_gas_payer(tx_signer: str, to_address: str, raw_gas_payer: Optional[str] = None) -> tuple[str, str, str]:
    """
    Returns (gas_payer, confidence, source).

    Priority:
      1. If the chain/indexer directly reports who paid gas (raw_gas_payer,
         e.g. from a trace or a relay-specific event log) -> that's VERIFIED.
      2. If `to_address` is a known relayer/sponsor contract -> the signer
         almost certainly didn't pay gas themselves -> flag as INFERRED,
         name the relayer, but don't claim to know the true sponsor wallet
         unless the trace data actually says so.
      3. Otherwise -> ASSUME signer paid their own gas (the normal case),
         but label it ASSUMED, not VERIFIED — leaves room for correction
         if better data shows up later.
    """
    if raw_gas_payer:
        return raw_gas_payer, "verified", "trace_data"

    to_lower = (to_address or "").lower()
    if to_lower in KNOWN_RELAYERS:
        return (
            f"unknown_via_{KNOWN_RELAYERS[to_lower]}",
            "inferred",
            f"relayer_contract:{KNOWN_RELAYERS[to_lower]}",
        )
    if to_lower in KNOWN_GAS_SPONSORS:
        return (
            f"unknown_via_{KNOWN_GAS_SPONSORS[to_lower]}",
            "inferred",
            f"gas_sponsor_contract:{KNOWN_GAS_SPONSORS[to_lower]}",
        )

    return tx_signer, "assumed", "default_signer_pays_own_gas"


def normalize_fee(
    chain: str,
    fee_model: GasFeeModel,
    gas_used: Optional[float],
    gas_price_native: Optional[float] = None,      # legacy chains
    base_fee_native: Optional[float] = None,        # EIP-1559 chains
    priority_fee_native: Optional[float] = None,     # EIP-1559 chains
    fixed_fee_native: Optional[float] = None,        # fixed-fee chains
    native_price_usd: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], str]:
    """
    Returns (gas_fee_native, gas_fee_usd, data_quality).

    Never fabricates a 0 for missing inputs — returns None + data_quality
    flag instead, so a caller building an average doesn't silently treat
    "we don't know" as "it cost nothing" (which would skew every aggregate
    built on top of this).
    """
    if native_price_usd is None:
        # can still compute the native fee, just can't convert to USD
        native_price_usd = None

    if fee_model == GasFeeModel.FIXED_FEE:
        if fixed_fee_native is None:
            return None, None, "missing"
        usd = fixed_fee_native * native_price_usd if native_price_usd else None
        return fixed_fee_native, usd, "ok" if native_price_usd else "partial"

    if fee_model == GasFeeModel.LEGACY:
        if gas_used is None or gas_price_native is None:
            return None, None, "missing"
        native_fee = gas_used * gas_price_native
        usd = native_fee * native_price_usd if native_price_usd else None
        return native_fee, usd, "ok" if native_price_usd else "partial"

    if fee_model == GasFeeModel.EIP1559:
        if gas_used is None or base_fee_native is None or priority_fee_native is None:
            return None, None, "missing"
        native_fee = gas_used * (base_fee_native + priority_fee_native)
        usd = native_fee * native_price_usd if native_price_usd else None
        return native_fee, usd, "ok" if native_price_usd else "partial"

    return None, None, "missing"  # unrecognized fee model — don't guess a number


def infer_funding_wallet(gas_payer: str, past_transactions: List[dict]) -> Optional[dict]:
    """
    Best-effort: find the earliest known inbound transfer TO gas_payer in
    the supplied transaction history, as a candidate "who funded this
    wallet in the first place" answer.

    past_transactions: list of dicts with keys: to_address, from_address, timestamp
    Must be pre-sorted ascending by timestamp, or this returns the wrong answer —
    intentionally not sorting here so caller controls (and knows) the source order.

    ALWAYS returns inferred=True — this is a plausibility hint for investigation,
    never presented as a confirmed fact.
    """
    for tx in past_transactions:
        if tx.get("to_address", "").lower() == gas_payer.lower():
            return {
                "funding_wallet": tx["from_address"],
                "inferred": True,
                "basis": "earliest_known_inbound_transfer",
                "as_of_timestamp": tx.get("timestamp"),
            }
    return None  # genuinely don't know — say so, don't guess


def track_gas_source(
    tx_hash: str, tx_signer: str, to_address: str, chain: str,
    fee_model: GasFeeModel, gas_used: Optional[float] = None,
    gas_price_native: Optional[float] = None, base_fee_native: Optional[float] = None,
    priority_fee_native: Optional[float] = None, fixed_fee_native: Optional[float] = None,
    native_price_usd: Optional[float] = None, raw_gas_payer: Optional[str] = None,
    is_contract_interaction: Optional[bool] = None,
    past_transactions_for_funding_lookup: Optional[List[dict]] = None,
) -> GasRecord:
    """Single entry point tying all of the above together for one transaction."""

    gas_payer, payer_confidence, payer_source = detect_gas_payer(tx_signer, to_address, raw_gas_payer)

    native_fee, usd_fee, quality = normalize_fee(
        chain, fee_model, gas_used, gas_price_native,
        base_fee_native, priority_fee_native, fixed_fee_native, native_price_usd,
    )

    record = GasRecord(
        tx_hash=tx_hash, tx_signer=tx_signer, gas_payer=gas_payer,
        gas_payer_confidence=payer_confidence, gas_payer_source=payer_source,
        is_contract_interaction=is_contract_interaction, fee_model=fee_model,
        gas_fee_native=native_fee, gas_fee_usd=usd_fee, data_quality=quality,
    )

    if payer_confidence == "inferred" and past_transactions_for_funding_lookup:
        funding = infer_funding_wallet(tx_signer, past_transactions_for_funding_lookup)
        if funding:
            record.funding_wallet = funding["funding_wallet"]
            record.funding_wallet_inferred = True

    return record


if __name__ == "__main__":
    print("--- Case 1: normal EIP-1559 tx, signer pays own gas ---")
    r1 = track_gas_source(
        tx_hash="0x1", tx_signer="0xUser1", to_address="0xUser2", chain="ethereum",
        fee_model=GasFeeModel.EIP1559, gas_used=21000,
        base_fee_native=0.00000003, priority_fee_native=0.000000002, native_price_usd=2400,
    )
    print(r1)

    print("\n--- Case 2: meta-transaction via known relayer (gas payer != signer) ---")
    r2 = track_gas_source(
        tx_hash="0x2", tx_signer="0xUser1", to_address="0xTrustedForwarder1", chain="ethereum",
        fee_model=GasFeeModel.EIP1559, gas_used=45000,
        base_fee_native=0.00000003, priority_fee_native=0.000000002, native_price_usd=2400,
        past_transactions_for_funding_lookup=[
            {"to_address": "0xUser1", "from_address": "0xOriginalFunder", "timestamp": "2026-08-01T00:00:00"},
        ],
    )
    print(r2)

    print("\n--- Case 3: missing gas data — must NOT silently become 0 ---")
    r3 = track_gas_source(
        tx_hash="0x3", tx_signer="0xUser3", to_address="0xUser4", chain="ethereum",
        fee_model=GasFeeModel.EIP1559, gas_used=None,  # missing from source data
        base_fee_native=0.00000003, priority_fee_native=0.000000002, native_price_usd=2400,
    )
    print(r3)
    assert r3.gas_fee_usd is None and r3.data_quality == "missing", "must not fabricate a fee"

    print("\n--- Case 4: Solana-style fixed fee, different model entirely ---")
    r4 = track_gas_source(
        tx_hash="0x4", tx_signer="SolWallet1", to_address="SolWallet2", chain="solana",
        fee_model=GasFeeModel.FIXED_FEE, fixed_fee_native=0.000005, native_price_usd=95,
    )
    print(r4)

    print("\n--- Case 5: no USD price available — native fee known, USD must stay None ---")
    r5 = track_gas_source(
        tx_hash="0x5", tx_signer="0xUser5", to_address="0xUser6", chain="ethereum",
        fee_model=GasFeeModel.LEGACY, gas_used=21000, gas_price_native=0.00000005,
        native_price_usd=None,
    )
    print(r5)
    assert r5.gas_fee_native is not None and r5.gas_fee_usd is None and r5.data_quality == "partial"

    print("\nAll assertions passed.")
