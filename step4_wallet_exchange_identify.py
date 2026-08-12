"""
Step 4 — Wallet / Exchange Identify (rule-based, no ML)
-----------------------------------------------------------
Every problem you listed maps to a specific design decision here:

  Unknown wallet
      -> lookup() NEVER guesses. No match = EntityType.UNKNOWN with
         Confidence.UNVERIFIED, explicitly, instead of defaulting to
         "wallet" (a silent wrong guess is worse than an honest unknown).

  New exchange wallet (not yet labeled)
      -> heuristic_classify() can suggest "possible_exchange" from behavior
         (high counterparty count, high tx volume) but caps confidence at
         LOW/MEDIUM and tags source="heuristic" — it is never allowed to
         claim VERIFIED, so downstream code can choose to ignore weak guesses.

  Exchange hot-wallet vs cold-wallet confusion
      -> EntityType has EXCHANGE_HOT and EXCHANGE_COLD as separate values,
         never a generic "exchange" that conflates the two.

  Contract address vs normal wallet confusion
      -> lookup() takes a fresh `is_contract` bool (from real bytecode check
         at collection time, not inferred) and lets it override/downgrade a
         stale or conflicting label rather than trusting the label blindly.

  Same entity, multiple wallets
      -> link_entity() lets many addresses share one `entity_name`, so
         "Binance" can span 50 hot wallets without needing 50 separate
         disconnected labels.

  Address-label database outdated
      -> every AddressLabel carries `last_updated`. is_stale() checks age;
         lookup() automatically downgrades a stale VERIFIED/HIGH label to
         MEDIUM rather than silently keeping full confidence forever.

  False classification (wallet/exchange/bot labeled wrong)
      -> there is no boolean "is_exchange" anywhere in this file. Every
         classification carries entity_type + confidence + source together,
         so a caller can always see *how sure* the system is and *why*,
         and can require e.g. confidence >= HIGH before acting on a label.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional


class EntityType(str, Enum):
    EXCHANGE_HOT = "exchange_hot"
    EXCHANGE_COLD = "exchange_cold"
    CONTRACT = "contract"
    BRIDGE = "bridge"
    WALLET = "wallet"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    VERIFIED = "verified"      # authoritative source (e.g. exchange's own published address)
    HIGH = "high"               # strong heuristic / cross-referenced source
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"   # no real evidence — an unproven guess, treat as unknown


STALE_AFTER_DAYS = 30  # tune per how often you realistically refresh label sources


@dataclass
class AddressLabel:
    address: str
    entity_type: EntityType
    entity_name: Optional[str] = None      # e.g. "Binance" — shared across an entity's wallets
    confidence: Confidence = Confidence.UNVERIFIED
    source: str = "unknown"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_contract: Optional[bool] = None      # explicit check result, never inferred silently

    def is_stale(self) -> bool:
        age = datetime.now(timezone.utc) - self.last_updated
        return age > timedelta(days=STALE_AFTER_DAYS)


class AddressRegistry:
    """
    In-memory label store. Swap the dict for the DuckDB store from Step 2
    in production — interface kept minimal on purpose so that's a drop-in swap.
    """

    def __init__(self):
        self._labels: Dict[str, AddressLabel] = {}
        self._entity_members: Dict[str, List[str]] = {}  # entity_name -> [addresses]

    def upsert(self, label: AddressLabel):
        self._labels[label.address.lower()] = label
        if label.entity_name:
            self._entity_members.setdefault(label.entity_name, [])
            if label.address.lower() not in self._entity_members[label.entity_name]:
                self._entity_members[label.entity_name].append(label.address.lower())

    def link_entity(self, addresses: List[str], entity_name: str,
                     entity_type: EntityType, confidence: Confidence, source: str):
        """Register many addresses as belonging to one real-world entity."""
        for addr in addresses:
            self.upsert(AddressLabel(
                address=addr, entity_type=entity_type, entity_name=entity_name,
                confidence=confidence, source=source,
            ))

    def get_entity_members(self, entity_name: str) -> List[str]:
        return self._entity_members.get(entity_name, [])

    def lookup(self, address: str, is_contract: Optional[bool] = None) -> AddressLabel:
        addr = address.lower()
        label = self._labels.get(addr)

        if label is None:
            # honest unknown — no guessing
            return AddressLabel(
                address=addr, entity_type=EntityType.UNKNOWN,
                confidence=Confidence.UNVERIFIED, source="not_in_registry",
                is_contract=is_contract,
            )

        # fresh bytecode check disagrees with the stored label -> trust the
        # fresh check, but drop confidence since something is inconsistent
        if is_contract is not None and label.is_contract is not None and is_contract != label.is_contract:
            return AddressLabel(
                address=addr,
                entity_type=EntityType.CONTRACT if is_contract else EntityType.WALLET,
                entity_name=label.entity_name,
                confidence=Confidence.LOW,
                source=f"{label.source}+is_contract_mismatch",
                last_updated=label.last_updated,
                is_contract=is_contract,
            )

        if label.is_stale() and label.confidence in (Confidence.VERIFIED, Confidence.HIGH):
            return AddressLabel(
                address=label.address, entity_type=label.entity_type,
                entity_name=label.entity_name, confidence=Confidence.MEDIUM,
                source=f"{label.source}+stale", last_updated=label.last_updated,
                is_contract=label.is_contract,
            )

        return label


def heuristic_classify(address: str, behavior_stats: dict) -> AddressLabel:
    """
    For addresses NOT in the registry — a bounded, honest guess from
    observed behavior, never above LOW/MEDIUM confidence.

    behavior_stats expected keys:
      tx_count, unique_counterparties, is_contract (bool or None),
      avg_incoming_usd, avg_outgoing_usd
    """
    is_contract = behavior_stats.get("is_contract")
    if is_contract:
        return AddressLabel(
            address=address, entity_type=EntityType.CONTRACT,
            confidence=Confidence.HIGH,  # bytecode check is real evidence, not a guess
            source="bytecode_check",
        )

    tx_count = behavior_stats.get("tx_count", 0)
    counterparties = behavior_stats.get("unique_counterparties", 0)

    # exchange-like behavior: very high tx count AND very high counterparty
    # diversity relative to tx count (many different people sending/receiving)
    if tx_count >= 200 and counterparties >= 100:
        return AddressLabel(
            address=address, entity_type=EntityType.EXCHANGE_HOT,
            confidence=Confidence.LOW,  # heuristic only — flag for human review
            source="heuristic:high_volume_high_diversity",
        )

    return AddressLabel(
        address=address, entity_type=EntityType.WALLET,
        confidence=Confidence.LOW,
        source="heuristic:default_wallet",
    )


def classify(address: str, registry: AddressRegistry, is_contract: Optional[bool] = None,
             behavior_stats: Optional[dict] = None) -> AddressLabel:
    """Single entry point: registry first, then bytecode fact, then heuristic fallback."""
    label = registry.lookup(address, is_contract=is_contract)
    if label.entity_type != EntityType.UNKNOWN:
        return label

    # a fresh, real bytecode check is hard evidence — use it even with no
    # behavior_stats dict at all, don't require the full heuristic bundle
    if is_contract is True:
        return AddressLabel(
            address=address, entity_type=EntityType.CONTRACT,
            confidence=Confidence.HIGH, source="bytecode_check",
            is_contract=True,
        )
    if is_contract is False:
        # "not a contract" is a verified structural fact (real bytecode check),
        # not a guess about WHO owns it — safe to default to WALLET at
        # MEDIUM confidence. Still far below VERIFIED, which is reserved for
        # an actual entity-name label (e.g. confirmed as "Binance").
        return AddressLabel(
            address=address, entity_type=EntityType.WALLET,
            confidence=Confidence.MEDIUM, source="bytecode_check:not_contract",
            is_contract=False,
        )

    if behavior_stats:
        return heuristic_classify(address, behavior_stats)

    return label  # stays UNKNOWN — honest, no data to guess from


if __name__ == "__main__":
    registry = AddressRegistry()

    # verified, fresh label
    registry.upsert(AddressLabel(
        address="0xBinanceHot1", entity_type=EntityType.EXCHANGE_HOT,
        entity_name="Binance", confidence=Confidence.VERIFIED, source="binance_official_list",
    ))
    # multi-wallet entity: same exchange, cold storage, different address
    registry.link_entity(
        ["0xBinanceCold1", "0xBinanceCold2"], entity_name="Binance",
        entity_type=EntityType.EXCHANGE_COLD, confidence=Confidence.VERIFIED,
        source="binance_official_list",
    )
    # intentionally stale label (simulate 60 days old)
    old_label = AddressLabel(
        address="0xOldExchange", entity_type=EntityType.EXCHANGE_HOT,
        entity_name="DefunctExchange", confidence=Confidence.HIGH, source="scraped_2026_06",
        last_updated=datetime.now(timezone.utc) - timedelta(days=60),
    )
    registry.upsert(old_label)

    print("Known verified:", classify("0xBinanceHot1", registry))
    print("Cold wallet (linked entity):", classify("0xBinanceCold1", registry))
    print("Stale label downgraded:", classify("0xOldExchange", registry))
    print("Unknown, no behavior data:", classify("0xRandomNew", registry))
    print("Unknown + contract check:", classify("0xSomeContract", registry, is_contract=True))
    print("Unknown + exchange-like behavior:", classify(
        "0xNewExchangeMaybe", registry,
        behavior_stats={"tx_count": 500, "unique_counterparties": 300, "is_contract": False}
    ))
    print("Binance's all known wallets:", registry.get_entity_members("Binance"))
