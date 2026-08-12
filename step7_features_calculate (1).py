"""
Step 7 — Features Calculate (rule-based, no ML)
---------------------------------------------------
This step builds features FOR later use by a model — it does not train or
run one. Every problem you listed maps to a specific design decision:

  Missing values
      -> compute_transaction_features() never fillna(0) blindly. Each
         feature either has a principled default (documented inline) or
         is left None with `data_quality` noted, same policy as Step 6.

  Outliers
      -> log1p on amount (heavy right-skew) is applied at calculation time,
         not "fixed" by clipping — clipping would destroy the signal that a
         $10M transaction genuinely IS more extreme than a $10K one, which
         is exactly what whale detection needs to see.

  Feature scaling
      -> deliberately NOT done here. Scaling is a model-time decision
         (depends on what model consumes these features) — doing it at
         calculation time would silently couple this module to one
         specific model, and break if some other consumer reads the same
         stored features later. Raw + engineered values are stored;
         scaling happens downstream, once, right before model input.

  Highly correlated / redundant features
      -> feature_correlation_report() computes a correlation matrix so
         redundancy is VISIBLE, but does not auto-drop anything — that's
         a modeling decision requiring judgment, not something this layer
         should silently decide for you.

  Data leakage / Look-ahead bias / Wrong time-window
      -> compute_wallet_rolling_features() takes an explicit `as_of`
         cutoff and filters to `timestamp < as_of` (strictly before, not
         <=, and not centered) before computing anything. There is no
         function anywhere in this file that can see data from after its
         own cutoff — enforced structurally, not by convention.

  Feature drift
      -> snapshot_feature_distribution() records mean/std/percentiles for
         a batch of computed features with a timestamp. Not a fix by
         itself — it's the instrumentation needed to DETECT drift later
         by comparing snapshots over time.

  Real-time feature calc slow hona
      -> IncrementalWalletStats keeps running sums (count, sum, sum-of-
         squares, last-seen) per wallet, updated in O(1) per new
         transaction — no full groupby-scan of history on every new tx,
         which is what makes a naive implementation slow at scale.

  Features not useful for a future model
      -> genuinely can't be guaranteed at this layer without a labeled
         target to validate against — flagged honestly as an open
         question, not something fixable by better feature-engineering
         code alone.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-transaction features (no leakage risk — each row uses only its own data)
# ---------------------------------------------------------------------------

def compute_transaction_features(tx: dict) -> dict:
    """
    tx expected keys: amount_usd, gas_fee_usd (optional), timestamp

    Missing-value policy, stated explicitly per field:
      - amount_usd missing -> cannot compute anything meaningful, raise
        (this should never happen for a real transaction; if it does,
        that's a Step 1/2 data-quality bug, not something to paper over here)
      - gas_fee_usd missing -> log_gas_fee = None, NOT 0. A $0 gas fee is a
        real (if rare) value; a MISSING one must stay distinguishable from it.
    """
    if tx.get("amount_usd") is None:
        raise ValueError(f"amount_usd missing for tx {tx.get('tx_hash', '?')} — upstream data-quality issue")

    amount = tx["amount_usd"]
    gas_fee = tx.get("gas_fee_usd")

    return {
        "log_amount": math.log1p(amount),           # heavy right-skew handled here, not by clipping
        "log_gas_fee": math.log1p(gas_fee) if gas_fee is not None else None,
        "amount_usd": amount,                         # raw value kept too — scaling deferred to model time
        "gas_fee_usd": gas_fee,
        "data_quality": "ok" if gas_fee is not None else "partial_missing_gas_fee",
    }


# ---------------------------------------------------------------------------
# Incremental per-wallet stats — O(1) update, no full history rescan
# ---------------------------------------------------------------------------

@dataclass
class IncrementalWalletStats:
    """
    Running statistics per wallet. Updated one transaction at a time.
    Mean/variance computed via Welford's online algorithm — numerically
    stable, and never requires re-reading the full transaction history.
    """
    count: int = 0
    mean_log_amount: float = 0.0
    m2_log_amount: float = 0.0   # sum of squared deviations, for variance
    last_seen: Optional[datetime] = None
    total_amount_usd: float = 0.0

    def update(self, log_amount: float, amount_usd: float, timestamp: datetime):
        self.count += 1
        delta = log_amount - self.mean_log_amount
        self.mean_log_amount += delta / self.count
        delta2 = log_amount - self.mean_log_amount
        self.m2_log_amount += delta * delta2
        self.total_amount_usd += amount_usd
        self.last_seen = timestamp

    @property
    def std_log_amount(self) -> float:
        if self.count < 2:
            return 0.0  # a real zero (no spread observed yet), not a missing value
        return math.sqrt(self.m2_log_amount / (self.count - 1))

    def hours_since_last_seen(self, now: datetime) -> Optional[float]:
        if self.last_seen is None:
            return None  # genuinely unknown for a wallet's first-ever transaction
        return (now - self.last_seen).total_seconds() / 3600


class WalletStatsTracker:
    """Holds one IncrementalWalletStats per wallet — the O(1)-per-update store."""

    def __init__(self):
        self._stats: Dict[str, IncrementalWalletStats] = {}

    def process_transaction(self, wallet: str, amount_usd: float, timestamp: datetime) -> dict:
        stats = self._stats.setdefault(wallet, IncrementalWalletStats())

        # snapshot BEFORE updating — a wallet's "time since its own last tx" and
        # "its behavior up to just before this one" must exclude the current
        # transaction itself, or every wallet's first anomaly score is biased
        # by information the model shouldn't have yet (this is the real-time
        # equivalent of the leakage guard in compute_wallet_rolling_features)
        hours_since_last = stats.hours_since_last_seen(timestamp)
        prior_mean = stats.mean_log_amount if stats.count > 0 else None
        prior_std = stats.std_log_amount if stats.count > 0 else None

        log_amount = math.log1p(amount_usd)
        stats.update(log_amount, amount_usd, timestamp)

        return {
            "wallet": wallet,
            "wallet_time_delta_hours": hours_since_last,   # None on first-ever tx, not 0
            "wallet_prior_mean_log_amount": prior_mean,      # None on first-ever tx
            "wallet_prior_std_log_amount": prior_std,
            "wallet_tx_count_so_far": stats.count,           # includes this tx
        }


# ---------------------------------------------------------------------------
# Rolling wallet-level features — leakage-safe by construction
# ---------------------------------------------------------------------------

def compute_wallet_rolling_features(
    transactions_df: pd.DataFrame, as_of: pd.Timestamp, window_hours: int = 24,
) -> pd.DataFrame:
    """
    Aggregates per-wallet behaviour using ONLY transactions strictly before
    `as_of`, within the trailing `window_hours`. This is the single choke
    point that prevents look-ahead bias for batch/backfill feature building
    — every caller MUST go through this cutoff, there's no code path here
    that can see a transaction dated >= as_of.

    window boundary is [as_of - window_hours, as_of) — right-open, so a
    transaction happening AT exactly `as_of` is correctly excluded (it's
    the "current" transaction being scored, not history).
    """
    window_start = as_of - pd.Timedelta(hours=window_hours)

    # strict inequalities on BOTH sides — the whole point of this function
    in_window = transactions_df[
        (transactions_df["timestamp"] >= window_start) & (transactions_df["timestamp"] < as_of)
    ]

    if in_window.empty:
        return pd.DataFrame(columns=["wallet", "mean_log_amount", "std_log_amount", "tx_count_in_window"])

    feats = in_window.copy()
    feats["log_amount"] = np.log1p(feats["amount_usd"])

    agg = feats.groupby("wallet").agg(
        mean_log_amount=("log_amount", "mean"),
        std_log_amount=("log_amount", "std"),
        tx_count_in_window=("log_amount", "count"),
    ).reset_index()
    agg["std_log_amount"] = agg["std_log_amount"].fillna(0)  # single-tx wallet: real 0 spread, not missing

    return agg


# ---------------------------------------------------------------------------
# Correlation / redundancy visibility (informational, no auto-dropping)
# ---------------------------------------------------------------------------

def feature_correlation_report(features_df: pd.DataFrame, threshold: float = 0.9) -> List[dict]:
    """
    Returns pairs of numeric features with |correlation| >= threshold.
    Informational only — deciding what to drop is a modeling judgment call,
    not something this function makes for you.
    """
    numeric = features_df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []

    corr = numeric.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and val >= threshold:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": round(float(val), 4)})
    return sorted(pairs, key=lambda p: -p["correlation"])


# ---------------------------------------------------------------------------
# Drift instrumentation (detection support, not correction)
# ---------------------------------------------------------------------------

def snapshot_feature_distribution(features_df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """
    Records summary statistics for numeric features at a point in time.
    Compare two snapshots taken weeks apart to SEE drift — this function
    does not correct for drift, it makes drift measurable.
    """
    numeric = features_df.select_dtypes(include=[np.number])
    snapshot = {"as_of": str(as_of), "n_rows": len(features_df), "features": {}}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            snapshot["features"][col] = None
            continue
        snapshot["features"][col] = {
            "mean": round(float(series.mean()), 6),
            "std": round(float(series.std()), 6) if len(series) > 1 else 0.0,
            "p50": round(float(series.median()), 6),
            "p99": round(float(series.quantile(0.99)), 6),
        }
    return snapshot


if __name__ == "__main__":
    # --- 1. per-transaction features, including missing-gas-fee case ---
    print("--- Per-transaction features ---")
    tx_ok = {"tx_hash": "0x1", "amount_usd": 150_000, "gas_fee_usd": 45.0}
    tx_missing_gas = {"tx_hash": "0x2", "amount_usd": 90_000, "gas_fee_usd": None}
    print(compute_transaction_features(tx_ok))
    print(compute_transaction_features(tx_missing_gas))
    assert compute_transaction_features(tx_missing_gas)["log_gas_fee"] is None, "must not fabricate a gas fee"

    # --- 2. incremental stats: first-ever tx for a wallet must show None, not 0 ---
    print("\n--- Incremental wallet stats (real-time path) ---")
    tracker = WalletStatsTracker()
    t0 = pd.Timestamp("2026-08-01 00:00")
    r1 = tracker.process_transaction("0xW1", 100_000, t0)
    print("first tx:", r1)
    assert r1["wallet_time_delta_hours"] is None, "first-ever tx must not claim a time delta"
    assert r1["wallet_prior_mean_log_amount"] is None, "first-ever tx must not claim prior history"

    r2 = tracker.process_transaction("0xW1", 120_000, t0 + pd.Timedelta(hours=3))
    print("second tx:", r2)
    assert r2["wallet_time_delta_hours"] == 3.0

    # --- 3. leakage guard: rolling features must never see the future ---
    print("\n--- Leakage guard test ---")
    df = pd.DataFrame([
        {"wallet": "0xA", "amount_usd": 100_000, "timestamp": pd.Timestamp("2026-08-01 10:00")},
        {"wallet": "0xA", "amount_usd": 200_000, "timestamp": pd.Timestamp("2026-08-01 11:00")},
        {"wallet": "0xA", "amount_usd": 9_000_000, "timestamp": pd.Timestamp("2026-08-01 15:00")},  # future spike
    ])
    as_of = pd.Timestamp("2026-08-01 12:00")  # BEFORE the 15:00 spike
    feats = compute_wallet_rolling_features(df, as_of=as_of, window_hours=24)
    print(feats)
    seen_tx_count = feats.loc[feats["wallet"] == "0xA", "tx_count_in_window"].iloc[0]
    assert seen_tx_count == 2, f"expected 2 (future spike excluded), got {seen_tx_count} — LEAKAGE BUG"
    print("Confirmed: the 15:00 future transaction was correctly excluded from a 12:00 as_of calculation.")

    # --- 4. correlation report ---
    print("\n--- Correlation report ---")
    corr_df = pd.DataFrame({
        "amount_usd": [100, 200, 300, 400],
        "log_amount": [4.6, 5.3, 5.7, 6.0],   # near-perfectly correlated with amount_usd on purpose
        "gas_fee_usd": [10, 55, 12, 48],       # unrelated
    })
    print(feature_correlation_report(corr_df, threshold=0.9))

    # --- 5. drift snapshot ---
    print("\n--- Drift snapshot ---")
    snap = snapshot_feature_distribution(corr_df, as_of=pd.Timestamp("2026-08-01"))
    print(snap)

    print("\nAll assertions passed.")
