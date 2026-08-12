"""
Step 3 — Whale Identify (No ML)
--------------------------------
Pluggable BaseWhaleStrategy architecture. Every "score" here is an honestly
labeled rule-based heuristic (threshold_score), never called "confidence" —
confidence implies a trained model, and there is none in this file.

Strategies, in order of what they catch:

  1. SimpleThresholdStrategy   — single tx >= static $ amount. Fast, cheap,
                                  but a whale who splits transfers walks
                                  straight under it.

  2. DynamicThresholdStrategy  — single tx >= rolling Nth percentile of
                                  recent trade sizes. Adapts to market
                                  conditions, still per-transaction, still
                                  blind to splitting.

  3. WalletAggregationStrategy — sums a wallet's activity over a sliding
                                  time window (chain-scoped), tracking BOTH:
                                    - gross  = sum of |amount| (total movement)
                                    - net    = sum of signed amount (in - out)
                                  Gross catches structuring (many small legs
                                  adding up to a big move). Net separates a
                                  real directional whale move from a wash /
                                  round-trip (send out, receive back — gross
                                  spikes, net stays near zero).

  4. ConvergenceStrategy       — catches the OTHER split pattern: many
                                  distinct wallets funneling into one
                                  collection wallet in a short window
                                  (structuring on the receiving side).
                                  WalletAggregationStrategy alone won't
                                  reliably catch this because each SENDING
                                  wallet only sends once — nothing looks
                                  unusual on the sender side.

Fixed vs an earlier draft that was reviewed and found broken on 4 points:
  - Was a TUMBLING window (resets on a fixed clock) → replaced with a real
    SLIDING window (deque, prunes anything older than window_seconds on
    every event) so a burst that straddles an old window boundary is still
    caught.
  - Tracked only one side ('from' OR 'to', picked by an unclear comment) and
    called it net when it was gross → now tracks signed net AND gross
    separately, explicitly, for both directions.
  - Reset the running total to 0 right after alerting → that's a loophole
    (trigger one small alert on purpose, then accumulate un-watched). Fixed
    with a per-key cooldown that suppresses repeat alerts without touching
    the underlying sliding-window sum.
  - Aggregation key was wallet_address alone → collisions possible across
    chains. Fixed to (chain, wallet_address).

No ML (no sklearn/pandas/numpy). Pure stdlib: time, collections.deque,
statistics. Every strategy is a heuristic hint for a human to review, not a
verdict — say so in the metadata rather than implying certainty.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import statistics
import time


# ---------------------------------------------------------------------------
# Shared signal type
# ---------------------------------------------------------------------------

@dataclass
class WhaleSignal:
    symbol: str
    signal_type: str            # "WHALE_BUY" / "WHALE_SELL" / "AGGREGATED_MOVE" /
                                 # "WASH_SUSPECT" / "CONVERGENCE"
    value: float                 # USD value most relevant to this signal
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    threshold_score: Optional[float] = None  # rule-based, NEVER a model confidence
    chain: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "chain": self.chain,
            "signal_type": self.signal_type,
            "value": self.value,
            "threshold_score": self.threshold_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class BaseWhaleStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def analyze(self, market_data: Dict[str, Any]) -> List[WhaleSignal]:
        """Return zero or more WhaleSignals for this batch of market data."""
        ...

    def is_enabled(self) -> bool:
        return self.enabled


# ---------------------------------------------------------------------------
# Strategy 1: static per-transaction threshold
# ---------------------------------------------------------------------------

class SimpleThresholdStrategy(BaseWhaleStrategy):
    """Fast first-pass filter. Misses anything split below min_trade_value."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_trade_value = config.get("min_trade_value", 100_000)

    def analyze(self, market_data: Dict[str, Any]) -> List[WhaleSignal]:
        signals = []
        chain = market_data.get("chain", "UNKNOWN")
        for trade in market_data.get("trades", []):
            value = trade.get("price", 0) * trade.get("quantity", 0)
            if value >= self.min_trade_value:
                score = min(1.0, 0.7 + value / 3_000_000)  # hand-picked formula
                signals.append(WhaleSignal(
                    symbol=market_data.get("symbol", "UNKNOWN"),
                    chain=chain,
                    signal_type="WHALE_BUY" if trade.get("side") == "buy" else "WHALE_SELL",
                    value=value,
                    threshold_score=round(score, 4),
                    metadata={"strategy": self.name, "rule": f">= ${self.min_trade_value:,}"},
                ))
        return signals


# ---------------------------------------------------------------------------
# Strategy 2: dynamic threshold — rolling percentile, no numpy
# ---------------------------------------------------------------------------

class DynamicThresholdStrategy(BaseWhaleStrategy):
    """
    Threshold adapts to recent market conditions instead of a fixed number.
    Caller supplies market_data["trailing_trade_values"] — e.g. last N hours
    of trade sizes for this symbol.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.percentile = config.get("percentile", 99)
        self.min_window_size = config.get("min_window_size", 30)

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        s = sorted(values)
        if not s:
            return 0.0
        k = (len(s) - 1) * (pct / 100.0)
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    def analyze(self, market_data: Dict[str, Any]) -> List[WhaleSignal]:
        history = market_data.get("trailing_trade_values", [])
        trades = market_data.get("trades", [])
        chain = market_data.get("chain", "UNKNOWN")
        signals = []

        if len(history) < self.min_window_size:
            return signals  # not enough data for a meaningful dynamic threshold

        dynamic_threshold = self._percentile(history, self.percentile)
        if dynamic_threshold <= 0:
            return signals

        for trade in trades:
            value = trade.get("price", 0) * trade.get("quantity", 0)
            if value >= dynamic_threshold:
                signals.append(WhaleSignal(
                    symbol=market_data.get("symbol", "UNKNOWN"),
                    chain=chain,
                    signal_type="WHALE_BUY" if trade.get("side") == "buy" else "WHALE_SELL",
                    value=value,
                    threshold_score=round(value / dynamic_threshold, 4),
                    metadata={
                        "strategy": self.name,
                        "dynamic_threshold": round(dynamic_threshold, 2),
                        "percentile": self.percentile,
                    },
                ))
        return signals


# ---------------------------------------------------------------------------
# Strategy 3: wallet aggregation — sliding window, chain-scoped, gross + net
# ---------------------------------------------------------------------------

class WalletAggregationStrategy(BaseWhaleStrategy):
    """
    Catches a whale splitting ONE wallet's activity into many small legs
    (structuring / smurfing) so it never trips a single-tx threshold.

    Each incoming transaction is expected as a dict with at minimum:
        {"chain": str, "wallet": str, "side": "in"|"out",
         "amount_usd": float, "timestamp": float}

    For every (chain, wallet), keeps a sliding window (deque) of recent
    signed amounts (+amount for "in", -amount for "out"). On each event it
    prunes anything older than window_seconds, then computes:
        gross = sum(|amount|)   over the window   -> total movement
        net   = sum(amount)     over the window    -> net directional flow

    Alert logic:
      - gross >= gross_threshold and |net| is a small fraction of gross
            -> WASH_SUSPECT (money churned back and forth, not really moved)
      - gross >= gross_threshold and |net| is a large fraction of gross
            -> AGGREGATED_MOVE (real accumulation/distribution split across
               many small transactions)

    A per-(chain, wallet) cooldown suppresses repeat alerts for the same
    ongoing burst WITHOUT resetting the sliding-window sum — so there's no
    "trigger a small alert on purpose to flush the counter" loophole.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_seconds = config.get("window_seconds", 3600)       # 1 hour default
        self.gross_threshold = config.get("gross_threshold", 1_500_000)
        self.net_fraction_wash = config.get("net_fraction_wash", 0.15)  # |net|/gross below this = wash
        self.cooldown_seconds = config.get("cooldown_seconds", 900)     # 15 min between repeat alerts
        self._windows: Dict[Tuple[str, str], deque] = {}
        self._last_alert: Dict[Tuple[str, str], float] = {}

    def _prune(self, key: Tuple[str, str], now: float) -> deque:
        dq = self._windows.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        return dq

    def analyze(self, market_data: Dict[str, Any]) -> List[WhaleSignal]:
        signals = []
        symbol = market_data.get("symbol", "UNKNOWN")
        for tx in market_data.get("wallet_transactions", []):
            chain = tx.get("chain", "UNKNOWN")
            wallet = tx.get("wallet")
            side = tx.get("side")  # "in" or "out"
            amount = float(tx.get("amount_usd", 0))
            ts = float(tx.get("timestamp", time.time()))
            if not wallet or amount <= 0:
                continue

            key = (chain, wallet)
            dq = self._prune(key, ts)
            signed = amount if side == "in" else -amount
            dq.append((ts, signed))

            gross = sum(abs(v) for _, v in dq)
            net = sum(v for _, v in dq)

            if gross < self.gross_threshold:
                continue

            last = self._last_alert.get(key, 0.0)
            if ts - last < self.cooldown_seconds:
                continue  # already alerted on this burst recently; sum keeps accumulating regardless

            net_fraction = abs(net) / gross if gross else 0.0
            is_wash = net_fraction < self.net_fraction_wash

            signals.append(WhaleSignal(
                symbol=symbol,
                chain=chain,
                signal_type="WASH_SUSPECT" if is_wash else "AGGREGATED_MOVE",
                value=round(gross, 2),
                threshold_score=round(gross / self.gross_threshold, 4),
                metadata={
                    "strategy": self.name,
                    "wallet": wallet,
                    "window_seconds": self.window_seconds,
                    "gross_usd": round(gross, 2),
                    "net_usd": round(net, 2),
                    "net_fraction_of_gross": round(net_fraction, 4),
                    "tx_count_in_window": len(dq),
                    "note": (
                        "High gross, low net — looks like funds churned back and "
                        "forth (round-trip / wash), not a real directional move. "
                        "Verify manually."
                        if is_wash else
                        "High gross AND high net — consistent with a large move "
                        "split into many small legs to dodge a per-tx threshold. "
                        "Verify manually, this is a heuristic, not a verdict."
                    ),
                },
            ))
            self._last_alert[key] = ts
        return signals


# ---------------------------------------------------------------------------
# Strategy 4: convergence — many distinct wallets funneling into one wallet
# ---------------------------------------------------------------------------

class ConvergenceStrategy(BaseWhaleStrategy):
    """
    Catches the OTHER split pattern that WalletAggregationStrategy can't see:
    a whale (or an exchange deposit pipeline, which is also worth flagging
    for review) collecting funds INTO one wallet FROM many distinct source
    wallets in a short window. Each sender only transacts once, so nothing
    looks unusual on the sender side — only visible by looking at the
    receiving wallet's inbound diversity.

    Expects the same wallet_transactions shape as WalletAggregationStrategy,
    keyed on the receiving wallet for "in" transactions only.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_seconds = config.get("window_seconds", 1800)  # 30 min default
        self.min_distinct_senders = config.get("min_distinct_senders", 8)
        self.total_threshold = config.get("total_threshold", 500_000)
        self._windows: Dict[Tuple[str, str], deque] = {}
        self._last_alert: Dict[Tuple[str, str], float] = {}
        self.cooldown_seconds = config.get("cooldown_seconds", 1800)

    def _prune(self, key: Tuple[str, str], now: float) -> deque:
        dq = self._windows.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        return dq

    def analyze(self, market_data: Dict[str, Any]) -> List[WhaleSignal]:
        signals = []
        symbol = market_data.get("symbol", "UNKNOWN")
        for tx in market_data.get("wallet_transactions", []):
            if tx.get("side") != "in":
                continue
            chain = tx.get("chain", "UNKNOWN")
            to_wallet = tx.get("wallet")
            from_wallet = tx.get("counterparty")
            amount = float(tx.get("amount_usd", 0))
            ts = float(tx.get("timestamp", time.time()))
            if not to_wallet or not from_wallet or amount <= 0:
                continue

            key = (chain, to_wallet)
            dq = self._prune(key, ts)
            dq.append((ts, from_wallet, amount))

            distinct_senders = {w for _, w, _ in dq}
            total = sum(a for _, _, a in dq)

            if len(distinct_senders) < self.min_distinct_senders or total < self.total_threshold:
                continue

            last = self._last_alert.get(key, 0.0)
            if ts - last < self.cooldown_seconds:
                continue

            signals.append(WhaleSignal(
                symbol=symbol,
                chain=chain,
                signal_type="CONVERGENCE",
                value=round(total, 2),
                threshold_score=round(total / self.total_threshold, 4),
                metadata={
                    "strategy": self.name,
                    "collection_wallet": to_wallet,
                    "distinct_senders": len(distinct_senders),
                    "window_seconds": self.window_seconds,
                    "sample_senders": list(distinct_senders)[:5],
                    "note": (
                        "Many distinct wallets funneled funds into one wallet in a "
                        "short window. Could be a whale consolidating split funds, "
                        "or an exchange hot-wallet deposit sweep. Verify manually."
                    ),
                },
            ))
            self._last_alert[key] = ts
        return signals


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class WhaleIdentifyEngine:
    def __init__(self, strategies: List[BaseWhaleStrategy]):
        self.strategies = [s for s in strategies if s.is_enabled()]

    def run(self, market_data: Dict[str, Any]) -> List[WhaleSignal]:
        signals = []
        for strategy in self.strategies:
            try:
                signals.extend(strategy.analyze(market_data))
            except Exception as e:
                # one bad strategy must not take down the others
                print(f"[WhaleIdentifyEngine] {strategy.name} failed: {e}")
        return signals


if __name__ == "__main__":
    import random
    random.seed(42)

    now = time.time()
    wallet_txs = []

    # --- Case 1: structuring — one wallet splits ~2M into 20 small OUT legs,
    #             each individually under any sane single-tx threshold.
    for i in range(20):
        wallet_txs.append({
            "chain": "BTC", "wallet": "0xStructurer", "side": "out",
            "amount_usd": round(random.uniform(80_000, 120_000), 2),
            "timestamp": now - random.uniform(0, 3000),  # all inside 1hr window
            "counterparty": f"0xDump{i:02d}",
        })

    # --- Case 2: wash / round-trip — sends 1M out then gets ~1M back in.
    #             gross should spike, net should stay near zero.
    wallet_txs.append({"chain": "ETH", "wallet": "0xWasher", "side": "out",
                        "amount_usd": 1_000_000, "timestamp": now - 200,
                        "counterparty": "0xTemp"})
    wallet_txs.append({"chain": "ETH", "wallet": "0xWasher", "side": "in",
                        "amount_usd": 980_000, "timestamp": now - 50,
                        "counterparty": "0xTemp"})

    # --- Case 3: convergence — 12 distinct wallets each send a moderate
    #             amount into one collection wallet within 30 minutes.
    for i in range(12):
        wallet_txs.append({
            "chain": "SOL", "wallet": "0xCollector", "side": "in",
            "amount_usd": round(random.uniform(50_000, 70_000), 2),
            "timestamp": now - random.uniform(0, 1500),
            "counterparty": f"0xFeeder{i:02d}",
        })

    # --- Case 4: same address string on a DIFFERENT chain — must NOT merge
    #             with Case 1's totals (chain-scoping check).
    wallet_txs.append({"chain": "ETH", "wallet": "0xStructurer", "side": "out",
                        "amount_usd": 50_000, "timestamp": now - 100,
                        "counterparty": "0xUnrelated"})

    engine = WhaleIdentifyEngine([
        SimpleThresholdStrategy({"min_trade_value": 500_000}),
        WalletAggregationStrategy({
            "window_seconds": 3600,
            "gross_threshold": 1_500_000,
            "net_fraction_wash": 0.15,
            "cooldown_seconds": 900,
        }),
        ConvergenceStrategy({
            "window_seconds": 1800,
            "min_distinct_senders": 8,
            "total_threshold": 500_000,
        }),
    ])

    signals = engine.run({
        "symbol": "MULTI",
        "trades": [],  # no single-tx trades big enough on purpose — proves aggregation is needed
        "wallet_transactions": wallet_txs,
    })

    print(f"Total signals: {len(signals)}\n")
    for s in signals:
        d = s.to_dict()
        print(f"[{d['signal_type']}] {d['chain']}/{d['symbol']} value=${d['value']:,.0f} "
              f"score={d['threshold_score']}")
        for k, v in d["metadata"].items():
            if k != "note":
                print(f"    {k}: {v}")
        print(f"    note: {d['metadata'].get('note', '')}\n")

    types = [s.signal_type for s in signals]
    print("Expect: one AGGREGATED_MOVE (0xStructurer/BTC), one WASH_SUSPECT (0xWasher/ETH), "
          "one CONVERGENCE (0xCollector/SOL). 0xStructurer/ETH (Case 4) should NOT trigger "
          "anything — proves chain-scoping works.")
    print("Signal types seen:", types)
