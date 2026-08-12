"""
live_loop.py — the REAL always-on entrypoint for this project.

This replaces the "live_loop.py" referenced in the deploy guide you were
given — that guide's file doesn't match what we've actually built here.
This version wires together the real Step 1/2/3 modules:

    Step 1 (MultiChainCollector.run_forever)
        -> collects live transactions from Ethereum/Bitcoin/Solana
    Step 2 (TransactionStore.store_transactions)
        -> every batch is written straight to the DB (SQLite or DuckDB,
           picked via the DATABASE_URL environment variable)
    Step 3 (WhaleIdentifyEngine, non-ML strategies only)
        -> SimpleThreshold + DynamicThreshold + WalletAggregation +
           Convergence run on each batch; signals are logged (wire a
           Telegram/Discord webhook in `handle_signals()` if you want alerts)

Environment variables (set these in Railway/Render's dashboard, NOT
hardcoded — see the warning in the guide you were given, which is correct
on this point):
    DATABASE_URL   e.g. "sqlite:///./whale.db" or "duckdb:///./whale.duckdb"
    CHAINS         comma-separated, e.g. "ethereum,bitcoin" (default: ethereum)
    POLL_INTERVAL  seconds between polls (default: 15)
"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Dict, List

from step1_data_collect import MultiChainCollector, EthereumAdapter, BitcoinAdapter, SolanaAdapter, PriceFeed
from step2_transaction_store import TransactionStore
from step3_whale_identify import (
    WhaleIdentifyEngine, SimpleThresholdStrategy, DynamicThresholdStrategy,
    WalletAggregationStrategy, ConvergenceStrategy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("live_loop")


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./whale.db")
CHAINS = [c.strip() for c in os.environ.get("CHAINS", "ethereum").split(",") if c.strip()]
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "15"))
ETH_RPC_URL = os.environ.get("ETH_RPC_URL", "https://rpc.ankr.com/eth")
BTC_API_URL = os.environ.get("BTC_API_URL", "https://blockstream.info/api")
SOL_RPC_URL = os.environ.get("SOL_RPC_URL", "https://api.mainnet-beta.solana.com")

def build_collector() -> MultiChainCollector:
    price_feed = PriceFeed()
    adapters = []
    if "ethereum" in CHAINS:
        adapters.append(EthereumAdapter(price_feed, rpc_url=ETH_RPC_URL))
    if "bitcoin" in CHAINS:
        adapters.append(BitcoinAdapter(price_feed, base_url=BTC_API_URL))
    if "solana" in CHAINS:
        adapters.append(SolanaAdapter(price_feed, rpc_url=SOL_RPC_URL))
    if not adapters:
        raise ValueError(f"No recognized chains in CHAINS={CHAINS!r} — expected some of ethereum,bitcoin,solana")
    return MultiChainCollector(adapters)


def build_engine() -> WhaleIdentifyEngine:
    return WhaleIdentifyEngine([
        SimpleThresholdStrategy({"min_trade_value": 100_000}),
        DynamicThresholdStrategy({"percentile": 99, "min_window_size": 30}),
        WalletAggregationStrategy({
            "window_seconds": 3600, "gross_threshold": 1_500_000,
            "net_fraction_wash": 0.15, "cooldown_seconds": 900,
        }),
        ConvergenceStrategy({}),  # defaults from step3 — tune here if needed
    ])


# rolling per-chain trade-value history, feeds DynamicThresholdStrategy
# without needing to re-query the DB on every batch
_trailing_values: Dict[str, deque] = {}
TRAILING_WINDOW_SIZE = 500  # keep last N trade values per chain in memory


def _build_market_data(chain: str, txs: List[dict]) -> dict:
    values = _trailing_values.setdefault(chain, deque(maxlen=TRAILING_WINDOW_SIZE))
    trades = []
    wallet_transactions = []

    for tx in txs:
        amount_usd = tx.get("amount_usd")
        if amount_usd is None:
            continue
        values.append(amount_usd)
        trades.append({"price": 1, "quantity": amount_usd, "side": "buy"})

        # WalletAggregationStrategy wants signed in/out legs per wallet —
        # a transfer is an "out" leg for the sender and an "in" leg for
        # the receiver, both derived from the SAME transaction here
        ts = tx["timestamp"].timestamp() if hasattr(tx.get("timestamp"), "timestamp") else time.time()
        if tx.get("from_address"):
            wallet_transactions.append({
                "chain": chain, "wallet": tx["from_address"], "side": "out",
                "amount_usd": amount_usd, "timestamp": ts,
            })
        if tx.get("to_address"):
            wallet_transactions.append({
                "chain": chain, "wallet": tx["to_address"], "side": "in",
                "amount_usd": amount_usd, "timestamp": ts,
            })

    return {
        "chain": chain, "symbol": chain,
        "trades": trades, "trailing_trade_values": list(values),
        "wallet_transactions": wallet_transactions,
    }


def handle_signals(chain: str, signals) -> None:
    """
    Every non-ML Step 3 strategy's output lands here. Wire a Telegram/
    Discord webhook call in this function if you want real alerts — kept
    as a log line for now so this file has no external-service dependency
    beyond the chain RPC/price feed itself.
    """
    for s in signals:
        logger.info(
            f"[SIGNAL] chain={chain} type={s.signal_type} value=${s.value:,.0f} "
            f"strategy={s.metadata.get('strategy')} meta={s.metadata}"
        )


async def main():
    logger.info(f"Starting live_loop — chains={CHAINS} db={DATABASE_URL} poll_interval={POLL_INTERVAL}s")

    store = TransactionStore(DATABASE_URL)
    engine = build_engine()
    collector = build_collector()

    async def on_batch(chain: str, txs: List[dict]):
        if not txs:
            return

        result = store.store_transactions(txs)
        logger.info(f"[{chain}] stored batch: {result} ({len(txs)} txs from this block)")

        market_data = _build_market_data(chain, txs)
        signals = engine.run(market_data)
        if signals:
            handle_signals(chain, signals)

    await collector.run_forever(poll_interval_seconds=POLL_INTERVAL, on_batch=on_batch)


if __name__ == "__main__":
    asyncio.run(main())
