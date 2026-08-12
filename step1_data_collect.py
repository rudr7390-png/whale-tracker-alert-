"""
Step 1 — Data Collect (MULTI-CHAIN, free sources, all transactions)
------------------------------------------------------------------------------
Why this changed from the Ethereum-only version:
  Whale-tracking limited to one chain misses cross-chain wallet behavior
  entirely (a whale moving between ETH and BTC, or using BTC as a
  store-of-value while active on ETH, is invisible to a single-chain
  collector). The earlier CryptoWhaleMonitor reference project covered
  BTC/ETH/SOL/TON/XRP for the same reason.

Architecture: a `ChainAdapter` abstract interface, one implementation per
chain. This is the key design decision — Ethereum and Bitcoin do NOT share
a data model:

  Ethereum (account-based)
    - one sender, one receiver per transaction (mostly)
    - free source: any public JSON-RPC node (eth.llamarpc.com, etc.)

  Bitcoin (UTXO-based)
    - a transaction can have MANY inputs and MANY outputs — there is no
      single "from" or "to" the way Ethereum has. Modeling it as one
      simple from/to would be factually wrong, not just simplified.
    - free source: Blockstream's public Esplora REST API
      (https://blockstream.info/api) — no key, no signup, full history

Both adapters output the SAME normalized transaction shape so Steps 2-7
don't need to know which chain a record came from — but the normalization
is honest about the UTXO case (see BitcoinAdapter docstring) rather than
pretending it's a clean from/to like Ethereum.

Adding another chain (Solana, TON, XRP) = write one more ChainAdapter
subclass with the same 3 methods. A SolanaAdapter is included as a third,
lighter-weight example of the same pattern (account-based like Ethereum,
but a different RPC schema) to show how a third chain slots in.

Free sources used (all no-key, no-signup):
  Ethereum : https://eth.llamarpc.com                    (public RPC)
  Bitcoin  : https://blockstream.info/api                 (Esplora REST)
  Solana   : https://api.mainnet-beta.solana.com           (public RPC)
  Price    : https://api.coingecko.com/api/v3               (free tier)
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None  # only required for actual network calls; offline parsing/tests don't need it

logger = logging.getLogger("data_collect_multichain")

COINGECKO_SIMPLE_PRICE = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_HISTORY_TMPL = "https://api.coingecko.com/api/v3/coins/{coin_id}/history"

WEI_PER_ETH = 10**18
SATOSHI_PER_BTC = 10**8
LAMPORTS_PER_SOL = 10**9


# ---------------------------------------------------------------------------
# Shared HTTP helper — retry/backoff/rate-limit handling, reused by every
# adapter regardless of whether it speaks JSON-RPC or plain REST
# ---------------------------------------------------------------------------

async def http_get_json(url: str, params: dict = None, max_retries: int = 5) -> Optional[dict]:
    if httpx is None:
        raise RuntimeError("httpx is required for network calls — pip install httpx")

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(f"Rate limited on {url} — waiting {wait:.1f}s (attempt {attempt}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            wait = 2 ** (attempt - 1) + random.uniform(0, 0.5)
            logger.warning(f"Network error on {url}: {e} — retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                wait = 2 ** (attempt - 1) + random.uniform(0, 0.5)
                logger.warning(f"Server error {e.response.status_code} on {url} — retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
            else:
                logger.error(f"Non-retryable HTTP error on {url}: {e}")
                return None
    logger.error(f"Giving up on {url} after {max_retries} attempts")
    return None


async def json_rpc_call(rpc_url: str, method: str, params: list, max_retries: int = 5) -> Optional[dict]:
    if httpx is None:
        raise RuntimeError("httpx is required for network calls — pip install httpx")

    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(rpc_url, json=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    msg = str(data["error"]).lower()
                    if "limit" in msg or "rate" in msg:
                        wait = 2 ** attempt + random.uniform(0, 1)
                        logger.warning(f"RPC rate-limited on '{method}' — waiting {wait:.1f}s")
                        await asyncio.sleep(wait)
                        continue
                    logger.error(f"RPC error on '{method}': {data['error']}")
                    return None
                return data.get("result")
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            wait = 2 ** (attempt - 1) + random.uniform(0, 0.5)
            logger.warning(f"RPC call '{method}' failed: {e} — retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    logger.error(f"Giving up on RPC '{method}' after {max_retries} attempts")
    return None


def _hex_to_int(h: Optional[str]) -> Optional[int]:
    return int(h, 16) if h is not None else None


# ---------------------------------------------------------------------------
# Price feed — shared across all chains (CoinGecko covers ETH/BTC/SOL/etc.)
# ---------------------------------------------------------------------------

class PriceFeed:
    """
    Live price via CoinGecko free `/simple/price`. Historical price via
    `/coins/{id}/history` is DAILY granularity only (a free-tier limit, not
    per-block precision) — every converted amount carries a
    `price_precision` flag so nothing downstream mistakes daily-average for
    exact.
    """

    COIN_IDS = {"ethereum": "ethereum", "bitcoin": "bitcoin", "solana": "solana"}

    def __init__(self):
        self._live_cache: Dict[str, tuple[float, float]] = {}
        self._history_cache: Dict[str, float] = {}

    async def get_live_price_usd(self, chain: str, max_age_seconds: int = 60) -> Optional[float]:
        coin_id = self.COIN_IDS.get(chain)
        if not coin_id:
            return None

        now = time.time()
        cached = self._live_cache.get(coin_id)
        if cached and (now - cached[1]) < max_age_seconds:
            return cached[0]

        data = await http_get_json(COINGECKO_SIMPLE_PRICE, {"ids": coin_id, "vs_currencies": "usd"})
        if not data or coin_id not in data:
            return None
        price = data[coin_id]["usd"]
        self._live_cache[coin_id] = (price, now)
        return price

    async def get_historical_price_usd(self, chain: str, dt: datetime) -> Optional[float]:
        coin_id = self.COIN_IDS.get(chain)
        if not coin_id:
            return None

        date_str = dt.strftime("%d-%m-%Y")
        cache_key = f"{coin_id}:{date_str}"
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]

        url = COINGECKO_HISTORY_TMPL.format(coin_id=coin_id)
        data = await http_get_json(url, {"date": date_str})
        if not data:
            return None
        try:
            price = data["market_data"]["current_price"]["usd"]
        except (KeyError, TypeError):
            return None
        self._history_cache[cache_key] = price
        return price


# ---------------------------------------------------------------------------
# Normalized transaction shape — same keys regardless of chain, but honest
# about which fields don't cleanly apply to a given chain's data model
# ---------------------------------------------------------------------------

def make_normalized_tx(
    *, chain: str, txid: str, from_address: Optional[str], to_address: Optional[str],
    amount: float, amount_usd: Optional[float], fee_native: Optional[float],
    fee_usd: Optional[float], block_number, timestamp: datetime,
    price_usd: Optional[float], price_precision: str,
    account_model: str,  # "account" (ETH-like) or "utxo" (BTC-like)
    extra: Optional[dict] = None,
) -> dict:
    return {
        "chain": chain,
        "txid": txid,
        "from_address": from_address,   # UTXO chains: primary input only, see BitcoinAdapter docstring
        "to_address": to_address,        # UTXO chains: primary (largest) output only
        "amount": amount,
        "amount_usd": amount_usd,
        "gas_fee_native": fee_native,     # renamed generically — "fee" makes more sense for BTC than "gas"
        "gas_fee_usd": fee_usd,
        "block_hash_or_number": str(block_number),
        "block_number": block_number,
        "timestamp": timestamp,
        "price_usd": price_usd,
        "price_precision": price_precision,
        "account_model": account_model,  # tells downstream steps how to interpret from/to
        "classification": None,           # decided later, by Step 3 — not this layer's job
        "extra": extra or {},              # full input/output lists for UTXO chains, etc.
    }


# ---------------------------------------------------------------------------
# ChainAdapter interface
# ---------------------------------------------------------------------------

class ChainAdapter(ABC):
    chain_name: str
    native_symbol: str
    account_model: str  # "account" | "utxo"

    def __init__(self, price_feed: PriceFeed):
        self.price_feed = price_feed

    @abstractmethod
    async def get_latest_block_number(self) -> Optional[int]:
        ...

    @abstractmethod
    async def fetch_block(self, block_number) -> Optional[dict]:
        """Returns raw, chain-native block data (schema differs per chain)."""
        ...

    @abstractmethod
    async def parse_block(self, block: dict, use_live_price: bool) -> List[dict]:
        """Returns a list of normalized tx dicts (see make_normalized_tx)."""
        ...


# ---------------------------------------------------------------------------
# Ethereum adapter (account-based) — also works for any EVM chain by
# swapping rpc_url/chain_name/native_symbol (Polygon, BSC, Arbitrum, etc.)
# ---------------------------------------------------------------------------

class EthereumAdapter(ChainAdapter):
    account_model = "account"

    def __init__(self, price_feed: PriceFeed, rpc_url: str = "https://eth.llamarpc.com",
                 chain_name: str = "ethereum", native_symbol: str = "ETH"):
        super().__init__(price_feed)
        self.rpc_url = rpc_url
        self.chain_name = chain_name
        self.native_symbol = native_symbol

    async def get_latest_block_number(self) -> Optional[int]:
        result = await json_rpc_call(self.rpc_url, "eth_blockNumber", [])
        return _hex_to_int(result) if result else None

    async def fetch_block(self, block_number: int) -> Optional[dict]:
        return await json_rpc_call(self.rpc_url, "eth_getBlockByNumber", [hex(block_number), True])

    async def _fetch_receipts(self, tx_hashes: List[str]) -> Dict[str, Optional[dict]]:
        # batched into one HTTP call — same pattern as the earlier ETH-only version
        if not tx_hashes or httpx is None:
            return {}
        payload = [
            {"jsonrpc": "2.0", "id": i, "method": "eth_getTransactionReceipt", "params": [h]}
            for i, h in enumerate(tx_hashes)
        ]
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.rpc_url, json=payload, timeout=30)
            resp.raise_for_status()
            results = resp.json()
        out = {}
        for item, h in zip(results, tx_hashes):
            out[h] = item.get("result") if "error" not in item else None
        return out

    async def parse_block(self, block: dict, use_live_price: bool) -> List[dict]:
        block_number = _hex_to_int(block["number"])
        block_dt = datetime.fromtimestamp(_hex_to_int(block["timestamp"]), tz=timezone.utc)
        tx_hashes = [tx["hash"] for tx in block.get("transactions", [])]
        receipts = await self._fetch_receipts(tx_hashes)

        if use_live_price:
            price = await self.price_feed.get_live_price_usd(self.chain_name)
            precision = "live" if price else "unknown"
        else:
            price = await self.price_feed.get_historical_price_usd(self.chain_name, block_dt)
            precision = "daily_historical" if price else "unknown"

        return parse_ethereum_block_transactions(block, receipts, price, precision, self.chain_name)


def parse_ethereum_block_transactions(
    block: dict, receipts_by_hash: Dict[str, Optional[dict]],
    price_usd: Optional[float], price_precision: str, chain_name: str,
) -> List[dict]:
    """Kept as a standalone function (not a method) so it's independently testable offline."""
    block_number = _hex_to_int(block["number"])
    parsed = []
    for tx in block.get("transactions", []):
        value_wei = _hex_to_int(tx.get("value", "0x0"))
        amount_eth = value_wei / WEI_PER_ETH

        receipt = receipts_by_hash.get(tx["hash"])
        gas_used = _hex_to_int(receipt["gasUsed"]) if receipt and receipt.get("gasUsed") else None

        if "maxFeePerGas" in tx and tx.get("maxFeePerGas"):
            effective_gas_price_wei = _hex_to_int(
                receipt.get("effectiveGasPrice") if receipt else tx.get("maxFeePerGas")
            )
        else:
            effective_gas_price_wei = _hex_to_int(tx.get("gasPrice"))

        fee_eth = None
        if gas_used is not None and effective_gas_price_wei is not None:
            fee_eth = (gas_used * effective_gas_price_wei) / WEI_PER_ETH

        parsed.append(make_normalized_tx(
            chain=chain_name, txid=tx["hash"], from_address=tx["from"], to_address=tx.get("to"),
            amount=amount_eth, amount_usd=amount_eth * price_usd if price_usd else None,
            fee_native=fee_eth, fee_usd=fee_eth * price_usd if (fee_eth and price_usd) else None,
            block_number=block_number,
            timestamp=datetime.fromtimestamp(_hex_to_int(block["timestamp"]), tz=timezone.utc),
            price_usd=price_usd, price_precision=price_precision, account_model="account",
            extra={"gas_used": gas_used, "gas_price_gwei": effective_gas_price_wei / 10**9 if effective_gas_price_wei else None,
                   "is_contract_creation": tx.get("to") is None},
        ))
    return parsed


# ---------------------------------------------------------------------------
# Bitcoin adapter (UTXO-based) — free Blockstream Esplora REST API
# ---------------------------------------------------------------------------

class BitcoinAdapter(ChainAdapter):
    """
    Honest handling of the UTXO model: a Bitcoin transaction can spend
    MANY inputs and create MANY outputs. There is no single "sender" or
    "receiver" the way Ethereum has one.

    Design decision (documented, not hidden):
      from_address = the FIRST input's previous-output address (a
                       reasonable single-value stand-in, not "the" sender —
                       a tx can have inputs from several different addresses)
      to_address    = the LARGEST-value output's address (a reasonable
                       stand-in for "the" recipient — excludes obvious
                       change outputs better than picking the first output)
      amount        = TOTAL value moved (sum of outputs), not just the
                       primary one, so whale-size detection isn't
                       artificially shrunk by this simplification
      extra["inputs"] / extra["outputs"] = the FULL lists, so Step 4
        (wallet clustering) can still do proper multi-input-address
        clustering later without being limited by the simplified from/to
    """

    chain_name = "bitcoin"
    native_symbol = "BTC"
    account_model = "utxo"

    def __init__(self, price_feed: PriceFeed, base_url: str = "https://blockstream.info/api"):
        super().__init__(price_feed)
        self.base_url = base_url

    async def get_latest_block_number(self) -> Optional[int]:
        if httpx is None:
            raise RuntimeError("httpx is required for network calls")
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/blocks/tip/height", timeout=15)
            resp.raise_for_status()
            return int(resp.text)

    async def fetch_block(self, block_number: int) -> Optional[dict]:
        """
        Returns {"header": {...}, "transactions": [...]} — transactions are
        paginated by Esplora (25 per page), fetched here in full.
        """
        if httpx is None:
            raise RuntimeError("httpx is required for network calls")

        async with httpx.AsyncClient() as client:
            hash_resp = await client.get(f"{self.base_url}/block-height/{block_number}", timeout=15)
            hash_resp.raise_for_status()
            block_hash = hash_resp.text

            header_data = await http_get_json(f"{self.base_url}/block/{block_hash}")
            if not header_data:
                return None

            all_txs = []
            start_index = 0
            while True:
                page = await http_get_json(f"{self.base_url}/block/{block_hash}/txs/{start_index}")
                if not page:
                    break
                all_txs.extend(page)
                if len(page) < 25:
                    break
                start_index += 25
                await asyncio.sleep(0.2)  # be polite to the free public API between pages

        return {"header": header_data, "transactions": all_txs}

    async def parse_block(self, block: dict, use_live_price: bool) -> List[dict]:
        header = block["header"]
        block_number = header["height"]
        block_dt = datetime.fromtimestamp(header["timestamp"], tz=timezone.utc)

        if use_live_price:
            price = await self.price_feed.get_live_price_usd(self.chain_name)
            precision = "live" if price else "unknown"
        else:
            price = await self.price_feed.get_historical_price_usd(self.chain_name, block_dt)
            precision = "daily_historical" if price else "unknown"

        return parse_bitcoin_block_transactions(block, price, precision)


def parse_bitcoin_block_transactions(block: dict, price_usd: Optional[float], price_precision: str) -> List[dict]:
    """Standalone function, independently testable offline — same reasoning as the ETH parser."""
    header = block["header"]
    block_number = header["height"]
    block_dt = datetime.fromtimestamp(header["timestamp"], tz=timezone.utc)

    parsed = []
    for tx in block["transactions"]:
        vin = tx.get("vin", [])
        vout = tx.get("vout", [])

        # coinbase transactions (block reward) have no real "from" — handle explicitly, don't crash
        is_coinbase = any(v.get("is_coinbase") for v in vin)
        from_address = None
        if not is_coinbase and vin:
            from_address = vin[0].get("prevout", {}).get("scriptpubkey_address")

        total_output_sats = sum(v.get("value", 0) for v in vout)
        amount_btc = total_output_sats / SATOSHI_PER_BTC

        to_address = None
        if vout:
            largest_output = max(vout, key=lambda v: v.get("value", 0))
            to_address = largest_output.get("scriptpubkey_address")  # may be None for OP_RETURN/non-standard outputs

        fee_sats = tx.get("fee")  # Esplora provides this directly, no manual input/output subtraction needed
        fee_btc = fee_sats / SATOSHI_PER_BTC if fee_sats is not None else None

        parsed.append(make_normalized_tx(
            chain="bitcoin", txid=tx["txid"], from_address=from_address, to_address=to_address,
            amount=amount_btc, amount_usd=amount_btc * price_usd if price_usd else None,
            fee_native=fee_btc, fee_usd=fee_btc * price_usd if (fee_btc and price_usd) else None,
            block_number=block_number, timestamp=block_dt,
            price_usd=price_usd, price_precision=price_precision, account_model="utxo",
            extra={
                "is_coinbase": is_coinbase,
                "input_count": len(vin),
                "output_count": len(vout),
                "inputs": [v.get("prevout", {}).get("scriptpubkey_address") for v in vin if v.get("prevout")],
                "outputs": [{"address": v.get("scriptpubkey_address"), "value_sats": v.get("value")} for v in vout],
            },
        ))
    return parsed


# ---------------------------------------------------------------------------
# Solana adapter (account-based, third example — lighter-weight, shows how
# quickly a new chain slots into the same ChainAdapter interface)
# ---------------------------------------------------------------------------

class SolanaAdapter(ChainAdapter):
    """
    Account-based like Ethereum, but a different RPC schema (getBlock,
    not eth_getBlockByNumber) and fees work differently (mostly-fixed base
    fee per signature, not gas*price). Included to demonstrate the pattern
    for a third/future chain, not as deeply tested as ETH/BTC above.
    """
    chain_name = "solana"
    native_symbol = "SOL"
    account_model = "account"

    def __init__(self, price_feed: PriceFeed, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        super().__init__(price_feed)
        self.rpc_url = rpc_url

    async def get_latest_block_number(self) -> Optional[int]:
        return await json_rpc_call(self.rpc_url, "getSlot", [])

    async def fetch_block(self, block_number: int) -> Optional[dict]:
        return await json_rpc_call(self.rpc_url, "getBlock", [
            block_number,
            {"encoding": "json", "transactionDetails": "full", "maxSupportedTransactionVersion": 0},
        ])

    async def parse_block(self, block: dict, use_live_price: bool) -> List[dict]:
        if use_live_price:
            price = await self.price_feed.get_live_price_usd(self.chain_name)
            precision = "live" if price else "unknown"
        else:
            price = None
            precision = "unknown"  # Solana historical daily pricing omitted here — same CoinGecko call would apply
        return parse_solana_block_transactions(block, price, precision)


def parse_solana_block_transactions(block: dict, price_usd: Optional[float], price_precision: str) -> List[dict]:
    block_time = block.get("blockTime")
    block_dt = datetime.fromtimestamp(block_time, tz=timezone.utc) if block_time else datetime.now(timezone.utc)
    parsed = []

    for tx in block.get("transactions", []):
        meta = tx.get("meta", {})
        message = tx.get("transaction", {}).get("message", {})
        account_keys = message.get("accountKeys", [])
        if not account_keys:
            continue

        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        fee_lamports = meta.get("fee", 0)
        sig = tx.get("transaction", {}).get("signatures", [None])[0]

        # BUG (flagged by user, confirmed) — the old code indexed account_keys
        # using an index derived from pre/postBalances, on the assumption
        # those arrays are the same length as account_keys. That assumption
        # breaks for versioned (v0) transactions using Address Lookup
        # Tables: accounts loaded via a lookup table are NOT listed in
        # message.accountKeys, but ARE included in pre/postBalances —
        # so balances can be longer than account_keys, causing exactly the
        # reported `IndexError: list index out of range`.
        #
        # Fixed: reconstruct the FULL address list that actually matches
        # the balance-array ordering — Solana's own convention is static
        # accountKeys first, then loadedAddresses.writable, then
        # loadedAddresses.readonly (this ordering is part of the Solana
        # JSON-RPC spec, not a guess). meta.loadedAddresses is absent for
        # legacy (non-versioned) transactions, where accountKeys alone is
        # already correct — .get(..., []) handles that case with zero
        # behavior change.
        loaded = meta.get("loadedAddresses") or {}
        full_address_list = (
            list(account_keys) + list(loaded.get("writable", [])) + list(loaded.get("readonly", []))
        )

        address_list_complete = len(full_address_list) >= max(len(pre_balances), len(post_balances))
        if not address_list_complete:
            logger.warning(
                f"Solana tx {sig}: balance array longer than resolvable address list "
                f"(addresses={len(full_address_list)}, balances={len(post_balances)}) — "
                f"some indices won't be mapped to an address, handled safely below, not crashing"
            )

        # simplification: account [0] is always the fee payer/signer in Solana's
        # transaction format; net balance change identifies rough amount moved
        from_address = account_keys[0] if account_keys else None
        amount_lamports = 0
        to_address = None

        if len(pre_balances) == len(post_balances) and len(post_balances) > 1:
            deltas = [post - pre for pre, post in zip(pre_balances, post_balances)]

            # only consider candidates we can ACTUALLY map to a real address —
            # this bounds check is what prevents the IndexError outright,
            # independent of whether full_address_list reconstruction above
            # caught every case (defense in depth: even if loadedAddresses
            # was itself incomplete for some reason, this can't crash)
            candidates = [
                (i, deltas[i]) for i in range(1, len(deltas))
                if deltas[i] > 0 and i < len(full_address_list)
            ]
            if candidates:
                candidate_idx, best_delta = max(candidates, key=lambda pair: pair[1])
                to_address = full_address_list[candidate_idx]
                amount_lamports = best_delta

        amount_sol = amount_lamports / LAMPORTS_PER_SOL
        fee_sol = fee_lamports / LAMPORTS_PER_SOL

        parsed.append(make_normalized_tx(
            chain="solana", txid=sig,
            from_address=from_address, to_address=to_address, amount=amount_sol,
            amount_usd=amount_sol * price_usd if price_usd else None,
            fee_native=fee_sol, fee_usd=fee_sol * price_usd if (fee_sol and price_usd) else None,
            block_number=block.get("parentSlot", 0) + 1, timestamp=block_dt,
            price_usd=price_usd, price_precision=price_precision, account_model="account",
            extra={
                "account_keys_count": len(account_keys),
                "loaded_addresses_count": len(full_address_list) - len(account_keys),
                "address_list_complete": address_list_complete,
            },
        ))
    return parsed


# ---------------------------------------------------------------------------
# Multi-chain orchestrator
# ---------------------------------------------------------------------------

class MultiChainCollector:
    def __init__(self, adapters: List[ChainAdapter]):
        self.adapters = {a.chain_name: a for a in adapters}

    async def backfill_chain(self, chain: str, start_block, end_block, delay_between_blocks: float = 0.3) -> List[dict]:
        """
        Returns the FULL result in memory. Fine for a small range or a
        quick test — do NOT use this for a real historical backfill on
        Colab or any memory-constrained environment; a few thousand blocks
        of full (not whale-only) transaction data will blow past available
        RAM. Use backfill_chain_to_store() below instead for anything more
        than a quick sanity check.
        """
        adapter = self.adapters[chain]
        all_tx = []
        for block_number in range(start_block, end_block + 1):
            raw_block = await adapter.fetch_block(block_number)
            if raw_block is None:
                logger.warning(f"[{chain}] could not fetch block {block_number} — skipping")
                continue
            txs = await adapter.parse_block(raw_block, use_live_price=False)
            all_tx.extend(txs)
            await asyncio.sleep(delay_between_blocks)
        logger.info(f"[{chain}] backfilled {start_block}-{end_block}: {len(all_tx)} transactions")
        return all_tx

    async def backfill_chain_to_store(
        self, chain: str, start_block: int, end_block: int, store,
        delay_between_blocks: float = 0.3, checkpoint_callback=None,
    ) -> dict:
        """
        Streams each block straight into `store` (a Step 2 TransactionStore)
        one block at a time, instead of accumulating the whole range in a
        Python list first. This is the direct answer to "historical
        collection should write to DuckDB, not sit in Colab RAM" — memory
        usage here stays roughly constant (one block's worth of
        transactions at a time), regardless of whether start->end covers
        100 blocks or 100,000.

        `checkpoint_callback(block_number)` is called after each block
        commits successfully — wire this to persist "last completed block"
        somewhere (even a plain text file) so a crashed/interrupted backfill
        can resume from where it left off instead of restarting at
        start_block. Not doing this yourself is the main way a long
        Colab-session backfill silently loses hours of progress on a
        disconnect.

        Returns aggregate {"saved": int, "duplicates": int, "failed": int,
        "blocks_processed": int, "blocks_skipped": int} across the whole range.
        """
        adapter = self.adapters[chain]
        totals = {"saved": 0, "duplicates": 0, "failed": 0, "blocks_processed": 0, "blocks_skipped": 0}

        for block_number in range(start_block, end_block + 1):
            raw_block = await adapter.fetch_block(block_number)
            if raw_block is None:
                logger.warning(f"[{chain}] could not fetch block {block_number} — skipping")
                totals["blocks_skipped"] += 1
                await asyncio.sleep(delay_between_blocks)
                continue

            txs = await adapter.parse_block(raw_block, use_live_price=False)

            # write THIS block's transactions immediately, then let them be
            # garbage-collected — `txs` goes out of scope at the next loop
            # iteration instead of growing an all_tx list for the whole range
            result = store.store_transactions(txs)
            for key in ("saved", "duplicates", "failed"):
                totals[key] += result[key]
            totals["blocks_processed"] += 1

            if checkpoint_callback:
                checkpoint_callback(block_number)

            if block_number % 100 == 0:
                logger.info(f"[{chain}] progress: block {block_number}/{end_block} — running totals: {totals}")

            await asyncio.sleep(delay_between_blocks)

        logger.info(f"[{chain}] backfill-to-store complete for {start_block}-{end_block}: {totals}")
        return totals

    async def run_forever(self, poll_interval_seconds: int = 15, on_batch=None):
        """Polls every configured chain's tip concurrently, forever."""
        last_processed = {chain: None for chain in self.adapters}
        logger.info(f"Starting live multi-chain collector for: {list(self.adapters.keys())}")

        async def poll_chain(chain: str, adapter: ChainAdapter):
            latest = await adapter.get_latest_block_number()
            if latest is None:
                return
            if last_processed[chain] is None:
                last_processed[chain] = latest - 1
            while last_processed[chain] < latest:
                last_processed[chain] += 1
                raw_block = await adapter.fetch_block(last_processed[chain])
                if raw_block is None:
                    continue
                txs = await adapter.parse_block(raw_block, use_live_price=True)
                if txs and on_batch:
                    await on_batch(chain, txs)

        while True:
            try:
                await asyncio.gather(*[poll_chain(c, a) for c, a in self.adapters.items()])
            except Exception as e:
                logger.error(f"Unexpected error in multi-chain poll cycle: {e}")
            await asyncio.sleep(poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # =======================================================================
    # Offline test 1: Ethereum parsing (account model) — same mocked block
    # style as the previous single-chain version
    # =======================================================================
    print("=== Ethereum (account-based) ===")
    eth_block = {
        "number": hex(19000000),
        "hash": "0xblockhash123",
        "timestamp": hex(int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp())),
        "transactions": [
            {"hash": "0xtx1", "from": "0xWhaleWallet", "to": "0xExchangeHot1",
             "value": hex(int(500 * WEI_PER_ETH)), "gasPrice": hex(30_000_000_000)},
            {"hash": "0xtx2", "from": "0xNormalUser1", "to": "0xNormalUser2",
             "value": hex(int(0.05 * WEI_PER_ETH)), "gasPrice": hex(25_000_000_000)},
        ],
    }
    eth_receipts = {
        "0xtx1": {"gasUsed": hex(21000), "effectiveGasPrice": hex(30_000_000_000)},
        "0xtx2": {"gasUsed": hex(21000), "effectiveGasPrice": hex(25_000_000_000)},
    }
    eth_parsed = parse_ethereum_block_transactions(eth_block, eth_receipts, 2400.0, "live", "ethereum")
    for tx in eth_parsed:
        print(f"  {tx['txid']}: {tx['amount']:.4f} {tx['from_address']}->{tx['to_address']} model={tx['account_model']}")
    assert len(eth_parsed) == 2
    assert eth_parsed[0]["account_model"] == "account"

    # =======================================================================
    # Offline test 2: Bitcoin parsing (UTXO model) — multi-input/output tx,
    # coinbase tx, must not crash and must produce sane from/to stand-ins
    # =======================================================================
    print("\n=== Bitcoin (UTXO-based) ===")
    btc_block = {
        "header": {"height": 850000, "timestamp": int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp())},
        "transactions": [
            {   # coinbase tx — no real inputs, must not crash on from_address lookup
                "txid": "btc_coinbase_tx",
                "vin": [{"is_coinbase": True}],
                "vout": [{"scriptpubkey_address": "bc1qMinerReward", "value": 625_000_000}],
                "fee": 0,
            },
            {   # normal 2-input, 2-output tx (typical: payment + change)
                "txid": "btc_tx1",
                "vin": [
                    {"is_coinbase": False, "prevout": {"scriptpubkey_address": "bc1qSender1", "value": 50_000_000}},
                    {"is_coinbase": False, "prevout": {"scriptpubkey_address": "bc1qSender2", "value": 30_000_000}},
                ],
                "vout": [
                    {"scriptpubkey_address": "bc1qRecipient", "value": 70_000_000},   # the actual payment
                    {"scriptpubkey_address": "bc1qSender1Change", "value": 9_500_000},  # change back to sender
                ],
                "fee": 500_000,
            },
            {   # tiny/normal transaction — must be kept, not filtered out (the whole point)
                "txid": "btc_tx2",
                "vin": [{"is_coinbase": False, "prevout": {"scriptpubkey_address": "bc1qNormalUser1", "value": 100_000}}],
                "vout": [{"scriptpubkey_address": "bc1qNormalUser2", "value": 95_000}],
                "fee": 5_000,
            },
        ],
    }
    btc_parsed = parse_bitcoin_block_transactions(btc_block, 60000.0, "live")
    for tx in btc_parsed:
        print(f"  {tx['txid']}: {tx['amount']:.4f} BTC  from={tx['from_address']} to={tx['to_address']} "
              f"coinbase={tx['extra']['is_coinbase']} inputs={tx['extra']['input_count']} outputs={tx['extra']['output_count']}")

    assert len(btc_parsed) == 3, "all 3 BTC transactions must be kept, including the small one"
    coinbase_tx = next(t for t in btc_parsed if t["extra"]["is_coinbase"])
    assert coinbase_tx["from_address"] is None, "coinbase tx must not fabricate a sender"
    multi_input_tx = next(t for t in btc_parsed if t["txid"] == "btc_tx1")
    assert multi_input_tx["from_address"] == "bc1qSender1", "primary input used as from_address stand-in"
    assert multi_input_tx["to_address"] == "bc1qRecipient", "largest output correctly picked over the change output"
    assert multi_input_tx["amount"] == (70_000_000 + 9_500_000) / SATOSHI_PER_BTC, "amount = TOTAL output value, not just the primary"
    assert len(multi_input_tx["extra"]["inputs"]) == 2, "full input list preserved for later clustering work"
    small_tx = next(t for t in btc_parsed if t["txid"] == "btc_tx2")
    assert small_tx["amount"] < 0.01, "small/normal BTC transaction present and correctly parsed, not filtered"

    print("\nAll assertions passed for both Ethereum (account model) and Bitcoin (UTXO model).")

    # =======================================================================
    # Offline test 3: Solana parsing — reproduces a real IndexError bug
    # (flagged by user, confirmed) where a versioned (v0) transaction using
    # Address Lookup Tables has pre/postBalances longer than accountKeys,
    # because ALT-loaded accounts appear in the balance arrays but NOT in
    # the static accountKeys list. The old code indexed account_keys
    # directly with a balance-array index and crashed. Fixed by
    # reconstructing the full address list (accountKeys + loadedAddresses)
    # that actually matches the balance-array ordering, plus a bounds
    # check as defense in depth.
    # =======================================================================
    print("\n=== Solana (account-based, with versioned-tx / Address Lookup Table case) ===")
    sol_block = {
        "blockTime": 1754870400,
        "parentSlot": 299999999,
        "transactions": [
            {   # versioned tx: only 2 static accountKeys, but balances cover 4
                # accounts (2 static + 2 loaded via ALT) — this exact shape
                # crashed the old parser with IndexError
                "transaction": {
                    "signatures": ["sig_versioned_tx_1"],
                    "message": {"accountKeys": ["SignerWallet", "StaticAccount2"]},
                },
                "meta": {
                    "preBalances":  [5_000_000_000, 1_000_000_000, 2_000_000_000, 500_000_000],
                    "postBalances": [4_998_995_000, 1_000_000_000, 2_000_000_000, 1_500_000_000],
                    "fee": 5000,
                    "loadedAddresses": {
                        "writable": ["LoadedWritableAcct"],
                        "readonly": ["LoadedReadonlyAcct"],  # the actual recipient, index 3
                    },
                },
            },
            {   # ordinary legacy (non-versioned) tx — must keep working exactly as before
                "transaction": {
                    "signatures": ["sig_legacy_tx_1"],
                    "message": {"accountKeys": ["LegacySigner", "LegacyRecipient"]},
                },
                "meta": {
                    "preBalances":  [2_000_000_000, 500_000_000],
                    "postBalances": [1_499_995_000, 1_000_000_000],
                    "fee": 5000,
                },
            },
        ],
    }
    sol_parsed = parse_solana_block_transactions(sol_block, price_usd=150.0, price_precision="live")
    for tx in sol_parsed:
        print(f"  {tx['txid']}: {tx['amount']:.6f} SOL  from={tx['from_address']} to={tx['to_address']} extra={tx['extra']}")

    versioned_tx = next(t for t in sol_parsed if t["txid"] == "sig_versioned_tx_1")
    assert versioned_tx["to_address"] == "LoadedReadonlyAcct", "recipient must resolve via loadedAddresses, not crash or misattribute"
    assert abs(versioned_tx["amount"] - 1.0) < 1e-9
    assert versioned_tx["extra"]["loaded_addresses_count"] == 2
    assert versioned_tx["extra"]["address_list_complete"] is True

    legacy_tx = next(t for t in sol_parsed if t["txid"] == "sig_legacy_tx_1")
    assert legacy_tx["to_address"] == "LegacyRecipient", "legacy (non-versioned) tx behavior must be unchanged"
    assert abs(legacy_tx["amount"] - 0.5) < 1e-9

    print("\nNo crash on the versioned-transaction / Address Lookup Table case that previously")
    print("raised IndexError. Recipient correctly resolved via loadedAddresses. Legacy tx behavior unchanged.")
    print("\nNOTE: no network calls were executed in this sandbox — verify EthereumAdapter,")
    print("BitcoinAdapter, and SolanaAdapter's live HTTP/RPC calls against real endpoints on your side.")
