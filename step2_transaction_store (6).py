"""
Step 2 — Transaction Store
----------------------------
Built on top of CryptoWhaleMonitor's DB layer, with the bugs fixed:

  BUG 1 (critical) — original had `if exists: return False` INSIDE the loop.
      One duplicate anywhere in a batch silently dropped every transaction
      after it. Fixed: `continue` — skip only the duplicate, keep processing
      the rest of the batch. Function now returns counts, not a bare bool,
      so the caller can actually see what happened.

  BUG 2 — `connect_args={"check_same_thread": False}` is SQLite-only.
      Passing it unconditionally breaks DuckDB (and any non-SQLite engine).
      Fixed: only applied when the dialect is actually sqlite.

  BUG 3 — `amount = Column(Float)` loses precision on financial data.
      Fixed: `Numeric(38, 18)` — enough range/precision for wei-level ETH
      amounts and satoshi-level BTC amounts without float rounding error.

  BUG 4 (added) — a plain "SELECT then INSERT" duplicate check has a race
      condition under concurrent writers (two pollers could both pass the
      check before either commits). Fixed: the composite primary key
      (blockchain, txid) is the real guard; the SELECT check is just an
      optimization to avoid an unnecessary round-trip, and IntegrityError
      is caught as the backstop (see Bug 7 for why this is a primary key
      and not a separate UniqueConstraint).

  BUG 5 (flagged by user, confirmed) — the original fix for Bug 4 called a
      bare `db.rollback()` inside the per-row except block. In SQLAlchemy,
      rollback() rolls back the ENTIRE current transaction, not just the
      failed statement. Since db.commit() only happens once, after the
      whole loop, any earlier rows in the same batch that were flushed-but-
      not-committed would be silently undone too — even though the `saved`
      counter had already counted them as successful. Example: TX A saved,
      TX B saved, TX C duplicate -> rollback() -> TX A and TX B are ALSO
      rolled back, despite saved==2 being reported.
      Fixed: initially with db.begin_nested() (SAVEPOINT scoped to just
      that one row) — but SAVEPOINT turned out to be unsupported by
      duckdb-engine (see Bug 8), so the final fix is commit-per-row
      instead, which achieves the same isolation without needing
      SAVEPOINT support at all.

  BUG 6 (flagged by user, confirmed) — `txid = Column(String, nullable=True)`
      lets the UniqueConstraint be silently bypassed. Standard SQL (SQLite,
      PostgreSQL, DuckDB, MySQL InnoDB) does NOT treat two NULLs as equal
      for uniqueness purposes — so multiple rows with txid=NULL can all be
      inserted, completely defeating dedup for exactly the rows most likely
      to be junk/incomplete data. Fixed: txid is now `nullable=False`. A
      real blockchain transaction always has a hash; a missing one is a
      Step 1/collection data-quality problem, not something this layer
      should silently store un-deduplicated. store_transactions() now
      rejects rows with a missing txid individually (counted as `failed`,
      with a reason) instead of crashing the batch or DB-erroring on commit.

  BUG 10 (flagged by user, confirmed) — `timestamp` was defined with
      `server_default=func.now()` but was NEVER read from the incoming tx
      dict when building the record. server_default only fires when no
      explicit value is given — so EVERY row, historical or live, got the
      wall-clock insert time instead of Step 1's actual block timestamp.
      Harmless-looking for live polling (insert happens seconds after the
      block), but silently destructive for historical backfill: a
      transaction from 3 months ago would be stored dated as "today",
      wrecking chronological order and every time-based Step 7 feature
      (rolling windows, leakage cutoffs, frequency/timing) that assumes
      this column reflects real transaction time. Fixed: `tx.get("timestamp")`
      is now read explicitly and passed to the record.

  BUG 9 (Step 1/Step 2 integration, found while wiring them together) —
      the old `store_transactions(blockchain: str, whales)` took the chain
      name as a SEPARATE argument from the actual transaction data. Step 1's
      multi-chain collector already tags every transaction dict with its
      own `chain` key (each adapter sets it directly) — passing blockchain
      again from the caller is redundant AND a mismatch risk: if a caller
      ever mixes up which batch came from which adapter (e.g. loops over
      chains and passes the wrong loop variable), every row would be
      silently mis-tagged with the wrong chain, with nothing to catch it.
      Fixed: `store_transactions()` now reads `chain` from EACH row
      individually. A `default_chain` param still exists for callers with
      hand-built dicts that don't set `chain` themselves (e.g. the demo
      below), but Step 1's real output never needs it — the field is
      already there per-row.

  MISSING FIELDS (flagged by user, confirmed) — Step 1's multi-chain output
      includes amount_usd, gas_fee_native, gas_fee_usd, block_number,
      price_precision, account_model, and extra (chain-specific detail:
      gas_used/is_contract_creation for ETH, inputs/outputs for BTC) — none
      of which had columns here. Silently dropping them would throw away
      exactly the fields the ML pipeline's gas-behavior and wallet-cluster
      features need. Fixed: added columns for all of them. `extra` is
      stored as JSON text (json.dumps/json.loads) rather than a native JSON
      column type, since JSON column support varies across SQLAlchemy
      dialect versions for DuckDB — text is guaranteed portable everywhere,
      at the cost of needing json.loads() on read (documented on get_recent).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import (
    create_engine, Column, String, Numeric, DateTime, BigInteger, Text
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

logger = logging.getLogger("transaction_store")

Base = declarative_base()


class WhaleTransaction(Base):
    __tablename__ = "whale_transactions"

    # BUG 7 (from real DuckDB run, confirmed) — the original surrogate
    # `id = Column(Integer, primary_key=True, autoincrement=True)` makes
    # SQLAlchemy emit `id SERIAL` in the CREATE TABLE DDL. DuckDB's SQL
    # dialect does not recognize SERIAL as a type name (that's a
    # Postgres-ism), so table creation fails immediately with
    # "Catalog Error: Type with name SERIAL does not exist!"
    #
    # Fixed by removing the surrogate key entirely and using the natural
    # key instead: (blockchain, txid) together ARE the unique identity of
    # a transaction record already — no autoincrement needed, so no
    # dialect-specific autoincrement syntax is generated at all. This is
    # portable across SQLite, DuckDB, and Postgres without any DB-specific
    # workaround, and it also makes the old UniqueConstraint redundant
    # (a composite primary key already enforces the same uniqueness).
    blockchain = Column(String, primary_key=True)  # populated from tx["chain"], see Bug 9
    txid = Column(String, primary_key=True, index=True)
    from_address = Column(String)
    to_address = Column(String)
    # Numeric, not Float — avoids binary floating-point rounding error on
    # financial amounts (e.g. 0.1 + 0.2 != 0.3 in float, matters at scale)
    amount = Column(Numeric(38, 18), nullable=False)             # native units (ETH, BTC, SOL, ...)
    amount_usd = Column(Numeric(38, 2), nullable=True)             # precomputed — avoids recomputing on every query
    price_usd = Column(Numeric(38, 8), nullable=True)               # per-unit price at time of tx
    price_precision = Column(String, nullable=True)                  # "live" | "daily_historical" | "unknown"
    gas_fee_native = Column(Numeric(38, 18), nullable=True)
    gas_fee_usd = Column(Numeric(38, 8), nullable=True)
    account_model = Column(String, nullable=True)                     # "account" | "utxo" — tells Step 4/5 how to read from/to
    block_hash_or_number = Column(String, nullable=True)
    block_number = Column(BigInteger, nullable=True)                   # numeric, for range queries — block_hash_or_number stays string for flexibility
    classification = Column(String, nullable=True)
    extra = Column(Text, nullable=True)                                  # JSON-encoded chain-specific detail (see Bug fix note above)
    timestamp = Column(DateTime(timezone=True), nullable=True, server_default=func.now())

    # no __table_args__ / UniqueConstraint needed anymore — the composite
    # primary key (blockchain, txid) above already enforces this at the DB
    # level, on every backend, without any extra constraint object


class TransactionStore:
    def __init__(self, database_url: str):
        """
        database_url examples:
          SQLite:  'sqlite:///./whale.db'
          DuckDB:  'duckdb:///./whale.duckdb'   (requires `pip install duckdb-engine`)
        """
        self.database_url = database_url
        is_sqlite = database_url.startswith("sqlite")

        # check_same_thread is SQLite-specific — do NOT pass it to DuckDB or others
        connect_args = {"check_same_thread": False} if is_sqlite else {}

        self.engine = create_engine(database_url, connect_args=connect_args)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        Base.metadata.create_all(bind=self.engine)

    def store_transactions(self, whales: Iterable[dict], default_chain: Optional[str] = None) -> dict:
        """
        Store a batch of transactions. Skips duplicates individually instead
        of aborting the whole batch on the first one found (Bug 1 fix).

        Each item in `whales` is expected to be a dict matching Step 1's
        normalized output shape (make_normalized_tx): chain, txid,
        from_address, to_address, amount, amount_usd, gas_fee_native,
        gas_fee_usd, block_hash_or_number, block_number, timestamp,
        price_usd, price_precision, account_model, classification, extra.

        `default_chain` is only a fallback for rows that don't carry their
        own `chain` key (e.g. hand-built test dicts) — Step 1's real output
        always sets `chain` per-row, so real pipeline calls don't need this
        argument at all (Bug 9 fix).

        Returns: {"saved": int, "duplicates": int, "failed": int}
        """
        db = self.SessionLocal()
        saved = duplicates = failed = 0

        for tx in whales:
            txid = tx.get("txid")
            chain = tx.get("chain") or default_chain

            # Bug 6 fix: txid is required (see class docstring). Reject just
            # this row, not the whole batch — a missing txid means upstream
            # (Step 1) data quality is broken and needs investigation, but
            # one bad row shouldn't block the rest of a valid batch.
            if not txid:
                failed += 1
                logger.warning(
                    f"Rejected row with missing txid (chain={chain}, "
                    f"from={tx.get('from_address')}, to={tx.get('to_address')}) — "
                    f"cannot dedup or store a transaction without its hash"
                )
                continue

            if not chain:
                failed += 1
                logger.warning(f"Rejected row with no chain identified (txid={txid}) — "
                                f"pass default_chain= if this batch is hand-built and doesn't set 'chain' per-row")
                continue

            # optimization only — NOT the source of truth for uniqueness,
            # the DB-level composite primary key is (see class docstring, Bug 4)
            exists = db.query(WhaleTransaction).filter_by(
                blockchain=chain, txid=txid
            ).first()
            if exists:
                duplicates += 1
                continue  # <-- Bug 1 fix: was `return False`, killed the whole batch

            amount = tx.get("amount")
            price_usd = tx.get("price_usd")
            # amount_usd: prefer Step 1's precomputed value; fall back to
            # computing it here only if the collector didn't supply one
            amount_usd = tx.get("amount_usd")
            if amount_usd is None and amount is not None and price_usd is not None:
                amount_usd = float(amount) * float(price_usd)

            extra = tx.get("extra")
            extra_json = json.dumps(extra) if extra else None

            # BUG 10 (flagged by user, confirmed) — `timestamp` was never
            # read from `tx` when building the record. The column has
            # `server_default=func.now()`, which only fires when NO value
            # is supplied at insert time — since the record construction
            # below never passed one, EVERY row got the current wall-clock
            # insert time, not the actual blockchain timestamp Step 1
            # already computed from the block header. For a live poll this
            # is barely noticeable (insert happens seconds after the block
            # is mined). For a historical backfill it's a serious bug: a
            # transaction from 3 months ago would be stored with TODAY's
            # date, silently destroying chronological order and breaking
            # every time-based feature in Step 7 (rolling windows, leakage
            # cutoffs, frequency/timing features all depend on this column
            # being the real transaction time, not the insert time).
            # Fixed: read tx["timestamp"] explicitly; server_default only
            # remains as a safety net for the rare case a row genuinely
            # has no timestamp at all (better than NULL, not a replacement
            # for the real value).
            tx_timestamp = tx.get("timestamp")

            record = WhaleTransaction(
                blockchain=chain,
                txid=txid,
                from_address=tx.get("from_address") or "Unknown",
                to_address=tx.get("to_address") or "Unknown",
                amount=amount,
                amount_usd=amount_usd,
                price_usd=price_usd,
                price_precision=tx.get("price_precision"),
                gas_fee_native=tx.get("gas_fee_native"),
                gas_fee_usd=tx.get("gas_fee_usd"),
                account_model=tx.get("account_model"),
                block_hash_or_number=tx.get("block_hash_or_number") or "Unknown",
                block_number=tx.get("block_number"),
                classification=tx.get("classification"),
                extra=extra_json,
                timestamp=tx_timestamp,  # <-- Bug 10 fix: was missing entirely
            )

            # BUG 8 (from real DuckDB run, confirmed) — the SAVEPOINT-based
            # fix (db.begin_nested()) for Bug 5 assumed nested transactions
            # were supported. duckdb-engine does NOT support SAVEPOINT at
            # the SQL level: "Parser Error: syntax error at or near
            # SAVEPOINT". That's a real limitation of the DuckDB dialect,
            # not something fixable from the SQLAlchemy side.
            #
            # Fixed with a more portable pattern that needs no nested-
            # transaction support at all: COMMIT PER ROW. Each row is its
            # own complete transaction — if a row fails, its own rollback()
            # only ever undoes that one uncommitted row, because every
            # earlier row was already fully committed to disk before we
            # even got here. This is strictly more portable than SAVEPOINT
            # (works identically on SQLite, DuckDB, Postgres, MySQL — no
            # dialect-specific feature required), at the cost of one commit
            # per row instead of one commit per batch. This also directly
            # answers the "write straight to DB during historical backfill,
            # don't accumulate in RAM" requirement — see Step 1's
            # backfill_chain_to_store(), which calls this once per block
            # instead of once for the whole range.
            try:
                db.add(record)
                db.commit()
                saved += 1
            except IntegrityError:
                db.rollback()  # safe here: only THIS row was ever uncommitted
                duplicates += 1
                logger.info(f"Duplicate caught at DB level (race condition guard): {chain}/{txid}")
            except Exception as e:
                db.rollback()
                failed += 1
                logger.error(f"Unexpected error storing row (chain={chain}, txid={txid}): {e}")

        db.close()

        result = {"saved": saved, "duplicates": duplicates, "failed": failed}
        logger.info(f"Batch store result: {result}")
        return result

    def get_recent(self, blockchain: str = None, limit: int = 50):
        """
        Note: `extra` comes back as a JSON string (see Bug fix note in the
        model docstring) — call json.loads(row.extra) if you need the
        chain-specific detail (gas_used, is_contract_creation, BTC
        inputs/outputs, etc.) rather than just the top-level columns.
        """
        db = self.SessionLocal()
        try:
            q = db.query(WhaleTransaction)
            if blockchain:
                q = q.filter_by(blockchain=blockchain)
            return q.order_by(WhaleTransaction.timestamp.desc()).limit(limit).all()
        finally:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from datetime import timedelta

    # demo with SQLite so it runs anywhere without extra deps;
    # swap to 'duckdb:///./whale.duckdb' once duckdb-engine is installed
    store = TransactionStore("sqlite:///./whale_demo.db")

    # a "3 months ago" timestamp, exactly the scenario the user flagged —
    # this must survive into the DB as-is, NOT get replaced by today's date
    three_months_ago = datetime.now(timezone.utc) - timedelta(days=90)

    # Pre-insert one row directly so txid '0xaaa' is a REAL duplicate against
    # something already committed, not just a same-batch repeat.
    store.store_transactions([
        {"chain": "ethereum", "txid": "0xaaa", "from_address": "0xA", "to_address": "0xB",
         "amount": "1500000.123456789012345678", "amount_usd": "3600000000.50",
         "price_usd": "2400.00", "price_precision": "live", "account_model": "account",
         "block_number": 19000000, "gas_fee_native": "0.00063", "gas_fee_usd": "1.51",
         "timestamp": three_months_ago},
    ])

    # Batch mixes an Ethereum row (account model) and a Bitcoin row (UTXO
    # model, with the extra input/output detail) — exactly what Step 1's
    # MultiChainCollector produces. No `blockchain` argument needed anymore:
    # each row carries its own `chain` (Bug 9 fix).
    batch = [
        {"chain": "ethereum", "txid": "0xbbb", "from_address": "0xC", "to_address": "0xD",
         "amount": "2200000.5", "amount_usd": "5280001200.00", "price_usd": "2400.00",
         "price_precision": "live", "account_model": "account", "block_number": 19000001,
         "gas_fee_native": "0.00063", "gas_fee_usd": "1.51",
         "extra": {"gas_used": 21000, "gas_price_gwei": 30.0, "is_contract_creation": False}},

        {"chain": "bitcoin", "txid": "btc_tx1", "from_address": "bc1qSender1", "to_address": "bc1qRecipient",
         "amount": "0.795", "amount_usd": "47700.00", "price_usd": "60000.00",
         "price_precision": "live", "account_model": "utxo", "block_number": 850000,
         "gas_fee_native": "0.000005", "gas_fee_usd": "0.30",
         "extra": {"is_coinbase": False, "input_count": 2, "output_count": 2,
                    "inputs": ["bc1qSender1", "bc1qSender2"],
                    "outputs": [{"address": "bc1qRecipient", "value_sats": 70000000}]}},

        {"chain": "ethereum", "txid": "0xaaa", "from_address": "0xA", "to_address": "0xB",
         "amount": "1500000.123456789012345678", "price_usd": "2400.00"},  # duplicate, in the MIDDLE

        {"chain": "ethereum", "txid": "0xddd", "from_address": "0xG", "to_address": "0xH",
         "amount": "42.0", "price_usd": "2400.00"},  # new, after the duplicate

        {"chain": "ethereum", "txid": None, "from_address": "0xI", "to_address": "0xJ",
         "amount": "100.0"},  # missing txid, should be rejected (Bug 6)
    ]

    result = store.store_transactions(batch)
    print(result)  # expect: saved=3, duplicates=1, failed=1

    rows = store.get_recent(limit=10)
    print(f"\nRows actually in DB after commit: {len(rows)} (expect 4: the pre-insert + 3 saved from batch)")
    for r in rows:
        print(f"[{r.blockchain}] {r.txid}  amount={r.amount}  amount_usd={r.amount_usd}  "
              f"gas_fee_usd={r.gas_fee_usd}  account_model={r.account_model}  "
              f"block={r.block_number}  extra={r.extra}")

    # the real proof: 0xbbb and btc_tx1 (committed BEFORE the 0xaaa duplicate hit)
    # must still be in the DB — this is what Bug 5 would have broken
    saved_txids = {r.txid for r in rows}
    assert "0xbbb" in saved_txids, "TX flushed before the duplicate was lost — Bug 5 regression!"
    assert "btc_tx1" in saved_txids, "Bitcoin TX flushed before the duplicate was lost — Bug 5 regression!"
    assert "0xddd" in saved_txids, "TX flushed after the duplicate was lost!"

    # the new proof: multi-chain fields actually round-trip through the DB
    btc_row = next(r for r in rows if r.txid == "btc_tx1")
    assert btc_row.blockchain == "bitcoin", "chain read from the row itself, not a separate argument (Bug 9)"
    assert btc_row.account_model == "utxo"
    btc_extra = json.loads(btc_row.extra)
    assert btc_extra["input_count"] == 2, "extra JSON (BTC input/output detail) survived the round-trip"

    eth_row = next(r for r in rows if r.txid == "0xbbb")
    assert eth_row.gas_fee_usd is not None, "gas fee data survived the round-trip — this was completely missing before"
    eth_extra = json.loads(eth_row.extra)
    assert eth_extra["gas_used"] == 21000

    # BUG 10 proof: the 3-months-ago row must keep ITS timestamp, not today's
    old_row = next(r for r in rows if r.txid == "0xaaa")
    stored_ts = old_row.timestamp
    if stored_ts.tzinfo is None:
        stored_ts = stored_ts.replace(tzinfo=timezone.utc)  # SQLite sometimes drops tzinfo on read
    age_days = (datetime.now(timezone.utc) - stored_ts).days
    print(f"\n0xaaa stored timestamp: {stored_ts}  (age: {age_days} days, expect ~90)")
    assert age_days >= 85, (
        f"BUG 10 REGRESSION: a 3-month-old transaction was stored with a recent "
        f"timestamp (age={age_days} days) — historical data is being silently "
        f"dated as 'today' instead of keeping its real transaction time"
    )

    print("\nConfirmed: rows flushed before AND after the mid-batch duplicate both survived,")
    print("AND all of Step 1's multi-chain fields (gas, amount_usd, account_model, extra JSON)")
    print("now round-trip through the database intact — nothing from Step 1's output is being dropped.")
    print("AND a historical (3-months-ago) timestamp survives intact — it is NOT overwritten")
    print("with today's insert-time date, which is exactly the bug that was just fixed.")
