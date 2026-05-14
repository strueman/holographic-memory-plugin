#!/usr/bin/env python3
"""Backfill embeddings for a benchmark DB after model/schema upgrade.

Usage:
    python3 backfill_embeddings.py --db PATH_TO_DB

If no --db specified, uses the default Chinese benchmark DB.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent" / "plugins" / "memory" / "holographic"))

from store import MemoryStore


def backfill(db_path: str) -> None:
    """Backfill embeddings for all facts in a DB."""
    print(f"Loading DB: {db_path}")
    store = MemoryStore(db_path)

    total = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    with_emb = store._conn.execute(
        "SELECT COUNT(*) FROM facts LEFT JOIN facts_vec ON facts.fact_id = facts_vec.rowid WHERE facts_vec.rowid IS NOT NULL"
    ).fetchone()[0]

    print(f"Total facts: {total}")
    print(f"Current embeddings: {with_emb}")
    print(f"Need embeddings: {total - with_emb}")

    if total == with_emb:
        print("All facts already have embeddings.")
        return

    count = store.backfill_embeddings()
    print(f"Backfilled {count} embeddings")

    final = store._conn.execute(
        "SELECT COUNT(*) FROM facts LEFT JOIN facts_vec ON facts.fact_id = facts_vec.rowid WHERE facts_vec.rowid IS NOT NULL"
    ).fetchone()[0]
    print(f"Final embeddings: {final}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings for a benchmark DB")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "benchmarks" / "chinese_cmrc2018.db"),
                        help="Path to benchmark DB")
    args = parser.parse_args()
    backfill(args.db)


if __name__ == "__main__":
    main()
