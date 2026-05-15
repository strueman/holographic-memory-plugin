#!/usr/bin/env python3
"""Backfill bge-m3 embeddings for Chinese benchmark DB."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent" / "plugins" / "memory" / "mnemoss"))

from store import MemoryStore

def main():
    db_path = Path.home() / ".hermes" / "benchmarks" / "chinese_cmrc2018.db"
    print(f"Loading DB: {db_path}")
    
    store = MemoryStore(str(db_path))
    
    # Count facts and current embeddings
    total = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    with_embeddings = store._conn.execute(
        "SELECT COUNT(*) FROM facts LEFT JOIN facts_vec ON facts.fact_id = facts_vec.rowid WHERE facts_vec.rowid IS NOT NULL"
    ).fetchone()[0]
    
    print(f"Total facts: {total}")
    print(f"Current embeddings: {with_embeddings}")
    print(f"Need embeddings: {total - with_embeddings}")
    
    if total == with_embeddings:
        print("All facts already have embeddings.")
        return
    
    count = store.backfill_embeddings()
    print(f"Backfilled {count} embeddings in {time.time():.0f}")
    
    # Verify
    final = store._conn.execute(
        "SELECT COUNT(*) FROM facts LEFT JOIN facts_vec ON facts.fact_id = facts_vec.rowid WHERE facts_vec.rowid IS NOT NULL"
    ).fetchone()[0]
    print(f"Final embeddings: {final}/{total}")

if __name__ == "__main__":
    main()
