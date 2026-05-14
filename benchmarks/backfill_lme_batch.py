#!/usr/bin/env python3
"""Backfill bge-m3 embeddings for LongMemEval benchmark DB using batch encoding."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent" / "plugins" / "memory" / "mnemoss"))

from store import MemoryStore
from embedding import _load, _sess, _tok

def encode_batch_batch(texts):
    """Encode multiple texts at once using ONNX batch inference."""
    from embedding import _load
    import embedding as emb_mod
    
    _load()
    
    # Tokenize all texts
    encoded = [emb_mod._tok.encode(t) for t in texts]
    max_len = max(len(e.ids) for e in encoded)
    
    # Pad sequences
    input_ids = np.array([e.ids + [0] * (max_len - len(e.ids)) for e in encoded], dtype=np.int64)
    attention_mask = np.array([[1] * len(e.ids) + [0] * (max_len - len(e.ids)) for e in encoded], dtype=np.int64)
    
    # Run inference
    outputs = emb_mod._sess.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })
    
    # Mean pool and L2 normalize
    results = []
    for i in range(len(texts)):
        token_embeddings = outputs[0][i]  # (seq_len, 1024)
        mask = attention_mask[i]
        pooled = (token_embeddings * mask[:, np.newaxis]).sum(axis=0) / mask.sum()
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        results.append(pooled.astype(np.float32))
    
    return results


def main():
    db_path = Path.home() / ".hermes" / "benchmarks" / "longmemeval_benchmark.db"
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
    
    # Get facts without embeddings
    rows = store._conn.execute("""
        SELECT f.fact_id, f.content FROM facts f
        LEFT JOIN facts_vec fv ON fv.rowid = f.fact_id
        WHERE fv.rowid IS NULL
    """).fetchall()
    
    print(f"Facts to backfill: {len(rows)}")
    
    # Backfill in batches of 50
    batch_size = 50
    total_backfilled = 0
    start = time.time()
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        texts = [row[1] for row in batch]
        fact_ids = [row[0] for row in batch]
        
        # Encode batch
        embeddings = encode_batch_batch(texts)
        
        # Insert embeddings
        for j, (fact_id, vec) in enumerate(zip(fact_ids, embeddings)):
            if vec is not None:
                vec_bytes = vec.tobytes()
                store._conn.execute("DELETE FROM facts_vec WHERE rowid = ?", (fact_id,))
                store._conn.execute(
                    "INSERT INTO facts_vec(rowid, embedding) VALUES (?, ?)",
                    (fact_id, vec_bytes),
                )
                total_backfilled += 1
        
        store._conn.commit()
        
        elapsed = time.time() - start
        rate = total_backfilled / elapsed if elapsed > 0 else 0
        print(f"Backfilled {total_backfilled}/{len(rows)} ({total_backfilled/len(rows)*100:.1f}%) - {rate:.1f} facts/s")
    
    elapsed = time.time() - start
    print(f"\nDone! Backfilled {total_backfilled} embeddings in {elapsed:.1f}s ({total_backfilled/elapsed:.1f} facts/s)")
    
    # Verify
    final = store._conn.execute(
        "SELECT COUNT(*) FROM facts LEFT JOIN facts_vec ON facts.fact_id = facts_vec.rowid WHERE facts_vec.rowid IS NOT NULL"
    ).fetchone()[0]
    print(f"Final embeddings: {final}/{total}")


if __name__ == "__main__":
    main()
