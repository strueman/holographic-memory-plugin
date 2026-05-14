#!/usr/bin/env python3
"""Analyze why bge-m3 underperforms MiniLM on LongMemEval."""

import sys
import json
import sqlite3
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent" / "plugins" / "memory" / "holographic"))

from store import MemoryStore
from retrieval import FactRetriever
from embedding import encode, EMBEDDING_DIM

def main():
    db_path = Path.home() / ".hermes" / "benchmarks" / "longmemeval_benchmark.db"
    print(f"Loading DB: {db_path}")
    
    store = MemoryStore(str(db_path))
    retriever = FactRetriever(store)
    
    # Load queries from the DB
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Get all queries
    queries = conn.execute("SELECT * FROM lme_instances").fetchall()
    print(f"Total queries: {len(queries)}")
    
    # Analyze embedding quality
    print("\n=== Embedding Quality Analysis ===")
    
    # Check embedding norms and distribution
    sample_facts = store._conn.execute(
        "SELECT fact_id, content FROM facts LIMIT 100"
    ).fetchall()
    
    embeddings = []
    for fact in sample_facts:
        emb = encode(fact["content"])
        if emb is not None:
            embeddings.append(emb)
    
    if embeddings:
        embeddings = np.array(embeddings)
        print(f"Embedding dim: {embeddings.shape[1]}")
        print(f"Embedding norms: mean={embeddings.mean(axis=1).mean():.4f}, "
              f"std={embeddings.mean(axis=1).std():.4f}")
        print(f"Embedding min: {embeddings.min():.4f}, max: {embeddings.max():.4f}")
        
        # Check pairwise similarity distribution
        sims = []
        for i in range(min(50, len(embeddings))):
            for j in range(i+1, min(50, len(embeddings))):
                sim = np.dot(embeddings[i], embeddings[j])
                sims.append(sim)
        
        if sims:
            print(f"Pairwise similarity: mean={np.mean(sims):.4f}, "
                  f"std={np.std(sims):.4f}, min={np.min(sims):.4f}, max={np.max(sims):.4f}")
    
    # Analyze retrieval results for failing queries
    print("\n=== Retrieval Analysis ===")
    
    # Find queries where bge-m3 RRF failed but MiniLM would have succeeded
    # We'll check the actual retrieval scores
    failures = []
    successes = []
    
    for q in queries[:100]:  # Sample first 100 queries
        query_text = q["question"]
        expected_ids = json.loads(q["expected_fact_ids"]) if q["expected_fact_ids"] else []
        
        # Get RRF results
        rrf_results = retriever.search(query_text, limit=10)
        rrf_ids = [r["fact_id"] for r in rrf_results]
        
        # Check if expected fact is in top 3
        found_top3 = any(fid in expected_ids for fid in rrf_ids[:3])
        found_any = any(fid in expected_ids for fid in rrf_ids)
        
        if found_top3 and expected_ids:
            successes.append({
                "query": query_text[:60],
                "expected": expected_ids[:3],
                "retrieved": rrf_ids[:3],
            })
        elif expected_ids:
            failures.append({
                "query": query_text[:60],
                "expected": expected_ids[:3],
                "retrieved": rrf_ids[:3],
            })
    
    print(f"Sample analysis (first 100 queries):")
    print(f"  Successes (top-3): {len(successes)}")
    print(f"  Failures: {len(failures)}")
    
    # Show some failures
    print("\n=== Sample Failures ===")
    for f in failures[:5]:
        print(f"  Query: {f['query']}")
        print(f"    Expected: {f['expected']}")
        print(f"    Retrieved: {f['retrieved']}")
        print()
    
    # Show some successes
    print("=== Sample Successes ===")
    for s in successes[:5]:
        print(f"  Query: {s['query']}")
        print(f"    Expected: {s['expected']}")
        print(f"    Retrieved: {s['retrieved']}")
        print()
    
    # Check vec candidate counts
    print("\n=== Vec Candidate Analysis ===")
    vec_counts = []
    fts_counts = []
    for q in queries[:100]:
        query_text = q["question"]
        fts_cands = retriever._fts_candidates(query_text, category=None, min_trust=0.3, limit=30)
        vec_cands = retriever._vec_candidates(query_text, category=None, limit=30)
        vec_counts.append(len(vec_cands))
        fts_counts.append(len(fts_cands))
    
    print(f"FTS candidates: mean={np.mean(fts_counts):.1f}, median={np.median(fts_counts):.1f}")
    print(f"Vec candidates: mean={np.mean(vec_counts):.1f}, median={np.median(vec_counts):.1f}")
    print(f"Queries with 0 vec candidates: {sum(1 for v in vec_counts if v == 0)}")
    
    # Check if vec candidates are actually useful
    print("\n=== Vec vs FTS Overlap ===")
    overlap_count = 0
    total_checked = 0
    for q in queries[:50]:
        query_text = q["question"]
        fts_cands = retriever._fts_candidates(query_text, category=None, min_trust=0.3, limit=30)
        vec_cands = retriever._vec_candidates(query_text, category=None, limit=30)
        fts_ids = set(c["fact_id"] for c in fts_cands)
        vec_ids = set(c["fact_id"] for c in vec_cands)
        if fts_ids and vec_ids:
            overlap = len(fts_ids & vec_ids)
            overlap_count += overlap
            total_checked += 1
            if total_checked <= 5:
                print(f"  Overlap: {overlap}/{len(fts_ids)} FTS, {overlap}/{len(vec_ids)} VEC")
    
    if total_checked > 0:
        print(f"  Average overlap: {overlap_count/total_checked:.1f}")
    
    conn.close()

if __name__ == "__main__":
    main()
