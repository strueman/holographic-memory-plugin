"""Holographic memory benchmark: RRF vs weighted-sum.

Uses the real retrieval pipeline (retrieval.py) for candidate generation
and scoring — no more standalone reimplementations.

Usage:
    python holographic_benchmark.py                    # run all benchmarks
    python holographic_benchmark.py --limit 3          # top-3 results only
    python holographic_benchmark.py --method both      # compare both (default)
    python holographic_benchmark.py --method rrf       # RRF only
    python holographic_benchmark.py --method weighted  # weighted-sum only
    python holographic_benchmark.py --queries "exact"  # run specific query types
    python holographic_benchmark.py --longmemeval      # use LongMemEval oracle dataset
    python holographic_benchmark.py --longmemeval --subset 50  # first 50 LME instances
    python holographic_benchmark.py --live             # test live FactRetriever.search()
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

DEFAULT_DB = Path.home() / ".hermes" / "memory_store.db"

# ---------------------------------------------------------------------------
# Plugin imports (real pipeline)
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path.home() / ".hermes" / "hermes-agent" / "plugins" / "memory" / "holographic"
sys.path.insert(0, str(_PLUGIN_DIR))

try:
    from store import MemoryStore
    from retrieval import FactRetriever
    HAS_PLUGIN = True
except ImportError as e:
    HAS_PLUGIN = False
    _IMPORT_ERR = e


# ---------------------------------------------------------------------------
# Golden query set
# ---------------------------------------------------------------------------

GOLDEN_QUERIES: List[dict] = [
    # --- Exact match ---
    {
        "query": "holographic memory plugin uses HRR with 1024-dim vectors",
        "expected_facts": [2],
        "type": "exact",
        "description": "Should find the architecture fact with high precision",
    },
    {
        "query": "RRF replaced weighted-sum scoring with rank-based fusion C=60",
        "expected_facts": [3],
        "type": "exact",
        "description": "Should find the RRF implementation fact",
    },
    {
        "query": "strueman holographic-memory-plugin GitHub repo",
        "expected_facts": [14],
        "type": "exact",
        "description": "Should find the GitHub repo fact",
    },
    {
        "query": "dusterbloom PR #2351 original author",
        "expected_facts": [15],
        "type": "exact",
        "description": "Should find the original author fact",
    },

    # --- Partial match ---
    {
        "query": "what memory architectures improve holographic recall",
        "expected_facts": [2, 5, 6, 7, 8, 9],
        "type": "partial",
        "description": "Broad query matching multiple research papers",
    },
    {
        "query": "what are the limitations of the current holographic plugin",
        "expected_facts": [4],
        "type": "partial",
        "description": "Conceptual partial match on limitations/gaps",
    },
    {
        "query": "Hindsight plugin CARA reflection",
        "expected_facts": [11],
        "type": "partial",
        "description": "Partial match on Hindsight + CARA keywords",
    },

    # --- Multi-hop ---
    {
        "query": "what research papers have HIGH applicability to holographic",
        "expected_facts": [5, 6, 8],
        "type": "multi_hop",
        "description": "Needs to match both applicability and HIGH keywords",
    },
    {
        "query": "what memory consolidation approaches exist",
        "expected_facts": [7, 8, 9],
        "type": "multi_hop",
        "description": "Multi-hop: consolidation + memory approaches",
    },

    # --- Conceptual ---
    {
        "query": "vector symbolic architectures VSA beyond HRR",
        "expected_facts": [2, 12],
        "type": "conceptual",
        "description": "Conceptual match on VSA/HRR relationship",
    },
    {
        "query": "how does the memory plugin integrate with the agent",
        "expected_facts": [17],
        "type": "conceptual",
        "description": "Conceptual match on integration points",
    },

    # --- No-match ---
    {
        "query": "whats for dinner tonight",
        "expected_facts": [],
        "type": "no_match",
        "description": "Should return low/no results",
    },
    {
        "query": "how to cook pasta",
        "expected_facts": [],
        "type": "no_match",
        "description": "Should return low/no results",
    },
]


# ---------------------------------------------------------------------------
# LongMemEval query loader
# ---------------------------------------------------------------------------

def load_longmemeval_queries(
    db_path: Path,
    subset: Optional[int] = None,
) -> List[dict]:
    """Load queries from a LongMemEval benchmark database."""
    if not db_path.exists():
        print(f"Error: LongMemEval DB not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lme_instances (
            question_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            question_type TEXT,
            expected_fact_ids TEXT
        )
    """)

    existing = conn.execute("SELECT COUNT(*) FROM lme_instances").fetchone()[0]
    if existing == 0:
        print("  LongMemEval instances not pre-extracted. Use prepare_longmemeval_db.py first.")
        conn.close()
        return []

    queries = []
    for row in conn.execute(
        "SELECT question_id, question, question_type, expected_fact_ids FROM lme_instances"
    ):
        if subset and len(queries) >= subset:
            break
        expected = json.loads(row[3]) if row[3] else []
        queries.append({
            "query": row[1],
            "expected_facts": expected,
            "type": row[2] or "unknown",
            "description": f"LongMemEval: {row[0]}",
        })

    conn.close()
    return queries


# ---------------------------------------------------------------------------
# Tokenization (for legacy standalone mode)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'\b\w+\b', re.IGNORECASE)

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# Scoring: real pipeline methods
# ---------------------------------------------------------------------------

def _get_candidates(retriever: FactRetriever, query: str, limit: int) -> tuple:
    """Get FTS5 + vec candidates from the real pipeline.

    Returns (fts_candidates, vec_candidates) as lists of dicts.
    """
    fts = retriever._fts_candidates(query, category=None, min_trust=0.3, limit=limit)
    vec = retriever._vec_candidates(query, category=None, limit=limit)
    return fts, vec


def _rrf_score_candidates(
    fts_candidates: list,
    vec_candidates: list,
    retriever: FactRetriever,
    query: str,
    limit: int,
    rrf_c: int = 60,
) -> List[dict]:
    """Apply RRF fusion on the candidate pool (mirrors FactRetriever.search)."""
    # Union candidates, deduplicate
    seen_ids = set()
    all_candidates = []
    for fact in fts_candidates:
        if fact["fact_id"] not in seen_ids:
            seen_ids.add(fact["fact_id"])
            all_candidates.append(fact)
    for fact in vec_candidates:
        if fact["fact_id"] not in seen_ids:
            seen_ids.add(fact["fact_id"])
            all_candidates.append(fact)

    if not all_candidates:
        return []

    # Build rank maps
    fts_rank_map = {f["fact_id"]: i+1 for i, f in enumerate(fts_candidates)}
    vec_rank_map = {f["fact_id"]: i+1 for i, f in enumerate(vec_candidates)}

    # Tokenize for Jaccard
    query_tokens = retriever._tokenize(query)

    scored = []
    for fact in all_candidates:
        content_tokens = retriever._tokenize(fact["content"])
        tag_tokens = retriever._tokenize(fact.get("tags", ""))
        all_tokens = content_tokens | tag_tokens

        # Jaccard
        jaccard = retriever._jaccard_similarity(query_tokens, all_tokens)

        # RRF fusion
        rrf_score = 0.0
        source_count = 0

        if fact["fact_id"] in fts_rank_map:
            rrf_score += 1.0 / (fts_rank_map[fact["fact_id"]] + rrf_c)
            source_count += 1
        if fact["fact_id"] in vec_rank_map:
            rrf_score += 1.0 / (vec_rank_map[fact["fact_id"]] + rrf_c)
            source_count += 1

        # Jaccard rank
        jaccard_rank = max(1, int((1.0 - jaccard) * len(all_candidates)) + 1)
        rrf_score += 1.0 / (jaccard_rank + rrf_c)
        source_count += 1

        # HRR rank (use neutral default for benchmark — HRR is consistent across methods)
        hrr_rank = max(1, int(0.5 * len(all_candidates)) + 1)
        rrf_score += 1.0 / (hrr_rank + rrf_c)
        source_count += 1

        if source_count > 0:
            rrf_score /= source_count

        score = rrf_score * fact["trust_score"]
        fact["score"] = score
        scored.append(fact)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def _weighted_score_candidates(
    fts_candidates: list,
    vec_candidates: list,
    retriever: FactRetriever,
    query: str,
    limit: int,
    fts_weight: float = 0.4,
    jaccard_weight: float = 0.3,
    vec_weight: float = 0.3,
) -> List[dict]:
    """Apply weighted-sum scoring on the same candidate pool."""
    # Union candidates, deduplicate
    seen_ids = set()
    all_candidates = []
    for fact in fts_candidates:
        if fact["fact_id"] not in seen_ids:
            seen_ids.add(fact["fact_id"])
            all_candidates.append(fact)
    for fact in vec_candidates:
        if fact["fact_id"] not in seen_ids:
            seen_ids.add(fact["fact_id"])
            all_candidates.append(fact)

    if not all_candidates:
        return []

    # Normalize ranks to [0, 1]
    fts_rank_map = {f["fact_id"]: i+1 for i, f in enumerate(fts_candidates)}
    vec_rank_map = {f["fact_id"]: i+1 for i, f in enumerate(vec_candidates)}
    max_fts_rank = len(fts_candidates) if fts_candidates else 1
    max_vec_rank = len(vec_candidates) if vec_candidates else 1

    query_tokens = retriever._tokenize(query)

    scored = []
    for fact in all_candidates:
        # FTS5 score (normalized rank)
        fts_r = fts_rank_map.get(fact["fact_id"], max_fts_rank)
        fts_score = max(0.0, 1.0 - (fts_r - 1) / max(max_fts_rank, 1))

        # Vec score (normalized rank)
        vec_r = vec_rank_map.get(fact["fact_id"], max_vec_rank)
        vec_score = max(0.0, 1.0 - (vec_r - 1) / max(max_vec_rank, 1))

        # Jaccard
        content_tokens = retriever._tokenize(fact["content"])
        tag_tokens = retriever._tokenize(fact.get("tags", ""))
        jaccard = retriever._jaccard_similarity(query_tokens, content_tokens | tag_tokens)

        # Weighted sum
        score = (fts_weight * fts_score +
                 jaccard_weight * jaccard +
                 vec_weight * vec_score)

        score *= fact["trust_score"]
        fact["score"] = score
        scored.append(fact)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[int], expected_ids: List[int], k: int) -> float:
    top_k = retrieved_ids[:k]
    relevant = sum(1 for fid in top_k if fid in expected_ids)
    return relevant / min(k, len(top_k)) if top_k else 0.0


def recall_at_k(retrieved_ids: List[int], expected_ids: List[int], k: int) -> float:
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(expected_ids)) / len(expected_ids)


def compute_metrics(retrieved: List[dict], expected_facts: List[int], k_values=(3, 5)):
    retrieved_ids = [r["fact_id"] for r in retrieved]
    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, expected_facts, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, expected_facts, k)

    if expected_facts:
        metrics["max_precision"] = precision_at_k(retrieved_ids, expected_facts, len(retrieved_ids))
        metrics["max_recall"] = recall_at_k(retrieved_ids, expected_facts, len(retrieved_ids))
    else:
        metrics["max_precision"] = 0.0 if retrieved_ids else 1.0
        metrics["max_recall"] = 1.0

    for k in k_values:
        top_k = retrieved_ids[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, fid in enumerate(top_k) if fid in expected_facts)
        n_rel = min(len(expected_facts), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query: str
    query_type: str
    expected_facts: List[int]
    description: str
    weighted_metrics: dict = field(default_factory=dict)
    rrf_metrics: dict = field(default_factory=dict)
    weighted_retrieved: List[int] = field(default_factory=list)
    rrf_retrieved: List[int] = field(default_factory=list)


def run_benchmark(
    db_path: Path = DEFAULT_DB,
    method: str = "both",
    limit: int = 10,
    query_types: Optional[List[str]] = None,
    queries: Optional[List[dict]] = None,
    live: bool = False,
):
    """Run the full benchmark using the real retrieval pipeline.

    If live=True, calls FactRetriever.search() directly instead of
    the candidate-pool comparison mode.
    """
    if not db_path.exists():
        print(f"Error: DB not found at {db_path}")
        return []

    if not HAS_PLUGIN:
        print(f"Error: holographic plugin not available: {_IMPORT_ERR}")
        print("  Make sure the plugin is installed at:", _PLUGIN_DIR)
        return []

    store = MemoryStore(str(db_path))
    retriever = FactRetriever(store)

    # Verify vec is actually working
    vec_ok = False
    try:
        import sqlite_vec  # noqa: F401
        from embedding import is_available
        vec_ok = is_available()
    except ImportError:
        pass
    if not vec_ok:
        print("WARNING: vec search unavailable (onnxruntime or sqlite_vec missing)")
        print("  Install: pip install onnxruntime sqlite-vec")

    # Check facts_vec table
    try:
        vec_count = store._conn.execute("SELECT count(*) FROM facts_vec").fetchone()[0]
        fact_count = store._conn.execute("SELECT count(*) FROM facts").fetchone()[0]
        print(f"DB: {fact_count} facts, {vec_count} vec embeddings ({vec_count/max(fact_count,1)*100:.0f}%)")
    except Exception:
        print("WARNING: facts_vec table not found — vec search will be disabled")

    # Use provided queries or fall back to golden + filtering
    if queries is None:
        all_queries = GOLDEN_QUERIES
        if query_types:
            all_queries = [q for q in all_queries if q["type"] in query_types]
    else:
        all_queries = queries

    results = []

    for i, q in enumerate(all_queries, 1):
        query_text = q["query"]
        expected = q["expected_facts"]
        qtype = q["type"]
        desc = q["description"]

        if live:
            # Live mode: call the real search() directly
            live_results = retriever.search(query_text, limit=limit)
            live_ids = [r["fact_id"] for r in live_results]
            live_metrics = compute_metrics(live_results, expected)

            dr = QueryResult(
                query=query_text,
                query_type=qtype,
                expected_facts=expected,
                description=desc,
                rrf_metrics=live_metrics,
                rrf_retrieved=live_ids,
            )
            print(f"\nQuery {i}: {query_text[:70]}")
            print(f"  Type: {qtype} | Expected: {expected}")
            print(f"  Live:  retrieved={live_ids} | P@3={live_metrics.get('precision@3', 0):.2f} R@3={live_metrics.get('recall@3', 0):.2f}")
        else:
            # Candidate-pool mode: fair comparison with same candidates
            fts_cands, vec_cands = _get_candidates(retriever, query_text, limit * 3)

            rrf_results = _rrf_score_candidates(fts_cands, vec_cands, retriever, query_text, limit)
            weighted_results = _weighted_score_candidates(fts_cands, vec_cands, retriever, query_text, limit)

            rrf_ids = [r["fact_id"] for r in rrf_results]
            weighted_ids = [r["fact_id"] for r in weighted_results]

            w_metrics = compute_metrics(weighted_results, expected)
            r_metrics = compute_metrics(rrf_results, expected)

            dr = QueryResult(
                query=query_text,
                query_type=qtype,
                expected_facts=expected,
                description=desc,
                weighted_metrics=w_metrics,
                rrf_metrics=r_metrics,
                weighted_retrieved=weighted_ids,
                rrf_retrieved=rrf_ids,
            )

            print(f"\nQuery {i}: {query_text[:70]}")
            print(f"  Type: {qtype} | Expected: {expected}")
            print(f"  FTS5={len(fts_cands)} Vec={len(vec_cands)} candidates")
            print(f"  Weighted: retrieved={weighted_ids} | P@3={w_metrics.get('precision@3', 0):.2f} R@3={w_metrics.get('recall@3', 0):.2f}")
            print(f"  RRF:      retrieved={rrf_ids} | P@3={r_metrics.get('precision@3', 0):.2f} R@3={r_metrics.get('recall@3', 0):.2f}")
            delta_p3 = r_metrics.get("precision@3", 0) - w_metrics.get("precision@3", 0)
            print(f"  Delta P@3: {delta_p3:+.2f}")

        results.append(dr)

    return results


def print_results(results: List[QueryResult]):
    """Print aggregated benchmark results."""
    if not results:
        print("No results to report.")
        return

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    # Overall averages
    w_p3 = sum(r.weighted_metrics.get("precision@3", 0) for r in results) / len(results)
    r_p3 = sum(r.rrf_metrics.get("precision@3", 0) for r in results) / len(results)
    w_p5 = sum(r.weighted_metrics.get("precision@5", 0) for r in results) / len(results)
    r_p5 = sum(r.rrf_metrics.get("precision@5", 0) for r in results) / len(results)
    w_r3 = sum(r.weighted_metrics.get("recall@3", 0) for r in results) / len(results)
    r_r3 = sum(r.rrf_metrics.get("recall@3", 0) for r in results) / len(results)
    w_r5 = sum(r.weighted_metrics.get("recall@5", 0) for r in results) / len(results)
    r_r5 = sum(r.rrf_metrics.get("recall@5", 0) for r in results) / len(results)
    w_ndcg3 = sum(r.weighted_metrics.get("ndcg@3", 0) for r in results) / len(results)
    r_ndcg3 = sum(r.rrf_metrics.get("ndcg@3", 0) for r in results) / len(results)

    print(f"  precision@3 : Weighted={w_p3:.3f}  RRF={r_p3:.3f}  Delta={r_p3 - w_p3:+.3f} ({(r_p3 - w_p3)/max(w_p3, 1e-6)*100:+.1f}%)")
    print(f"  precision@5 : Weighted={w_p5:.3f}  RRF={r_p5:.3f}  Delta={r_p5 - w_p5:+.3f} ({(r_p5 - w_p5)/max(w_p5, 1e-6)*100:+.1f}%)")
    print(f"  recall@3    : Weighted={w_r3:.3f}  RRF={r_r3:.3f}  Delta={r_r3 - w_r3:+.3f} ({(r_r3 - w_r3)/max(w_r3, 1e-6)*100:+.1f}%)")
    print(f"  recall@5    : Weighted={w_r5:.3f}  RRF={r_r5:.3f}  Delta={r_r5 - w_r5:+.3f} ({(r_r5 - w_r5)/max(w_r5, 1e-6)*100:+.1f}%)")
    print(f"  ndcg@3      : Weighted={w_ndcg3:.3f}  RRF={r_ndcg3:.3f}  Delta={r_ndcg3 - w_ndcg3:+.3f} ({(r_ndcg3 - w_ndcg3)/max(w_ndcg3, 1e-6)*100:+.1f}%)")

    # Per-query wins
    rrf_wins = sum(1 for r in results if r.rrf_metrics.get("precision@3", 0) > r.weighted_metrics.get("precision@3", 0))
    w_wins = sum(1 for r in results if r.weighted_metrics.get("precision@3", 0) > r.rrf_metrics.get("precision@3", 0))
    ties = len(results) - rrf_wins - w_wins
    print(f"\n  Per-query wins (P@3): RRF={rrf_wins}  Weighted={w_wins}  Tie={ties}")

    # By query type
    types = {}
    for r in results:
        qtype = r.query_type
        if qtype not in types:
            types[qtype] = []
        types[qtype].append(r)

    print("\n  By query type:")
    for qtype, qs in types.items():
        w_p3 = sum(r.weighted_metrics.get("precision@3", 0) for r in qs) / len(qs)
        r_p3 = sum(r.rrf_metrics.get("precision@3", 0) for r in qs) / len(qs)
        print(f"    {qtype:25s}: Weighted={w_p3:.3f}  RRF={r_p3:.3f}  Delta={r_p3 - w_p3:+.3f}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holographic memory benchmark")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="Path to SQLite DB")
    parser.add_argument("--method", choices=["both", "rrf", "weighted"], default="both")
    parser.add_argument("--limit", type=int, default=10, help="Top-K results")
    parser.add_argument("--queries", type=str, nargs="*", default=None,
                        help="Filter by query types: exact, partial, multi_hop, conceptual, no_match")
    parser.add_argument("--longmemeval", action="store_true",
                        help="Use LongMemEval oracle dataset instead of golden queries")
    parser.add_argument("--lme-db", type=str, default=None,
                        help="Path to LongMemEval benchmark DB (default: same as --db)")
    parser.add_argument("--lme-subset", type=int, default=None,
                        help="Number of LongMemEval instances to use")
    parser.add_argument("--live", action="store_true",
                        help="Test live FactRetriever.search() pipeline directly")
    args = parser.parse_args()

    db_path = Path(args.db)

    # Load queries
    if args.longmemeval:
        lme_db = Path(args.lme_db) if args.lme_db else db_path
        queries_data = load_longmemeval_queries(lme_db, subset=args.lme_subset)
        if not queries_data:
            print("Error: No LongMemEval queries loaded. Prepare the DB first with prepare_longmemeval_db.py")
            sys.exit(1)
    else:
        queries_data = None  # use GOLDEN_QUERIES with filtering

    start = time.time()
    results = run_benchmark(
        db_path=db_path,
        method=args.method,
        limit=args.limit,
        query_types=args.queries,
        queries=queries_data,
        live=args.live,
    )
    elapsed = time.time() - start

    print_results(results)
    print(f"Benchmark completed in {elapsed:.2f}s")
