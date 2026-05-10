"""Benchmark entity-indexed retrieval vs plain FTS5.

Tests whether pre-extracting entities from facts improves retrieval
by doing entity-first lookups before falling back to FTS5 text search.

Usage:
    python entity_benchmark.py                          # use current DB
    python entity_benchmark.py --db benchmarks/...      # custom DB
    python entity_benchmark.py --longmemeval            # use LME data
    python entity_benchmark.py --longmemeval --lme-db ...
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Set

_DEFAULT_DB = Path.home() / ".hermes" / "memory_store.db"
_LME_DB = Path.home() / ".hermes" / "benchmarks" / "longmemeval_benchmark.db"

# Tokenization
_TOKEN_RE = re.compile(r'\b\w+\b', re.IGNORECASE)

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)
# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_ALLCAPS      = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b')
_RE_MIXED        = re.compile(r'\b([A-Z]{2,}\s+\d+\s+[A-Z]{2,})\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)

def _is_fragment(text: str) -> bool:
    """Check if text looks like a sentence fragment (not a real entity)."""
    fragments = {
        'm happy', "i'm", "i'll", "i've", "i'd", "i am",
        "it's", "there's", "that's", "you're", "you've",
        "let's", "we'll", "we're", "he's", "she's",
        "that's", "this is", "the fact", "the user", "the assistant",
        "the model", "the question", "the answer", "the topic",
    }
    return text.lower() in fragments

def _extract_entities(text: str) -> List[str]:
    """Extract entity candidates using multiple regex strategies.

    Rules (in order):
    1. Capitalized multi-word phrases  e.g. "John Doe"
    2. All-caps acronyms (2+ words)    e.g. "HRR", "GPU", "SSD"
    3. Mixed alphanumeric              e.g. "RX 7900 XTX"
    4. Double-quoted terms             e.g. "Python"
    5. Single-quoted terms             e.g. 'pytest'
    6. AKA patterns                    e.g. "Guido aka BDFL"

    Filters out sentence fragments and single short words.
    Returns deduplicated list preserving first-seen order.
    """
    seen: set = set()
    candidates: List[str] = []

    def _add(name: str) -> None:
        stripped = name.strip()
        if stripped and stripped.lower() not in seen and not _is_fragment(stripped):
            seen.add(stripped.lower())
            candidates.append(stripped)

    for m in _RE_CAPITALIZED.finditer(text):
        _add(m.group(1))

    for m in _RE_ALLCAPS.finditer(text):
        _add(m.group(1))

    for m in _RE_MIXED.finditer(text):
        _add(m.group(1))

    for m in _RE_DOUBLE_QUOTE.finditer(text):
        _add(m.group(1))

    for m in _RE_SINGLE_QUOTE.finditer(text):
        _add(m.group(1))

    for m in _RE_AKA.finditer(text):
        _add(m.group(1))
        _add(m.group(2))

    return candidates

# FTS5 helper
def compute_fts5_rank(conn: sqlite3.Connection, query: str, limit: int = 50) -> List[tuple]:
    query_tokens = tokenize(query)
    query_match = " OR ".join(query_tokens)
    rows = conn.execute(
        """
        SELECT facts_fts.rowid AS fact_id, facts_fts.rank as fts_rank, f.content, f.trust_score
        FROM facts_fts
        JOIN facts f ON f.fact_id = facts_fts.rowid
        WHERE facts_fts MATCH ?
        ORDER BY facts_fts.rank
        LIMIT ?
        """,
        (query_match, limit),
    ).fetchall()
    return rows

def compute_jaccard(query_tokens: List[str], fact_content: str) -> float:
    fact_tokens = set(tokenize(fact_content))
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    return len(query_set & fact_tokens) / len(query_set | fact_tokens)

# Scoring methods
def weighted_sum_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> List[dict]:
    """Original weighted-sum: FTS5 + Jaccard + trust."""
    query_tokens = tokenize(query)
    candidates = compute_fts5_rank(conn, query, limit=50)
    if not candidates:
        return []

    # Normalize ranks
    min_rank = min(r[1] for r in candidates)
    max_rank = max(r[1] for r in candidates)
    rank_range = max_rank - min_rank if max_rank != min_rank else 1.0

    results = []
    for fact_id, fts_rank, content, trust in candidates:
        fts_score = max(0.0, (fts_rank - max_rank) / rank_range) if rank_range else 1.0
        jaccard = compute_jaccard(query_tokens, content)
        score = 0.4 * fts_score + 0.3 * jaccard
        results.append({
            "fact_id": fact_id, "content": content, "score": score,
            "fts_score": fts_score, "jaccard": jaccard, "trust": trust,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

def rrf_search(
    conn: sqlite3.Connection,
    query: str,
    rrf_constant: int = 60,
    limit: int = 10,
) -> List[dict]:
    """RRF: rank-based fusion."""
    query_tokens = tokenize(query)
    candidates = compute_fts5_rank(conn, query, limit=50)
    if not candidates:
        return []

    # Compute Jaccard ranks
    jaccard_scores = [(fid, compute_jaccard(query_tokens, content))
                      for fid, rank, content, trust in candidates]
    jaccard_scores.sort(key=lambda x: x[1], reverse=True)
    jaccard_ranks = {fid: i+1 for i, (fid, j) in enumerate(jaccard_scores)}

    results = []
    for pos, (fact_id, rank, content, trust) in enumerate(candidates, start=1):
        j_r = jaccard_ranks.get(fact_id, len(candidates))
        fts_rrf = rrf_constant / (pos + rrf_constant)
        jac_rrf = rrf_constant / (j_r + rrf_constant)
        score = fts_rrf + jac_rrf
        results.append({
            "fact_id": fact_id, "content": content, "score": score,
            "trust": trust, "fts_position": pos, "jaccard_rank": j_r,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

def entity_boosted_search(
    conn: sqlite3.Connection,
    query: str,
    entity_boost: float = 50.0,
    limit: int = 10,
) -> List[dict]:
    """Entity-indexed + FTS5 hybrid retrieval.

    1. Extract entities from the query
    2. Direct lookup: which facts contain these entities?
    3. FTS5 search for broader matching
    4. Merge: entity matches get a rank boost, FTS5 candidates follow

    The entity_boost parameter controls how strongly entity matches
    are prioritized over text matches.
    """
    query_entities = _extract_entities(query)
    query_tokens = tokenize(query)

    # Entity lookup: which facts contain query entities?
    entity_fact_ids: Dict[str, List[int]] = {}
    if query_entities:
        try:
            placeholders = ",".join("?" * len(query_entities))
            rows = conn.execute(
                f"""
                SELECT e.name, fe.fact_id
                FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE e.name IN ({placeholders})
                """,
                query_entities,
            ).fetchall()
            for name, fact_id in rows:
                entity_fact_ids.setdefault(name, []).append(fact_id)
        except sqlite3.OperationalError:
            pass

    entity_fact_set: Set[int] = set()
    for fids in entity_fact_ids.values():
        entity_fact_set.update(fids)

    # FTS5 candidates
    fts_candidates = compute_fts5_rank(conn, query, limit=50)

    # Build unified candidate list with entity boost
    candidate_scores: Dict[int, dict] = {}

    min_rank = min(r[1] for r in fts_candidates) if fts_candidates else 0
    max_rank = max(r[1] for r in fts_candidates) if fts_candidates else 0
    rank_range = max_rank - min_rank if max_rank != min_rank else 1.0

    for fact_id, fts_rank, content, trust in fts_candidates:
        fts_score = max(0.0, (fts_rank - max_rank) / rank_range) if rank_range else 1.0
        jaccard = compute_jaccard(query_tokens, content)
        base_score = 0.4 * fts_score + 0.3 * jaccard

        is_entity_match = fact_id in entity_fact_set
        if is_entity_match:
            # Entity match: entity_score = entity_boost * trust, as a rank-based boost
            # Higher entity_boost = stronger entity preference
            # Clamp to [1, entity_boost] to ensure entity wins over non-entity
            entity_score = entity_boost * trust
            # Entity matches always beat non-entity matches (unless base is extremely high)
            score = max(base_score, entity_score)
        else:
            score = base_score

        candidate_scores[fact_id] = {
            "fact_id": fact_id, "content": content, "score": score,
            "trust": trust, "is_entity_match": is_entity_match,
        }

    results = sorted(candidate_scores.values(), key=lambda x: x["score"], reverse=True)
    return results[:limit]

# Metrics
def precision_at_k(retrieved_ids: List[int], expected_ids: List[int], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for fid in top_k if fid in expected_ids) / min(k, len(top_k))

def recall_at_k(retrieved_ids: List[int], expected_ids: List[int], k: int) -> float:
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(expected_ids)) / len(expected_ids)

def compute_metrics(retrieved: List[dict], expected_facts: List[int], k=3):
    retrieved_ids = [r["fact_id"] for r in retrieved]
    return {
        "precision@3": precision_at_k(retrieved_ids, expected_facts, 3),
        "recall@3": recall_at_k(retrieved_ids, expected_facts, 3),
        "entity_hits": sum(1 for r in retrieved if r.get("is_entity_match", False) and r["fact_id"] in set(expected_facts)),
    }

# Query loaders
GOLDEN_QUERIES = [
    {"query": "holographic memory plugin uses HRR with 1024-dim vectors", "expected_facts": [2], "type": "exact", "description": "Architecture fact"},
    {"query": "RRF replaced weighted-sum scoring with rank-based fusion C=60", "expected_facts": [3], "type": "exact", "description": "RRF fact"},
    {"query": "strueman holographic-memory-plugin GitHub repo", "expected_facts": [14], "type": "exact", "description": "GitHub repo"},
    {"query": "what memory architectures improve holographic recall", "expected_facts": [5, 6, 7, 8, 9], "type": "partial", "description": "Research papers"},
    {"query": "what research papers have HIGH applicability to holographic", "expected_facts": [5, 6, 8], "type": "multi_hop", "description": "HIGH applicability"},
    {"query": "how does the memory plugin integrate with the agent", "expected_facts": [17], "type": "conceptual", "description": "Integration"},
    {"query": "what's for dinner tonight", "expected_facts": [], "type": "no_match", "description": "No match"},
]

def load_longmemeval_queries(db_path: Path, subset: Optional[int] = None) -> List[dict]:
    if not db_path.exists():
        print(f"Error: DB not found at {db_path}")
        return []
    conn = sqlite3.connect(str(db_path))
    existing = conn.execute("SELECT COUNT(*) FROM lme_instances").fetchone()[0]
    if existing == 0:
        print("  No LME instances. Run prepare_longmemeval_db.py first.")
        conn.close()
        return []
    queries = []
    for row in conn.execute("SELECT question_id, question, question_type, expected_fact_ids FROM lme_instances"):
        if subset and len(queries) >= subset:
            break
        expected = json.loads(row[3]) if row[3] else []
        queries.append({
            "query": row[1], "expected_facts": expected,
            "type": row[2] or "unknown",
            "description": f"LME: {row[0]}",
        })
    conn.close()
    return queries

# Main
@dataclass
class QueryResult:
    query: str
    expected_facts: List[int]
    weighted: List[int]
    rrf: List[int]
    entity: List[int]
    w_metrics: dict
    r_metrics: dict
    e_metrics: dict

def run(db_path: Path, queries: List[dict], subset: Optional[int] = None):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    all_results = []

    for q in queries:
        query_text = q["query"]
        expected = q["expected_facts"]

        w = weighted_sum_search(conn, query_text, limit=10)
        r = rrf_search(conn, query_text, limit=10)
        e = entity_boosted_search(conn, query_text, limit=10)

        all_results.append(QueryResult(
            query=query_text, expected_facts=expected,
            weighted=[x["fact_id"] for x in w],
            rrf=[x["fact_id"] for x in r],
            entity=[x["fact_id"] for x in e],
            w_metrics=compute_metrics(w, expected),
            r_metrics=compute_metrics(r, expected),
            e_metrics=compute_metrics(e, expected),
        ))
    conn.close()
    return all_results

def print_results(results: List[QueryResult]):
    print("=" * 80)
    print("ENTITY INDEXED RETRIEVAL vs RRF vs WEIGHTED-SUM")
    print("=" * 80)

    for i, r in enumerate(results):
        print(f"\nQuery {i+1}: {r.query}")
        print(f"  Expected: {r.expected_facts}")
        print(f"  Weighted: {r.weighted[:5]} | P@3={r.w_metrics['precision@3']:.2f} R@3={r.w_metrics['recall@3']:.2f}")
        print(f"  RRF:      {r.rrf[:5]} | P@3={r.r_metrics['precision@3']:.2f} R@3={r.r_metrics['recall@3']:.2f}")
        print(f"  Entity:   {r.entity[:5]} | P@3={r.e_metrics['precision@3']:.2f} R@3={r.e_metrics['recall@3']:.2f} (entity_hits={r.e_metrics['entity_hits']})")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for method, key in [("weighted", "w_metrics"), ("rrf", "r_metrics"), ("entity", "e_metrics")]:
        p3 = sum(r.__dict__[key]["precision@3"] for r in results) / len(results)
        r3 = sum(r.__dict__[key]["recall@3"] for r in results) / len(results)
        eh = sum(r.__dict__[key]["entity_hits"] for r in results)
        print(f"  {method:12s}: P@3={p3:.3f}  R@3={r3:.3f}  entity_hits={eh}")

    # Entity wins
    entity_wins = sum(1 for r in results
                      if r.e_metrics["precision@3"] > r.r_metrics["precision@3"]
                      and r.e_metrics["precision@3"] > 0)
    rrf_wins = sum(1 for r in results
                   if r.r_metrics["precision@3"] > r.e_metrics["precision@3"]
                   and r.r_metrics["precision@3"] > 0)
    ties = sum(1 for r in results
               if r.e_metrics["precision@3"] == r.r_metrics["precision@3"])
    print(f"\n  Entity vs RRF (P@3): Entity wins={entity_wins}  RRF wins={rrf_wins}  Tie={ties}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity-indexed retrieval benchmark")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--longmemeval", action="store_true")
    parser.add_argument("--lme-db", type=str, default=None)
    parser.add_argument("--subset", type=int, default=50)
    args = parser.parse_args()

    if args.longmemeval:
        db = Path(args.lme_db) if args.lme_db else _LME_DB
        queries = load_longmemeval_queries(db, subset=args.subset)
    else:
        db = Path(args.db) if args.db else Path.home() / ".hermes" / "memory_store.db"
        queries = GOLDEN_QUERIES

    print(f"DB: {db}")
    start = time.time()
    results = run(db, queries, subset=args.subset)
    elapsed = time.time() - start
    print_results(results)
    print(f"\nCompleted in {elapsed:.2f}s")
