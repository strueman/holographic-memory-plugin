# Benchmark Findings — 2026-05-10

## SQLite Profiling Correction

**Finding:** cProfile reports "30 execute calls" per query, but only **3 are explicit Python calls**:
1. FTS5 MATCH query
2. `sqlite_master` check (for vec table)
3. Full fact fetch (JOIN facts_fts ON facts)

The remaining ~27 calls are **FTS5 internal index operations**:
- `SELECT pgno FROM facts_fts_idx WHERE segid=? AND term<=?` — B-tree navigation
- `SELECT sz FROM facts_fts_docsize WHERE id=?` — document size lookups

These are unavoidable FTS5 internals. To reduce them, you'd need to change the search backend entirely.

**Verification:** Use `conn.set_trace_callback()` to distinguish explicit vs internal calls:
```python
calls = []
conn.set_trace_callback(lambda sql, *a: calls.append(sql[:200]) if not sql.strip().startswith('--') else None)
```

## Tokenize Memoization Experiment

**Hypothesis:** Cache tokenize results by fact_id since facts don't change between queries.

**Result:** 2% **slower** — dict lookup overhead for 561 entries > tokenize cost (sub-millisecond).

**Verdict:** Reverted. Tokenize is already fast enough.

## HRR Batch Vectorization Experiment

**Hypothesis:** Pre-compute query HRR vector once, stack all candidate bytes into a single (n, dim) matrix, compute cosine similarity in one vectorized operation.

**Result:** 1.2% speedup — within noise. HRR encoding is only 2.4ms total; ONNX (13ms) dominates.

**Verdict:** Reverted. Adds complexity for no meaningful gain.

## Personal Memory Dataset v0.4.1 — With Vector Embeddings

**Setup:** 1694 facts, 181 queries, 100% vec coverage (backfilled via `store.backfill_embeddings()`).

**Results:**
- P@3: Weighted=0.297, RRF=0.287 — Weighted wins by 3.1%
- P@5: Weighted=0.213, RRF=0.218 — RRF wins by 2.1%
- R@5: Weighted=0.594, RRF=0.612 — RRF wins by 3.0%
- NDCG@3: Weighted=0.541, RRF=0.544 — RRF wins by 0.6%

**Per query type:**
| Type | Weighted | RRF | Delta |
|------|----------|-----|-------|
| personal-memory | 0.265 | 0.265 | 0.0% |
| temporal | 0.427 | 0.400 | -6.3% |
| comparative | 0.427 | 0.427 | 0.0% |
| multi-hop | 0.231 | 0.231 | 0.0% |
| entity-resolution | 0.200 | 0.200 | 0.0% |
| ambiguous | 0.258 | 0.212 | -17.8% |

**Key insight:** RRF's equal weighting dilutes strong FTS5 signals when vec adds noise. Temporal and ambiguous queries suffer because FTS5 is already strong for these, and vec dilutes the ranking.

## RRF Tuning Problem for 5-Signal Mode

**Problem:** RRF constant C=60 was designed for 2-signal mode (FTS5 + Jaccard). In 5-signal mode (FTS5 + Vec + Jaccard + HRR + Access), equal weighting is suboptimal.

**Observed behavior:**
- FTS5 is strong for temporal/personal queries → vec noise dilutes ranking
- Vec is strong for multi-hop/entity-resolution → FTS5 noise dilutes ranking
- Equal weighting doesn't adapt to signal quality

**Potential solutions:**
1. **Per-signal weighting:** Weight FTS5 higher for FTS5-strong queries, vec higher for vec-strong queries
2. **Signal quality gating:** Only include vec signal if vec candidates overlap with FTS5 candidates
3. **Dynamic C:** Adjust C based on number of active signals

**TODO:** This is the next optimization target after vec embeddings are confirmed essential.

## Optimization Decision Framework

| Speedup | Decision |
|---------|----------|
| < 5% | Revert — within noise |
| 5–10% | Marginal — only if trivial to implement |
| 10–20% | Worth it — document in benchmark tracker |
| > 20% | Major win — consider architectural changes |

**Always measure with and without** the optimization. The "speedup" might be a fluke from cache effects or measurement noise.

## Profile Command (Ready to Run)

```bash
python3 -c "
import cProfile, pstats, io
from retrieval import FactRetriever
from store import MemoryStore
store = MemoryStore('path/to/db')
retriever = FactRetriever(store)
# Warm up
retriever.search('test query', limit=10)
# Profile
profiler = cProfile.Profile()
profiler.enable()
for _ in range(10):
    retriever.search('test query', limit=10)
profiler.disable()
s = io.StringIO()
pstats.Stats(profiler, stream=s).sort_stats('cumulative').print_stats(20)
print(s.getvalue())
"
```

## SQLite Trace Command (Ready to Run)

```bash
python3 -c "
from store import MemoryStore
store = MemoryStore('path/to/db')
calls = []
store._conn.set_trace_callback(lambda sql, *a: calls.append(sql[:200]) if not sql.strip().startswith('--') else None)
retriever = FactRetriever(store)
retriever.search('test', limit=10)
print(f'Explicit SQL calls: {len(calls)}')
for c in calls: print(c)
"
```
