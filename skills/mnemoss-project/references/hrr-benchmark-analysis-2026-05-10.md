# HRR Optimization Analysis (2026-05-10)

## Experiment: Batch HRR Vectorization

**Goal:** Reduce HRR encoding overhead by pre-computing query vector once and using numpy batch operations.

**Approach:**
1. Pre-compute `encode_text(query)` once (outside the candidate loop)
2. Collect all candidate HRR bytes
3. Stack into (n, dim) numpy matrix: `np.stack([np.frombuffer(v) for v in vectors])`
4. Vectorized cosine: `np.mean(np.cos(hrr_matrix - query_vec), axis=1)`
5. Map results back to fact_ids via dict

**Result: 1.2% speedup — within noise.**

## Warm Query Cost Breakdown (cProfile, 10 queries, ONNX warmed)

| Component | Time | % of total |
|-----------|------|-----------|
| ONNX (vec embedding) | 13.1ms | ~42% |
| SQLite execute (30 calls*) | 9.6ms | ~31% |
| Tokenize (330 calls) | 3.4ms | ~11% |
| FTS5 candidates | 2.6ms | ~8% |
| HRR encode_text (batch) | 2.4ms | ~8% |
| Evidence gap analysis | 0.8ms | ~3% |
| Phrase proximity | 0.6ms | ~2% |
| **Total warm** | **~32ms** | |
| **Total cold (ONNX init)** | **~55ms** | |

*See SQLite Profiling correction below.

## SQLite Profiling Correction (2026-05-10)

**Initial finding:** Profiler reported "30 execute calls" per query.

**Deep dive:** Used `conn.set_trace_callback()` to trace all SQLite operations. Found:
- Only **3 explicit Python-level execute calls** per query:
  1. FTS5 MATCH query
  2. `sqlite_master` check (for vec table existence)
  3. Full fact fetch (JOIN facts_fts ON facts)
- The "27 additional calls" were **FTS5 internal index operations** (prefixed with `--` in trace output)
  - `SELECT pgno FROM facts_fts_idx WHERE segid=? AND term<=?` — B-tree navigation
  - `SELECT sz FROM facts_fts_docsize WHERE id=?` — document size lookups
  - These are unavoidable FTS5 internal operations, not Python-level calls

**Conclusion:** SQLite is already minimal. The 9.6ms for "30 calls" is actually 3 explicit calls (~3.2ms) + 27 FTS5 internal ops (~6.4ms). Neither can be reduced without changing the DB schema or using a different search backend.

## Optimization Experiments

### HRR Batch Vectorization
- **Approach:** Pre-compute query vector, stack candidates into (n, dim) matrix, numpy vectorized cosine
- **Result:** 1.2% speedup — within noise
- **Why:** HRR encoding is only 2.4ms total; ONNX (13ms) dominates
- **Decision:** Reverted

### Tokenize Memoization
- **Approach:** Cache tokenize results by fact_id (facts don't change between queries)
- **Result:** 2% **slower** — dict lookup overhead for 561 entries exceeded tokenize cost
- **Why:** tokenize is already sub-millisecond (microseconds per fact via simple string ops)
- **Decision:** Reverted

## Cold vs Warm

- Cold start (ONNX load): ~55ms/query
- Warm: ~32ms/query
- Gap: ~23ms from ONNX session creation

## Optimization Candidates (in order of impact)

1. **ONNX warmup** — 13ms/query fixed cost. Options:
   - Batch multiple queries through ONNX at once
   - Use a smaller/faster embedding model
   - Cache embeddings for repeated queries
   - ONNX session reuse (already done, but cold start is unavoidable)

2. **SQLite execute calls** — 30 calls/query is excessive
   - Consolidate FTS5 + Jaccard + vec into fewer queries
   - Use query plans that return all needed data in one shot
   - Pre-compute and cache intermediate results

3. **Tokenize memoization** — 330 calls/query, all redundant
   - Cache tokenize results by fact_id (facts don't change between queries)
   - Could save ~3ms/query

4. **Evidence gap analysis** — 0.8ms/query
   - Only runs on first result, but could be skipped for fast queries

## Decision

**HRR vectorization reverted.** The complexity added does not justify the 1.2% speedup. Focus optimization efforts on ONNX and SQLite bottlenecks instead.
