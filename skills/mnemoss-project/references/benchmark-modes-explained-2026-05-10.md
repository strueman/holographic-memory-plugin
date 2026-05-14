# Benchmark Modes Explained

> **Date:** 2026-05-10
> **Purpose:** Understanding the difference between candidate-pool and live benchmark modes

## Two Benchmark Modes

The holographic benchmark script (`holographic_benchmark.py`) has two modes:

### Mode 1: Candidate-Pool Comparison (default)

Uses `_get_candidates()` which calls `_fts_candidates()` and `_vec_candidates()` directly.
This is a **standalone** implementation that mirrors the retrieval pipeline but doesn't include:
- Chinese LIKE fallback (`_fts_candidates_with_chinese_fallback`)
- Phrase proximity scoring
- Evidence gap analysis
- Temporal decay

**Command:** `python benchmarks/holographic_benchmark.py --db DB --method both`

**Use case:** Comparing RRF vs weighted-sum on the same candidate pool. Good for isolating scoring method differences.

**Limitation:** Doesn't test the full `search()` pipeline — Chinese LIKE fallback is NOT included.

### Mode 2: Live Pipeline (`--live`)

Calls `FactRetriever.search()` directly, which includes the **full** pipeline:
- Chinese LIKE fallback via `_fts_candidates_with_chinese_fallback()`
- Phrase proximity scoring
- Evidence gap analysis
- Temporal decay
- All RRF signals

**Command:** `python benchmarks/holographic_benchmark.py --db DB --queries-db DB --live`

**Use case:** Testing the complete search experience, including Chinese support.

**Limitation:** Can't isolate scoring method differences (RRF vs weighted) because it calls the actual `search()` method.

## Key Discovery

When benchmarking Chinese support:
- **Candidate-pool mode:** Returns 0% P@3 because `_fts_candidates()` returns 0 for Chinese (no LIKE fallback)
- **Live mode:** Returns meaningful P@3 (e.g., 14%) because `search()` includes LIKE fallback

**Always use `--live` when testing features that modify the search pipeline** (like Chinese support).

## Benchmarking Chinese Support

```bash
# Live mode — tests full pipeline including Chinese LIKE fallback
python benchmarks/holographic_benchmark.py \
    --db ~/.hermes/benchmarks/chinese_cmrc2018.db \
    --queries-db ~/.hermes/benchmarks/chinese_cmrc2018.db \
    --method both \
    --live

# Candidate-pool mode — tests scoring methods only (no Chinese fallback)
python benchmarks/holographic_benchmark.py \
    --db ~/.hermes/benchmarks/chinese_cmrc2018.db \
    --queries-db ~/.hermes/benchmarks/chinese_cmrc2018.db \
    --method both
```

## Benchmark DB Format

Must have:
- `facts` table with full MemoryStore schema (including `facts_fts` virtual table)
- `queries` table with columns: `query_id`, `query`, `query_type`, `expected`
- `expected` column: JSON array of fact_ids

Conversion script: `benchmarks/create_chinese_benchmark.py`
