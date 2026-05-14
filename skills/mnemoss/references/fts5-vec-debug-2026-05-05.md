# FTS5 + Vec Debugging Chain (2026-05-05)

## Problem
Benchmark results were identical after pulling latest plugin code. Si suspected vec search wasn't actually running.

## Investigation

### Step 1: Check if vec dependencies exist
- `sqlite_vec`: importable (v0.1.9)
- `onnxruntime`: **NOT installed** -- embedding model couldn't run
- `facts_vec` table: **did not exist** in benchmark DB

### Step 2: Install onnxruntime + backfill embeddings
```bash
pip install --break-system-packages onnxruntime
```
Backfilled all 10,960 facts (144s, ~14ms per embedding).

### Step 3: Re-run benchmark -- still identical results
Turns out the benchmark script is standalone -- it reimplements its own FTS5 + Jaccard + HRR scoring and never calls `retrieval.py`. The vec signal was never in the benchmark pipeline.

### Step 4: Test the real retrieval pipeline
Called `FactRetriever._fts_candidates()` directly:
- FTS5 returned **0 candidates** for every LongMemEval query
- Vec returned 90 candidates per query (working fine)
- Root cause: FTS5 MATCH treats full query as AND-expression. Long queries (20+ words) return 0 because no single fact contains all terms.

### Step 5: Fix FTS5 query preprocessing
Added `_prepare_fts5_query()` in `retrieval.py`:
1. Extracts multi-word capitalized phrases (kept as phrase matches)
2. Strips stop words, keeps content words >= 3 chars
3. Joins content words with OR for recall
4. Falls back to original query if processing yields nothing

Result: FTS5 now returns 30 candidates per query. Full pipeline (FTS5 + vec + Jaccard + HRR) achieves 82% hit rate on expected facts in top-3.

## Lessons
- Always verify vec is actually running by checking `facts_vec` table population
- The benchmark script is a standalone tool -- it doesn't test the real pipeline
- FTS5 MATCH needs query preprocessing for natural language queries
- When benchmark numbers don't change, check if the code path actually changed
