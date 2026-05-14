# Benchmark Pitfalls & Findings

Known issues and lessons from benchmarking the holographic memory plugin.

## Benchmark Script Fixes

### --queries-db Flag (v0.3.0)
Added `--queries-db` flag to load queries from a custom DB with a `queries` table. Also added `load_custom_queries()` function.

**Key:** Query format accepts both new keys (`query`, `expected`, `query_type`) and legacy keys (`expected_facts`, `type`, `description`) — mapped automatically.

### LME DB as Store Fix
Previously, `--lme-db` only loaded queries but the store still used the production DB (29 facts). Fixed: when `--lme-db` is provided, the store also uses that DB.

## Structural Link Pitfalls

### Threshold Changes Don't Affect Retrieval
Structural links (in `fact_links` table) are **only used for cross-level compression (consolidation)**, not retrieval. The retrieval pipeline uses FTS5 + vector search + Jaccard + RRF fusion.

**Impact:**
- Lowering thresholds from 0.65→0.55 (HRR) and 0.75→0.70 (embedding) does NOT change benchmark results
- Re-linking a corpus with new thresholds produces identical benchmark scores
- Threshold tuning affects consolidation quality, not retrieval accuracy

### Re-linking a Corpus
To test threshold impact, call `_create_links_for_fact(fact_id)` for each fact. The LME DB was re-linked with new thresholds (177K links vs 32K old) but benchmark results were identical.

## Dataset Pitfalls

### Synthetic vs Real Data
LongMemEval is synthetic conversational data (user/assistant turns, avg 1,215 chars). Not representative of real personal memory. Created custom dataset with short declarative facts (50-300 chars).

### Query Format Mismatches
The benchmark expects specific key names in query dicts:
- `expected` or `expected_facts` — list of fact_ids
- `type` or `query_type` — query category
- `description` — human-readable description (optional)

Custom DB queries table uses `expected` (JSON array of fact_ids).

## Kanban Failure

### Workers Crash with --skills Flag
Kanban workers crash with: `error: option --skills not recognized`

The `--skills` flag is not supported by the current hermes CLI version. Use `delegate_task` instead for parallel fact generation or dataset expansion tasks.

### delegate_task as Alternative
`delegate_task` worked perfectly for parallel fact generation:
- 6 workers ran in parallel
- Each received isolated context
- Completed in ~140s total
- No coordination needed
- Results aggregated manually

## Small Dataset Limitations

With 187-428 facts:
- P@3 is noisy (single fact changes move the needle)
- Recall is high (most facts are retrievable)
- Results may not generalize to larger corpora
- Need 1000+ facts for statistically meaningful results

## Benchmark Commands

```bash
# Custom dataset
python3 benchmarks/holographic_benchmark.py \
  --db PATH_TO_DB \
  --queries-db PATH_TO_DB \
  --method both

# LongMemEval (full)
python3 benchmarks/holographic_benchmark.py --longmemeval \
  --lme-db ~/.hermes/benchmarks/longmemeval_benchmark.db \
  --lme-subset 500

# Live pipeline test
python3 benchmarks/holographic_benchmark.py --longmemeval \
  --lme-db ~/.hermes/benchmarks/longmemeval_benchmark.db \
  --live --lme-subset 5
```
