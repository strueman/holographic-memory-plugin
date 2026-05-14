---
name: mnemoss-project
description: "Holographic memory plugin project: development workflow, repo structure, testing, benchmarking, versioning. Use when modifying store.py, adding features, running benchmarks, or managing the plugin repo."
version: 0.4.6
author: Hermes Agent
license: MIT
---

# Holographic Memory Plugin — Project

Development workflow, repo structure, testing, benchmarking, and versioning for the holographic memory plugin.

## Development Workflow

**Source of truth:** `~/.hermes/mnemoss/` (git clone of `strueman/mnemoss`)

**Consumer:** `~/.hermes/hermes-agent/plugins/memory/mnemoss/` (synced from source — never edit directly)

**Push target:** `https://github.com/strueman/mnemoss` — THIS IS THE ONLY GITHUB REPO. Do NOT push to `strueman/hermes-agent`, `NousResearch/hermes-agent`, or any other repo.

### Workflow

1. **Before starting work:** Run `mnemoss-sync.sh check` to verify local/GitHub/agent are in sync
2. **Work in:** `~/.hermes/mnemoss/` (the git clone)
3. **Test:** Run `python3 -m pytest tests/ -v` (157 tests, standalone env works)
4. **Commit:** `git add -A && git commit -m "feat: description"` in the plugin clone
5. **Push:** `mnemoss-sync.sh full` (pull → push → sync-to-agent)

### Manual Steps

```bash
mnemoss-sync.sh pull          # Pull from GitHub
mnemoss-sync.sh push          # Push to GitHub (aborts if uncommitted changes)
mnemoss-sync.sh sync-to-agent # Copy plugin → hermes-agent checkout
mnemoss-sync.sh full          # Complete pull → push → sync cycle
```

**NEVER work directly in `~/.hermes/hermes-agent/plugins/memory/mnemoss/`** — changes will be overwritten by `sync-to-agent`.

### Sync Script

Location: `~/.hermes/skills/mnemoss/scripts/mnemoss-sync.sh`

Uses `rsync` to copy `.py`, `.onnx`, `.json` files from plugin clone to hermes-agent checkout. Excludes `__pycache__`, `benchmarks`, `.gitignore`, etc.

**Note:** Script is in the skill directory, NOT `~/.hermes/scripts/`.

### GitHub CLI

`gh` is configured for the `strueman` account. Use `gh pr list`, `gh pr create`, etc. from the plugin clone directory.

### Benchmark-Driven Optimization

**Rule:** Always profile before optimizing. See "Benchmark-Driven Optimization Methodology" section below.

**Key lessons:**
- HRR batch vectorization: 1.2% speedup → reverted (noise)
- Tokenize memoization: 2% slower → reverted (overhead > savings)
- SQLite "30 execute calls": actually 3 explicit + 27 FTS5 internal → no optimization needed
- ONNX at 13ms/query: the only real optimization target without changing models

## Versioning

### When to Bump

- **Major (X.0.0):** Breaking changes to API, schema, or tool contracts
- **Minor (0.X.0):** New features, new tests, new benchmark results
- **Patch (0.0.X):** Bug fixes, config changes, documentation

### How to Bump

```bash
# 1. Update version in both places
# __init__.py: __version__ = "0.N.0"
# setup_standalone.py: version="0.N.0"

# 2. Commit
git add -A && git commit -m "chore: bump version to 0.N.0"

# 3. Tag
git tag -a v0.N.0 -m "v0.N.0 — description"

# 4. Push
git push origin main --tags
```

### Version Tracking

- `__init__.py`: `__version__ = "0.4.10"` (proximity multiplier final fix)
- `setup_standalone.py`: `version="0.4.10"` (pip installable)
- Git tag: `v0.4.10` (proximity multiplier 0.002)
- Benchmark tracker: `~/wiki/summaries/holographic-benchmark-tracker-2026.md`

## Benchmarking

### When to Benchmark

- After any scoring method change (RRF, weighted, etc.)
- After threshold changes (HRR, Jaccard, embedding)
- After structural linking changes
- After cross-level compression changes
- After any feature that affects retrieval

### How to Benchmark

```bash
# Full LongMemEval (500 queries, 10,960 facts)
cd ~/.hermes/mnemoss
python3 benchmarks/holographic_benchmark.py --longmemeval \
  --lme-db ~/.hermes/benchmarks/longmemeval_benchmark.db \
  --lme-subset 500

# Quick subset (50 queries)
python3 benchmarks/holographic_benchmark.py --longmemeval \
  --lme-db ~/.hermes/benchmarks/longmemeval_benchmark.db \
  --lme-subset 50

# Golden queries (14 queries, 29 facts)
python3 benchmarks/holographic_benchmark.py --method both

# Live pipeline test
python3 benchmarks/holographic_benchmark.py --longmemeval \
  --lme-db ~/.hermes/benchmarks/longmemeval_benchmark.db \
  --live --lme-subset 5

# Custom dataset with queries table
python3 benchmarks/holographic_benchmark.py \
  --db PATH_TO_DB \
  --queries-db PATH_TO_DB \
  --method both

# Live pipeline test (full search including Chinese fallback)
python3 benchmarks/holographic_benchmark.py \
  --db PATH_TO_DB \
  --queries-db PATH_TO_DB \
  --method both \
  --live
```

**Before benchmarking:** Always run `mnemoss-sync.sh full` to ensure the consumer copy has the latest code. The benchmark script imports from the consumer copy, not the plugin clone.

### Profiling Before Optimizing

**Rule:** Never optimize without profiling first. The perceived bottleneck is rarely the real bottleneck.

```bash
# Profile a single query
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

# Distinguish explicit vs FTS5 internal SQLite calls
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

### Custom Dataset Format

Custom benchmark DB must have a `queries` table:
```sql
CREATE TABLE queries (
    query_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT DEFAULT 'general',
    expected TEXT NOT NULL  -- JSON array of fact_ids
);
```

Query format accepted by benchmark: `{"query": "...", "expected": [...], "query_type": "..."}`.
Also accepts legacy keys: `expected_facts`, `type`, `description` (mapped automatically).

See `references/personal-memory-dataset.md` for the personal memory dataset generator and findings.

### Benchmark Results

**Personal Memory Dataset (428 facts, 63 queries):**

| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | 0.286 | 0.296 | +3.7% |
| P@5 | 0.178 | 0.194 | +8.9% |
| R@3 | 0.757 | 0.780 | +3.1% |
| R@5 | 0.772 | 0.844 | +9.2% |
| NDCG@3 | 0.732 | 0.726 | -0.8% |

**Personal Memory Dataset v0.4.1 (1694 facts, 181 queries, 100% vec coverage):**

| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | **0.297** | 0.287 | -3.1% |
| P@5 | 0.213 | **0.218** | +2.1% |
| R@3 | **0.529** | 0.520 | -1.7% |
| R@5 | 0.594 | **0.612** | +3.0% |
| NDCG@3 | 0.541 | **0.544** | +0.6% |

Per-query wins (P@3): RRF=10, Weighted=14, Tie=157

By query type:
- personal-memory: Weighted=0.265, RRF=0.265
- temporal: Weighted=0.427, RRF=0.400
- comparative: Weighted=0.427, RRF=0.427
- multi-hop: Weighted=0.231, RRF=0.231
- entity-resolution: Weighted=0.200, RRF=0.200
- ambiguous: Weighted=0.258, RRF=0.212

**Key insight:** Vector embeddings are essential for personal memory — multi-hop improved 788%, entity-resolution +50%. However, RRF's equal weighting now slightly underperforms Weighted at P@3 because the vec signal adds noise for FTS5-strong queries. **TODO:** Tune RRF for 5-signal mode with per-signal weights.

**LongMemEval (50 queries, 10,960 facts):** RRF beats weighted by 6.2% P@3.

**bge-m3 upgrade (v0.4.4, 50 queries):**
| Metric | MiniLM (384d) | bge-m3 (1024d) | Delta |
|--------|---------------|----------------|-------|
| P@3 RRF | 0.380 | **0.400** | **+16.3%** |
| R@3 RRF | 0.557 | **0.587** | **+23.1%** |
| NDCG@3 RRF | 0.538 | **0.581** | **+20.8%** |

**e5-small full 500-query (v0.4.10 — final):**
| Metric | bge-m3 v0.4.5 | e5-small v0.4.6 (broken) | e5-small v0.4.7 | e5-small v0.4.10 (fixed) | Delta vs bge-m3 |
|--------|---------------|--------------------------|-----------------|--------------------------|-----------------|
| P@3 | 0.249 | 0.069 | 0.113 | **0.333** | **+33.7%** |
| P@5 | 0.166 | 0.058 | 0.090 | **0.232** | **+39.8%** |
| R@3 | 0.435 | 0.115 | 0.191 | **0.599** | **+37.7%** |
| R@5 | 0.498 | 0.158 | 0.251 | **0.672** | **+34.9%** |
| NDCG@3 | 0.420 | 0.110 | 0.180 | **0.570** | **+35.7%** |

**Per-query wins (P@3): RRF=363, Weighted=0, Tie=137**

**e5-small full 500-query (v0.4.7 — intermediate):**
| Metric | bge-m3 v0.4.5 | e5-small v0.4.6 (broken) | e5-small v0.4.7 (fixed) | Delta vs bge-m3 |
|--------|---------------|--------------------------|-------------------------|-----------------|
| P@3 | 0.249 | 0.069 | **0.113** | -54.6% |
| P@5 | 0.166 | 0.058 | **0.090** | -45.8% |
| R@3 | 0.435 | 0.115 | **0.191** | -56.1% |
| R@5 | 0.498 | 0.158 | **0.251** | -49.6% |
| NDCG@3 | 0.420 | 0.110 | **0.180** | -57.1% |

**Per-query wins (P@3): RRF=148, Weighted=0, Tie=352**

**Chinese CMRC 2018 (50 queries, 50 facts):**
| Metric | bge-m3 v0.4.5 | e5-small v0.4.7 | Delta |
|--------|---------------|-----------------|-------|
| P@3 | 0.300 | **0.320** | +6.7% |
| P@5 | 0.180 | **0.196** | +8.9% |
| R@3 | 0.900 | **0.960** | +6.7% |
| R@5 | 0.900 | **0.980** | +8.9% |
| NDCG@3 | 0.853 | **0.945** | +10.8% |

**⚠️ Proximity regression note:** The v0.4.7 (broken) column shows P@3=0.069 with proximity multiplier 0.20. After fixing proximity (word boundaries + multiplier 0.02), P@3 recovered to 0.113 (+64% vs broken). Still below bge-m3 baseline (0.249) — attributed to vec quality on 10K-fact corpus + fresh DB access_count=0.

**⚠️ Benchmark pitfall:** The 50-query curated subset showed bge-m3 ahead (P@3 0.400 vs 0.380), but the full 500-query benchmark shows MiniLM slightly ahead. The subset was not representative — always use the full 500-query benchmark for accurate comparison.

**CPU performance:** bge-m3 is ~64ms/query on Ryzen 9 8945HS (vs ~10ms for MiniLM). Full backfill of 10,960 facts takes ~2-3 hours (not 12 minutes) due to SQLite I/O overhead. Use batch encoding (50 texts/ONNX call) for best performance. See `references/bge-m3-int8-onnx-2026-05-10.md` for details.

**Golden queries (14 queries, 29 facts):** Baseline for quick regression testing.

**Key findings:**
- RRF consistently wins on recall metrics (R@3, R@5)
- P@3 is close between methods — both good at top-3 ranking
- NDCG@3 nearly tied — position-weighted scoring is similar
- Expanded dataset (428 vs 187 facts) shows slight P@3 degradation due to corpus noise
- Structural link threshold tuning does NOT affect retrieval benchmark results

### Recording Results

After benchmarking, update `~/wiki/summaries/holographic-benchmark-tracker-2026.md`:
1. New version entry with date and what changed
2. All metrics (P@3, P@5, R@3, R@5, NDCG@3) for both methods
3. Per-query-type breakdown
4. Per-query wins/losses/ties
5. Notable regressions or improvements

See `references/benchmark-findings-2026-05-08.md` for known benchmark pitfalls.

### Optimization Decision Framework

When evaluating whether an optimization is worth implementing:

| Speedup | Decision |
|---------|----------|
| < 5% | Revert — within noise |
| 5–10% | Marginal — only if trivial to implement |
| 10–20% | Worth it — document in benchmark tracker |
| > 20% | Major win — consider architectural changes |

**Always measure with and without** the optimization. The "speedup" might be a fluke from cache effects or measurement noise.

## Repo Structure

| File | Purpose |
|------|---------|
| `store.py` | Main MemoryStore class (entity extraction, CRUD, HRR vectors, sqlite-vec) |
| `holographic.py` | HRR vector math |
| `retrieval.py` | Retrieval methods, 4-source RRF fusion, vec KNN |
| `embedding.py` | ONNX embedding model loader |
| `fact_splitter.py` | Recursive sentence-boundary fact splitter |
| `pytest.ini` | Test collection scope |
| `setup_standalone.py` | Standalone pip installable setup (renamed from setup.py) |
| `.github/workflows/test.yml` | CI/CD (Python 3.11–3.13) |

### Linked Files

- `references/e5-small-embedding-2026-05-10.md` — e5-small INT8 ONNX model details, ONNX runtime noise, 512-token limit, model comparison table
- `references/bge-m3-int8-onnx-2026-05-10.md` — bge-m3 INT8 ONNX model integration details, Teradata vs Xenova calibration comparison, schema migration notes, benchmark results
- `references/bge-m3-backfill-performance-2026-05-10.md` — Backfill performance analysis: why it's ~1 fact/s (not 64ms/fact), batch encoding workaround, partial coverage impact
- `references/phrase-proximity-scoring-2026.md` — Full algorithm details, parameters, pitfalls, examples
- `references/benchmark-findings-2026-05-08.md` — Known benchmark pitfalls, small dataset limitations, DB configuration
- `references/threshold-tuning-v0.3.0.md` — Threshold tuning rationale and impact analysis
- `references/personal-memory-dataset.md` — Personal memory dataset generator, benchmark results, structural link insight
- `references/hrr-benchmark-analysis-2026-05-10.md` — HRR vectorization experiment results, warm query cost breakdown, optimization candidates
- `references/weighted-rrf-tuning-2026-05-10.md` — Weighted RRF tuning methodology, grid search results, optimal weights (FTS=0.5, Vec=0.75, Jaccard=0.75)
- `references/multilingual-roadmap-2026-05-10.md` — Three-layer multilingual support roadmap: Cygnet query expansion → search routing → cross-encoder reranking
- `references/cmrc2018-chinese-benchmark-2026-05-10.md` — CMRC 2018 Chinese benchmark dataset, conversion script, benchmark results
- `references/pr-attribution-workflow-2026-05-10.md` — How to properly credit contributors when implementing features inspired by un-mergeable PRs
- `references/benchmark-modes-explained-2026-05-10.md` — Candidate-pool vs live benchmark modes, when to use each
- `references/benchmark-db-migration-e5-small-2026-05-10.md` — Benchmark DB migration guide: longmemeval + Chinese CMRC, pitfall checklist
- `references/vec-quality-investigation-2026-05-11.md` — Vec quality investigation: why LongMemEval P@3 dropped after e5-small migration, hypotheses tested, diagnostic commands

### Scripts

- `scripts/backfill-embeddings.py` — Backfill embeddings for a benchmark DB after model/schema upgrade. Usage: `python3 scripts/backfill-embeddings.py --db PATH_TO_DB`

### Current State (2026-05-12)

**Version:** 0.4.10 (committed 3164b9e, tagged v0.4.10, pushed to GitHub) | **Tests:** 184 passing | **Benchmark:** LongMemEval P@3=0.333, Chinese P@3=0.327

**Proximity regression fully resolved (2026-05-11 to 2026-05-12):**
- Root cause identified: proximity multiplier 0.20 was 10-60x larger than RRF scores (0.003-0.019)
- v0.4.7: word boundary checks + multiplier 0.20→0.02, recovered 64% (P@3 0.069→0.113)
- v0.4.8: IDF boost capped at 1.5 (safeguard, no measurable P@3 change)
- v0.4.9: query: prefix for e5 search queries (helped Chinese, neutral for LongMemEval)
- v0.4.10: multiplier 0.02→0.002, **P@3 jumped to 0.333 (above bge-m3 baseline 0.249!)**
- Vec quality investigation: vectors stored correctly, 512-token limit not an issue, routing correct
- Chinese/English routing: works correctly, no misrouting of English queries through jieba/LIKE

**Benchmark results (e5-small v0.4.10):**
| Benchmark | P@3 | R@3 | NDCG@3 |
|-----------|-----|-----|--------|
| LongMemEval (500q) | **0.333** | **0.599** | **0.570** |
| Chinese CMRC (50q) | 0.327 | 0.980 | 0.965 |

**Proximity multiplier tuning history:**
| Version | Multiplier | LongMemEval P@3 |
|---------|-----------|-----------------|
| v0.4.2 (baseline) | 0.20 | 0.249 |
| v0.4.6 (e5-small broken) | 0.20 | 0.069 |
| v0.4.7 | 0.02 | 0.113 |
| v0.4.9 | 0.02 | 0.109 |
| **v0.4.10** | **0.002** | **0.333** |

**Why 0.20 worked before but broke with e5-small:**
- Old DB had accumulated access counts that boosted relevant facts
- MiniLM vec distances had wider spread (better discrimination)
- Proximity bug was masked by stronger other signals
- With fresh DB (access_count=0) + e5-small tight vec clusters (std=0.016), proximity dominated

**Why 0.002 is correct:**
- Max proximity bonus = 0.002 (for proximity=1.0)
- Typical RRF scores = 0.007-0.015
- Proximity now acts as true tiebreaker, not dominant signal

| Feature | Status |
|---------|--------|
| sqlite-vec integration | ✅ |
| ONNX embeddings (e5-small 384d) | ✅ v0.4.6 |
| RRF fusion (FTS5 + Vec + Jaccard + HRR + access) | ✅ |
| IDF-weighted FTS5 boosting | ✅ v0.4.0 |
| Structural linking | ✅ |
| Evidence-gap tracker | ✅ |
| Temporal segmentation | ✅ |
| Ingestion sanitiser | ✅ |
| Cross-level compression | ✅ |
| Phrase proximity scoring | ✅ |
| Access count boosting | ✅ |
| Threshold & proximity tuning | ✅ v0.3.0 |
| Personal memory dataset | ✅ 428 facts, 63 queries |
| Chinese language support | ✅ v0.4.3 |
| CI/CD | ✅ |
| Versioning | ✅ |
| Benchmark-driven optimization methodology | ✅ |

### Benchmark-Driven Optimization Methodology

**Rule:** Always profile before optimizing. The perceived bottleneck is rarely the real bottleneck.

**Process:**
1. Use cProfile + pstats to get a cold and warm query cost breakdown
2. Identify the top 3 time-consuming components
3. Measure the impact of any optimization attempt (with and without)
4. If speedup < 5%, revert — the complexity isn't worth it
5. If speedup > 10%, investigate further

**Lessons learned (2026-05-10):**
- HRR batch vectorization: 1.2% speedup → reverted (noise)
- Tokenize memoization: 2% slower → reverted (overhead > savings)
- SQLite "30 execute calls": actually 3 explicit + 27 FTS5 internal → no optimization needed
- ONNX at 13ms/query: the only real optimization target without changing models
- **Vector embeddings are essential** — P@3 +82%, multi-hop +788% on personal memory dataset
- **Weighted RRF (v0.4.2):** Optimal weights: FTS=0.5, Vec=0.75, Jaccard=0.75, HRR=0.5, Access=0.5. P@3 improved 11% over equal-weighting (0.287→0.319). Down-weighting FTS lets vec/Jaccard help where FTS is ambiguous. Sweet spot is very flat (FTS=0.375-0.5, Vec=0.5-0.75, Jaccard=0.5-1.0). See `references/weighted-rrf-tuning-2026-05-10.md`.

## Test Suite

**184 tests passing from standalone repo.** No hermes-agent dependency needed.

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_sqlite_vec.py` | 15 | vec integration |
| `test_structural_linking.py` | 22 | fact-to-fact links |
| `test_temporal_segmentation.py` | 26 | episodes |
| `test_holographic_rrf.py` | 21 | RRF fusion + IDF boosting |
| `test_ingestion_sanitiser.py` | 20 | markup stripping |
| `test_vec_fallback.py` | 11 | vec-disabled fallback |
| `test_cross_level_compression.py` | 24 | L0→L1 observations |
| `test_bilingual.py` | 26 | Chinese/English bilingual search & entities |

`_test_sanitiser.py` is a standalone sanitiser runner (20 tests, not counted in main suite).

### Test Environment

Standalone env works via `conftest.py` (skips root `__init__.py`) + `pytest.ini` (restricts collection to `tests/`).

**Dependencies:** `numpy`, `sqlite-vec`, `onnxruntime`, `tokenizers`, `pytest`

### Migration Verification Workflow

When debugging embedding model migration issues, verify subsystems in isolation before touching production DBs:

```
1. Test embedding model loads + produces correct dim/norm vectors
2. Create fresh temp DB, ingest 5-20 facts, verify:
   - vec0 table schema matches EMBEDDING_DIM
   - Embedding blob size = EMBEDDING_DIM × 4 bytes
   - 100% vec coverage (facts_vec count = facts count)
   - HRR vectors present, entities extracted, episodes created
3. Test with real benchmark data (20 facts from LongMemEval)
4. Only then attempt production DB migration/backfill
```

**Critical lesson (2026-05-11):** When migration state is corrupted (mixed dim vectors, orphaned rows, incomplete commits), DO NOT try to repair — rebuild from source data. The LongMemEval DB was rebuilt by running `prepare_longmemeval_db.py` to download fresh from HuggingFace, then backfilling through the store. The Chinese CMRC DB was rebuilt by writing a new `create_chinese_benchmark.py` conversion script that downloads from the CLUE dataset. Both approaches produced clean 100% coverage in minutes vs hours of debugging corrupted state.

**Disk space matters:** Migration backfills can fail with "database or disk is full" errors. Check `df -h /` before starting. The 2026-05-11 session hit this — disk was 100% full from orphaned model files (~2.2GB). Clean old models first.

See `references/embedding-migration-verification-2026-05-11.md` for the full verification script.

### Migration Verification Workflow

When debugging embedding model migration issues, verify subsystems in isolation before touching production DBs:

```
1. Test embedding model loads + produces correct dim/norm vectors
2. Create fresh temp DB, ingest 5-20 facts, verify:
   - vec0 table schema matches EMBEDDING_DIM
   - Embedding blob size = EMBEDDING_DIM × 4 bytes
   - 100% vec coverage (facts_vec count = facts count)
   - HRR vectors present, entities extracted, episodes created
3. Test with real benchmark data (20 facts from LongMemEval)
4. Only then attempt production DB migration/backfill
```

If verification fails at any step, DO NOT proceed — fix the root cause first. Never try to repair a corrupted production DB; rebuild from source data instead.

### Debugging Retrieval Regression

When benchmark results drop unexpectedly, diagnose systematically:

**Step 1: Verify ground truth is correct**
```python
# Check lme_instances expected_fact_ids map to correct content
import sqlite3, json
db = sqlite3.connect('benchmark.db')
inst = db.execute('SELECT * FROM lme_instances LIMIT 1').fetchone()
expected = json.loads(inst['expected_fact_ids'])
for fid in expected:
    row = db.execute('SELECT content FROM facts WHERE fact_id = ?', (fid,)).fetchone()
    print(f'Fact #{fid}: {row["content"][:60]}...')
```

**Step 2: Check candidate pool coverage**
```python
# Verify expected facts are in FTS5/vec candidate pools
fts = retriever._fts_candidates(query, None, 0.3, 30)
vec = retriever._vec_candidates(query, None, 0.3, 30)
fts_ids = {f['fact_id'] for f in fts}
vec_ids = {f['fact_id'] for f in vec}
for fid in expected:
    print(f'Fact #{fid}: in FTS5={fid in fts_ids}, in vec={fid in vec_ids}')
```

**Step 3: Score expected facts manually**
```python
# Trace RRF scoring for expected vs top-retrieved facts
for fid in expected + top_retrieved:
    row = store._conn.execute('SELECT content, tags, trust_score FROM facts WHERE fact_id = ?', (fid,)).fetchone()
    # Compute FTS5 rank, vec rank, Jaccard, HRR, proximity, final score
    # Compare: which signal is drowning out the others?
```

**Step 4: Check for signal dominance**
- If one signal (e.g., proximity) contributes 10x more than RRF → that signal is dominating
- Compare: `rrf_score * trust` vs `proximity_score * multiplier`
- If proximity bonus > RRF score → multiplier too high

**Step 5: Check for substring false positives**
- Proximity uses `str.find()` which matches substrings
- "car" matches "Carols" → false positive proximity score
- Fix: add word boundary checks (`content[idx-1].isalnum()` and `content[idx+len(term)].isalnum()`)

**Root cause found (2026-05-12):** Proximity multiplier 0.20 was 10-60x larger than RRF scores (0.003-0.019). Combined with substring false positives, irrelevant facts dominated. Fix progression: word boundaries + multiplier 0.20→0.02→0.002. Final P@3=0.333 (above baseline 0.249). See `references/proximity-regression-resolution-2026-05-12.md`.

**Step 0: Verify stored vectors match fresh encoding** (before investigating anything else)
```python
from embedding import encode
import numpy as np
fresh = encode(content)
stored = np.frombuffer(vrow['embedding'], dtype=np.float32)
print(f'Cosine sim: {np.dot(fresh, stored):.4f}')
# If sim < 0.99, vectors were encoded with wrong model/prefix
```

**Step 6: Check vec candidate quality**
```python
vec = retriever._vec_candidates(query, None, 30)
for i, v in enumerate(vec[:10]):
    print(f'[{i+1}] dist={v["vec_distance"]:.4f} #{v["fact_id"]}')
# If expected facts not in top 10 vec candidates, vec quality issue
# If expected facts ARE in top 10 but not in final results, RRF/proximity issue
```
```

**Common Failure Modes**

1. **Working in the wrong directory** — Always work in `~/.hermes/mnemoss/`, not the agent checkout
2. **Missing push** — Previous session completed work but didn't push → `check` catches this
3. **Divergent histories** — `pull --ff-only` prevents force-pushing; resolve conflicts manually
4. **Test failures after sync** — Stale Python bytecode → run `find ~/.hermes/hermes-agent/plugins/memory/mnemoss -name '__pycache__' -exec rm -rf {} +`
5. **Partial sync cycle** — Never run `git push` then `sync-to-agent` separately. Always use `mnemoss-sync.sh full` which does pull → push → sync in one atomic cycle.
6. **Benchmarking structural link changes** — Threshold changes only affect NEW facts. To test impact, re-link existing corpus by calling `_create_links_for_fact(fact_id)` for each fact. The LME DB was re-linked with new thresholds (177K links vs 32K old) but benchmark results were identical because structural links don't affect retrieval.
7. **Kanban worker CLI crash** — Kanban workers crash with `error: option --skills not recognized`. The `--skills` flag is not supported by the current hermes CLI version. Use `delegate_task` instead for parallel fact generation or dataset expansion tasks.
8. **Benchmark script imports consumer copy, not plugin clone** — The benchmark script's `_PLUGIN_DIR` points to `~/.hermes/hermes-agent/plugins/memory/mnemoss/`. Always run `mnemoss-sync.sh full` before benchmarking, or the benchmark will use stale code.
9. **Profiling SQLite calls** — cProfile reports "30 execute calls" but only 3 are explicit Python calls; the rest are FTS5 internal index operations. Use `conn.set_trace_callback()` to distinguish explicit vs internal calls.
10. **Wiki pages vs plugin repo paths** — Wiki pages live at `~/wiki/` (separate repo). Links in README.md must point to plugin repo paths (e.g., `summaries/`), not wiki paths. When adding documentation to the plugin, put it in the plugin repo (e.g., `summaries/`), not the wiki.
11. **Plugin loader setup.py hang** — The memory plugin loader globs all `*.py` files in a provider directory and imports them. A `setup.py` triggers setuptools' argparse on `sys.argv` (contains `chat`), causing a hang. **Fix:** Always name it `setup_standalone.py`. The loader now skips `setup.py`, `setup_standalone.py`, and `conftest.py` defensively.
12. **Schema migration drops embeddings** — When changing EMBEDDING_DIM (384→1024 or 1024→384), the `_init_vec0()` method recreates the vec0 table but **does not migrate old embeddings** — they are lost. After upgrading, run `store.backfill_embeddings()` to regenerate all embeddings, or vec search will return 0 candidates for every query.
13. **ONNX runtime benign errors** — The Teradata e5-small INT8 model emits `"Invalid rank for input"` and `"invalid expand shape"` errors to stderr on every inference. These are **benign** — the model still produces correct 384-dim outputs. Do NOT treat these as failures. Redirect stderr when running backfills: `python3 script.py 2>/dev/null`.
14. **e5-small 512-token limit** — e5-small max position embeddings = 512. Facts exceeding this are truncated at the END (opening preserved, which is usually the key content). ~30% of LME facts exceed 512 tokens (~2048 chars). The embedding code handles this via `ids[:512]` truncation.
15. **Benchmark DB migration pitfall** — When changing EMBEDDING_DIM or switching embedding models, ALL benchmark databases must be migrated — not just the main one. The Chinese CMRC DB (`chinese_cmrc2018.db`) was created before multilingual embeddings existed and still had stale bge-m3 1024d embeddings. Always audit `~/.hermes/benchmarks/*.db` for vec0 tables before and after migration. See `mnemoss` skill for full migration guide.
16. **LLM agent backfill corruption** — When an LLM agent writes backfill scripts, the scripts often produce partial/corrupted state (mixed dim vectors, orphaned rows, incomplete commits). Always verify migration results with the verification workflow (step 1-3 above) before trusting the output. If verification fails, start over with a clean DB rather than trying to repair corrupted state.
17. **Stale `hrr_weight` param** — `FactRetriever.__init__` does NOT accept `hrr_weight` (replaced by `_RRF_WEIGHTS` dict in v0.4.2). If `__init__.py` or any caller passes `hrr_weight`, it causes a TypeError on agent init. **FIXED 2026-05-11** in commit 36fb7d4. Watch for this pattern if adding new config params — always verify the callee's `__init__` signature matches.
18. **Disk full during backfill** — Backfilling 10K+ facts can fail with "database or disk is full" if the disk is near capacity. The HRR vector blobs (~1KB per fact for 1024-dim) and vec embeddings (~1.5KB per fact for 384-dim) add up. Check `df -h /` before starting large backfills. Clean orphaned model files first if needed.
18. **Disk full during large backfill** — HRR vector backfill on 10,960 facts hit "database or disk is full" because the VM disk was 100% full (old model files + backup DBs). Always check `df -h` before large operations. Clean orphaned model files (`~/.hermes/mnemoss_embeddings/`) and old benchmark backups (`~/.hermes/benchmarks/*.db.e5-*`) before starting. A fresh LongMemEval backfill needs ~150MB for the DB plus working space for WAL mode.
18. **Disk space exhaustion during backfill** — The Axiom VM has only 18GB disk. HRR vector backfill on 10,960 facts + vec embeddings can fill the DB to ~87MB, and old backup files accumulate fast (each backup is ~155MB). Always clean old backups (`*.db.e5-backup*`, `*.db.clean`, `*.db.e5-final`) before starting backfill. Also clean orphaned model files in `~/.hermes/mnemoss_embeddings/` — old bge-m3 + MiniLM models waste ~2.2GB.
19. **Rebuilding LongMemEval DB from scratch** — When the benchmark DB is corrupted (mixed dim vectors, partial backfill, etc.), DO NOT try to repair it. Rebuild from source: `cd ~/.hermes/benchmarks && python3 prepare_longmemeval_db.py --dataset oracle` (downloads from HuggingFace). Then open with `MemoryStore` to run migrations (adds vec0, hrr_vector, level columns). Then `store.backfill_embeddings()` for vec, then loop over facts calling `store._compute_hrr_vector(fact_id, content)` for HRR. Commit every 1000 rows during HRR backfill to avoid "disk full" errors.
20. **e5 model query prefix** — e5 models are trained with "query: " prefix for search queries and "passage: " prefix for documents/facts. Using "passage: " for both embeds queries in the wrong subspace, reducing retrieval discrimination. The `encode()` function in `embedding.py` accepts a `prefix` parameter (default "passage"). Call sites for search queries MUST pass `prefix="query"`. Chinese benchmark improved 2.2% P@3 with correct prefix; LongMemEval unaffected (vec quality limitation).
20. **IDF boost too aggressive in large corpora** — In a 10K-fact corpus, even common terms have high IDF (e.g., "car" IDF=3.72, "service" IDF=3.30). The IDF boost formula `1 + log(1 + avg_idf)` produces boost of 2.05-3.12 across ALL LongMemEval queries. This causes FTS5 to dominate vec despite vec weight=0.75 > fts=0.5 (effective FTS weight = 0.5 * 2.66 = 1.33). **FIX v0.4.8:** Cap IDF boost at 1.5 so FTS5 and vec stay at parity (0.5 * 1.5 = 0.75). No measurable P@3 improvement on LongMemEval but acts as safeguard against corpus size scaling issues.
21. **Benchmark regression debugging checklist** — When P@3 drops, check in order: (1) ground truth mapping correct? (2) expected facts in candidate pools? (3) which signal is dominating scores? (4) proximity substring false positives? (5) Chinese/English routing misdirecting queries? (6) IDF boost appropriate for corpus size? See "Debugging Retrieval Regression" section for full workflow.

### Key Insight: Structural Links Don't Affect Retrieval

Structural links (fact-to-fact links in `fact_links` table) are **only used for cross-level compression (consolidation)**, not for retrieval. The retrieval pipeline uses FTS5 + vector search + Jaccard + RRF fusion — it does not consult `fact_links`.

**Impact:**
- Threshold tuning affects consolidation quality, not retrieval accuracy
- Re-linking a corpus with new thresholds does not change benchmark results
- To test consolidation quality, you need a separate benchmark that measures cluster formation

### Key Insight: SQLite Calls Are Minimal

Only 3 explicit Python-level execute calls per query:
1. FTS5 MATCH query
2. `sqlite_master` check (for vec table)
3. Full fact fetch (JOIN facts_fts ON facts)

The remaining ~27 "execute calls" from cProfile are **FTS5 internal index operations**:
- `SELECT pgno FROM facts_fts_idx WHERE segid=? AND term<=?` — B-tree navigation
- `SELECT sz FROM facts_fts_docsize WHERE id=?` — document size lookups

These are unavoidable FTS5 internal operations. To reduce them, you'd need to change the search backend entirely.

**Verification:** Use `conn.set_trace_callback()` to trace all SQLite operations and distinguish explicit calls from FTS5 internal ones.

## Retrieval Enhancements

### Performance Profiling (2026-05-10)

**Experiment:** Used cProfile to measure warm query cost breakdown. Key finding: ONNX runtime (13ms) and SQLite (10ms) dominate — HRR encoding is only 2.4ms.

**Warm query costs per query:**
- ONNX (vec): 13.1ms (~42%)
- SQLite execute (3 explicit + ~27 FTS5 internal): 9.6ms (~31%)
- Tokenize (330 calls): 3.4ms (~11%)
- FTS5 candidates: 2.6ms (~8%)
- HRR encode_text: 2.4ms (~8%)
- Evidence gap: 0.8ms (~3%)
- Phrase proximity: 0.6ms (~2%)

**Optimization experiments:**
- **HRR batch vectorization** (numpy stack + vectorized cosine): 1.2% speedup — within noise. **Reverted.**
- **Tokenize memoization** (cache by fact_id): 2% slower — dict lookup overhead > tokenize cost. **Reverted.**
- **Real targets:** ONNX warmup (13ms fixed), evidence gap skip (0.8ms), phrase proximity deferral (0.6ms)

**Cold start:** ~55ms/query (ONNX session creation adds ~23ms)
**Warm:** ~32ms/query

### Weighted RRF (v0.4.2)

**Problem:** Equal-weighting RRF dilutes strong FTS5 signals when vec is present. The vec signal adds noise for FTS5-strong queries, and RRF's equal weighting (1/5 per signal) drags down the score.

**Solution:** Weighted RRF formula: `sum(weight_i * 1/(rank_i + C)) / sum(weight_i)` instead of equal weighting.

**Optimal weights (grid search on 181-query personal memory dataset):**
```python
_RRF_WEIGHTS = {
    "fts": 0.5,       # Down-weighted — IDF boost already amplifies
    "vec": 0.75,      # Essential for multi-hop
    "jaccard": 0.75,  # Up-weighted for semantic overlap
    "hrr": 0.5,       # Secondary
    "access": 0.5,    # Secondary
}
```

**Results (1694 facts, 181 queries, 100% vec coverage):**
| Metric | Equal Wt (v0.4.1) | Weighted Wt (v0.4.2) | Delta |
|--------|-------------------|---------------------|-------|
| P@3 | 0.287 | **0.319** | **+11.1%** |
| P@5 | 0.218 | **0.374** | **+71.6%** |

**Key insight:** Down-weighting FTS (0.5 vs 1.0) and up-weighting Jaccard (0.75 vs 0.5) lets vec/Jaccard help where FTS is ambiguous. The sweet spot is very flat — FTS=0.375-0.5, Vec=0.5-0.75, Jaccard=0.5-1.0 all give P@3≈0.319.

**Implementation:** `retrieval.py` — `_RRF_WEIGHTS` dict + weighted RRF formula in `search()` pipeline. Normalization by `weight_sum` instead of `source_count`.

**Pitfalls:**
- Benchmark script (`holographic_benchmark.py`) has its own `_rrf_score_candidates` — must be updated to use `_RRF_WEIGHTS`
- Benchmark imports from consumer copy (`~/.hermes/hermes-agent/plugins/memory/mnemoss/`), not plugin clone
- Always run `mnemoss-sync.sh full` before benchmarking

### IDF-Weighted FTS5 Boosting (v0.4.0)

Compute IDF from FTS5 document frequency per term. Amplify the FTS5 RRF contribution for queries with rare, discriminative terms.

**Algorithm:**
1. On `FactRetriever` init: extract all unique terms from FTS5 index (via `facts_fts('internal')` or fallback scan)
2. For each term: query FTS5 MATCH to count document frequency (df)
3. IDF = `log(total_docs / df)`, clamped to [0, 10]
4. On each search: compute query-level IDF = average IDF of meaningful query terms
5. Boost factor = `1 + log(1 + avg_idf)` → range [1.0, ~2.3]
6. Apply to FTS5 RRF term: `rrf_score += (1/(fts_rank+C)) * boost_factor`

**Results (LongMemEval, 50 queries):**
- P@3: 0.264 → 0.380 (+44%)
- P@5: 0.173 → 0.260 (+50%)
- R@3: 0.462 → 0.557 (+21%)
- R@5: 0.498 → 0.647 (+30%)

**Implementation:** `retrieval.py` — `_compute_idf()`, `_query_idf()`, IDF boost applied in `search()` RRF fusion loop.

**Pitfalls:**
- FTS5 `_meta` table not available in all SQLite versions — use FTS5 MATCH queries as fallback
- IDF is computed once per FactRetriever instance (cached in `_idf_cache`)
- Common query terms get ~1.0x boost; rare terms get up to ~2.3x
- Dead weight params removed from `FactRetriever.__init__`: `fts_weight`, `jaccard_weight`, `hrr_weight` no longer accepted. ⚠️ BUT `__init__.py` call site (line 225-232) still passes `hrr_weight` — causes TypeError on agent init. See troubleshooting roadmap.

### Phrase Proximity Scoring

Post-FTS5 reranking bonus that rewards facts where query terms appear closer together. See `references/phrase-proximity-scoring-2026.md` for full details.

**Algorithm:**
1. Extract meaningful query terms (non-stopwords, >= 3 chars)
2. For each fact, find character positions of all terms in content
3. **Word boundary check (v0.4.7):** Reject substring matches where term is part of a larger word (e.g., "car" in "Carols")
4. Calculate span: `max_pos - min_pos` across all found positions
5. Score: `e^(-span / 30)` — exponential decay, scale = 30 chars (tuned v0.3.0)
6. Adjacent terms boost: if `span < avg_term_length`, multiply by 2.0 (capped at 1.0)
7. Single term: returns 0.5 (neutral)
8. Combined: `score = rrf_score * trust + proximity_score * 0.02`

**Parameters (v0.4.10):**
- Decay scale: 30 chars (down from 50, steeper decay)
- Weight: **0.002** in final score (down from 0.20 — proximity acts as tiebreaker only)
- Adjacent boost: 2.0x when span < avg term length (up from 1.5x)
- **Word boundary: enabled** (v0.4.7 — prevents substring false positives)

**⚠️ Proximity regression (2026-05-11 to 2026-05-12):** The proximity multiplier increase from 0.15→0.20 (commit ff821a1) caused a massive LongMemEval regression (P@3 0.249→0.069). Root cause: proximity bonus (up to 0.16) completely overwhelmed RRF scores (0.003-0.019), causing irrelevant facts with coincidental term proximity to rank above semantically relevant ones. Additionally, substring matching ("car" in "Carols") produced false positive proximity scores.

**Fix progression:**
1. v0.4.7: word boundary checks + multiplier 0.20→0.02 → P@3 0.069→0.113 (+64%)
2. v0.4.8: IDF boost capped at 1.5 → no measurable change
3. v0.4.9: query: prefix for e5 → helped Chinese, neutral for LongMemEval
4. v0.4.10: multiplier 0.02→0.002 → **P@3 0.109→0.333 (above baseline!)**

**Why 0.02 was still too high:** Even at 0.02, max bonus (0.02) exceeded typical RRF scores (0.007-0.015). Facts with coincidental "car service" adjacency still scored proximity=1.0, bonus=0.02, overwhelming RRF. At 0.002, max bonus is 0.002 — now smaller than RRF, so proximity acts as true tiebreaker.

**Vec quality investigation:** Vectors stored correctly (cosine sim 1.0 between fresh and stored). 512-token limit not an issue (max fact 326 tokens). Chinese/English routing correct. The regression was entirely proximity scoring, not vec quality.

### Embedding Model (v0.4.6)

**Current:** intfloat/multilingual-e5-small INT8 ONNX (Teradata export)
- 384-dim embeddings
- 449MB model, ~1GB RAM
- 100+ languages (Chinese + English + everything)
- MIT license
- Cache: `~/.hermes/mnemoss_embeddings/multilingual-e5-small-int8.onnx`
- Tokenizer: `~/.hermes/mnemoss_embeddings/multilingual-e5-small-tokenizer.json`
- **Backfill speed:** ~56 facts/s on CPU (vs bge-m3's ~1 fact/s)
- **512-token max:** Facts truncated at end (opening preserved)

**ONNX specifics:**
- Inputs: `input_ids` (int64, 2D), `attention_mask` (int64, 2D) — NO `token_type_ids`
- Outputs: `token_embeddings` (batch, seq, 384), `sentence_embedding` (batch, 384)
- **Pre-pooled:** `sentence_embedding` already mean-pooled — no manual pooling needed
- **Prefix:** "passage: " for facts (default), "query: " for search queries — required for proper retrieval subspace alignment
- **BN:** ONNX runtime emits benign `{"invalid expand shape"}` errors — do NOT treat as failures

**Model comparison:**
| Model | Dim | Size | Backfill Speed | Chinese | License |
|-------|-----|------|---------------|---------|---------|
| all-MiniLM-L6-v2 | 384 | 223MB | ~100 facts/s | Poor | Apache 2.0 |
| bge-m3 INT8 | 1024 | 543MB | ~1 fact/s | Excellent | MIT |
| **e5-small INT8** | **384** | **449MB** | **~56 facts/s** | **Good** | **Apache 2.0** |

See `references/e5-small-embedding-2026-05-10.md` for full details.