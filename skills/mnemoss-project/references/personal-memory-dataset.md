# Personal Memory Dataset

Custom benchmark dataset for testing the holographic memory plugin with realistic personal memory queries.

## Dataset Structure

**Location:** `~/.hermes/benchmarks/personal_memory_benchmark.db`

**Contents:**
- 428 declarative facts (50-300 chars each)
- 63 realistic queries with expected fact_ids
- Categories: personal, work, finance, health, preferences, events, shopping, travel, hobbies, social, technology, education

## Generation

### Initial Dataset (187 facts)
Generated manually with `benchmarks/generate_personal_memory_dataset.py`.

### Expansion (241 facts)
Generated via 6 parallel `delegate_task` workers:

| Worker | Category | Facts |
|--------|----------|-------|
| 1 | Projects & work history | 40 |
| 2 | Research & learning | 40 |
| 3 | Personal life & relationships | 40 |
| 4 | Finance & purchases | 40 |
| 5 | Health & fitness | 40 |
| 6 | Preferences & habits | 40 |

Each worker received a prompt with format requirements, category coverage, and style guidelines. Workers returned JSON arrays of strings.

## Benchmark Results

### Expanded Dataset (428 facts, 63 queries)

| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | 0.286 | 0.296 | +3.7% |
| P@5 | 0.178 | 0.194 | +8.9% |
| R@3 | 0.757 | 0.780 | +3.1% |
| R@5 | 0.772 | 0.844 | +9.2% |
| NDCG@3 | 0.732 | 0.726 | -0.8% |

### Original Dataset (187 facts, 63 queries)

| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | 0.291 | 0.291 | 0.0% |
| P@5 | 0.178 | 0.194 | +8.9% |
| R@3 | 0.765 | 0.765 | 0.0% |
| R@5 | 0.772 | 0.844 | +9.2% |
| NDCG@3 | 0.737 | 0.721 | -2.2% |

### Analysis
- Slight P@3 degradation (0.291→0.286) with more corpus noise
- RRF maintains recall advantage on R@3 and R@5
- NDCG@3 nearly tied in expanded dataset
- Higher recall (0.765→0.757) indicates more retrievable facts

## Custom Dataset Format

SQLite DB with `queries` table:
```sql
CREATE TABLE queries (
    query_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT DEFAULT 'personal-memory',
    expected TEXT NOT NULL  -- JSON array of fact_ids
);
```

Run benchmark with:
```bash
python3 benchmarks/holographic_benchmark.py \
  --db PATH_TO_DB \
  --queries-db PATH_TO_DB \
  --method both
```

## Lessons Learned

1. **delegate_task works for parallel fact generation** — 6 workers ran in parallel, completed in ~140s total
2. **Kanban workers crash with --skills flag** — Use delegate_task instead for this class of task
3. **Facts should be 50-300 chars** — Longer facts hurt FTS5 tokenization
4. **Use "User" as subject** — Consistent naming improves retrieval
5. **Include specific dates/names/numbers** — Helps FTS5 match queries
6. **Mix categories** — Real personal memory spans many domains
7. **Add multi-hop queries** — Some queries require combining info from multiple facts
8. **Add no-match queries** — Some queries have no answer in the facts
