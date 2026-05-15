# Chinese Benchmark — bge-m3 Embeddings Results

## Setup

**Dataset:** CMRC 2018 trial subset — 50 facts, 50 queries
**Query types:** 38 entity, 9 general, 1 numeric, 2 reasoning
**Embeddings:** bge-m3 INT8 ONNX (1024-dim, 100% coverage)
**Method:** Live `FactRetriever.search()` (includes Chinese LIKE fallback)
**Backfill:** `scripts/backfill-embeddings.py --db ~/.hermes/benchmarks/chinese_cmrc2018.db`

## Results

### Before bge-m3 (no embeddings, 0% vec coverage)

| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | 0.000 | 0.140 | — |
| P@5 | 0.000 | 0.140 | — |
| R@3 | 0.000 | 0.140 | — |
| R@5 | 0.000 | 0.140 | — |
| NDCG@3 | 0.000 | 0.140 | — |

Per-query wins (P@3): RRF=7, Weighted=0, Tie=43

### After bge-m3 (1024-dim, 100% vec coverage)

| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | 0.000 | **0.300** | **+114%** |
| P@5 | 0.000 | **0.180** | **+28%** |
| R@3 | 0.000 | **0.900** | **+543%** |
| R@5 | 0.000 | **0.900** | **+543%** |
| NDCG@3 | 0.000 | **0.853** | **+509%** |

Per-query wins (P@3): RRF=45, Weighted=0, Tie=5

### By Query Type (after bge-m3)

| Type | Count | P@3 |
|------|-------|-----|
| entity | 38 | 0.298 |
| general | 9 | 0.296 |
| numeric | 1 | 0.333 |
| reasoning | 2 | 0.333 |

## Key Findings

1. bge-m3 embeddings dramatically improve Chinese retrieval
2. Entity queries: P@3=0.298, general: P@3=0.296 — consistent across types
3. Chinese P@3 (0.300) now comparable to Personal Memory English (0.319) despite much smaller corpus (50 vs 1,694 facts)
4. FTS5 returns 0 for pure Chinese — LIKE fallback + bge-m3 vec fills the gap
5. RRF found 45/50 queries with P@3 win (vs 7/50 without embeddings)

## Comparison with English Benchmarks (all bge-m3)

| Dataset | Facts | Queries | P@3 (RRF) |
|---------|-------|---------|-----------|
| LongMemEval | 10,960 | 500 | 0.400 |
| Personal Memory | 1,694 | 181 | 0.319 |
| **Chinese CMRC 2018** | **50** | **50** | **0.300** |

Chinese P@3 is comparable to English despite 34x fewer facts, confirming bge-m3 handles Chinese well.
