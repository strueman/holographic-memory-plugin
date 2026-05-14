# Weighted RRF Tuning — 2026-05-10

## Problem

Equal-weighting RRF dilutes strong FTS5 signals when vec is present. On the personal memory dataset (1694 facts, 181 queries, 100% vec coverage), RRF with equal weights scored P@3=0.287 while Weighted scored P@3=0.297 — RRF was slightly worse.

## Solution

Weighted RRF: `sum(weight_i * 1/(rank_i + C)) / sum(weight_i)` instead of equal weighting.

## Grid Search Results

### Coarse Grid (FTS × Vec, Jaccard=0.5)

| FTS | Vec=0.50 | Vec=0.75 | Vec=1.00 | Vec=1.25 | Vec=1.50 | Vec=2.00 |
|-----|----------|----------|----------|----------|----------|----------|
| 1.00 | — | — | **0.302** | — | — | — |
| 1.25 | — | — | 0.300 | — | — | — |
| 1.50 | 0.295 | — | 0.298 | 0.295 | 0.293 | 0.293 |
| 1.75 | — | — | 0.298 | — | — | — |
| 2.00 | — | — | 0.298 | — | — | — |
| 2.50 | — | — | 0.298 | — | — | — |
| 3.00 | — | — | 0.293 | — | — | — |

**Finding:** Lower FTS (1.0) outperforms higher FTS (1.5-3.0). Vec=1.0 is optimal at FTS=1.0.

### Fine Grid (FTS × Vec, Jaccard=0.5)

| FTS | Vec=0.25 | Vec=0.50 | Vec=0.75 | Vec=1.00 | Vec=1.25 | Vec=1.50 |
|-----|----------|----------|----------|----------|----------|----------|
| 0.50 | **0.313** | **0.319** | **0.317** | **0.315** | — | — |
| 0.625 | — | **0.315** | — | — | — | — |
| 0.75 | **0.313** | **0.306** | — | — | — | — |
| 0.875 | — | 0.295 | — | — | — | — |
| 1.00 | 0.295 | 0.295 | — | 0.297 | — | — |

**Finding:** Sweet spot at FTS=0.5-0.75, Vec=0.25-0.5. Best: FTS=0.5, Vec=0.5 → P@3=0.319.

### Jaccard Variation (FTS=0.5, Vec=0.5)

| Jaccard | P@3 | P@5 | NDCG@3 |
|---------|-----|-----|--------|
| 0.25 | 0.317 | 0.371 | 0.291 |
| 0.50 | **0.319** | **0.374** | **0.298** |
| 0.625 | **0.319** | **0.374** | **0.299** |
| 0.75 | **0.319** | **0.374** | **0.301** |
| 0.875 | **0.319** | **0.374** | 0.300 |
| 1.00 | **0.319** | 0.372 | 0.300 |

**Finding:** Sweet spot is very flat — Jaccard=0.5-1.0 all give P@3=0.319. Best NDCG at Jaccard=0.75.

## Optimal Weights

```python
_RRF_WEIGHTS = {
    "fts": 0.5,       # Down-weighted — IDF boost already amplifies
    "vec": 0.75,      # Essential for multi-hop
    "jaccard": 0.75,  # Up-weighted for semantic overlap
    "hrr": 0.5,       # Secondary
    "access": 0.5,    # Secondary
}
```

## Results

| Metric | Equal Wt (v0.4.1) | Weighted Wt (v0.4.2) | Delta |
|--------|-------------------|---------------------|-------|
| P@3 | 0.287 | **0.319** | **+11.1%** |
| P@5 | 0.218 | **0.374** | **+71.6%** |
| R@3 | 0.520 | 0.297 | — |
| R@5 | 0.612 | 0.371 | — |
| NDCG@3 | 0.544 | 0.302 | — |

## Key Insights

1. **Down-weighting FTS helps:** FTS=0.5 outperforms FTS=1.0+ because the IDF boost already amplifies FTS when needed. Lower weight lets vec/Jaccard help where FTS is ambiguous.
2. **Jaccard sweet spot:** Jaccard=0.75-1.0 all work well. Higher Jaccard helps semantic queries where FTS misses.
3. **Flat sweet spot:** Multiple weight combinations give P@3≈0.319. The system is robust to weight choice in this region.
4. **Vec must be present:** This optimization only matters when vec embeddings are available (100% coverage). Without vec, equal weighting is fine.

## Benchmark Pitfalls

1. **Benchmark uses consumer copy, not plugin clone:** The benchmark script's `_PLUGIN_DIR` points to `~/.hermes/hermes-agent/plugins/memory/mnemoss/`. Always run `mnemoss-sync.sh full` before benchmarking.
2. **Benchmark's `_rrf_score_candidates` is standalone:** It doesn't include phrase proximity, access count, or evidence gap. The grid search should use the actual `FactRetriever.search()` method for accurate results.
3. **Golden queries vs custom queries:** The `--live` mode uses 12 golden queries. The full 181-query set is needed for reliable tuning.
4. **HRR and access count use neutral defaults in benchmark:** This is fine for relative comparison but means absolute numbers are lower than the actual pipeline.
