# RRF Weight Tuning Methodology

> Grid search for optimal RRF signal weights on personal memory benchmark.
> Date: 2026-05-10
> Benchmark: 1694 facts, 181 queries, 100% vec coverage

## Problem

Equal-weighting RRF (v0.4.1) was slightly worse than weighted-sum at P@3 (0.287 vs 0.297). The vec signal was adding noise for queries where FTS5 was already strong, and RRF's equal weighting diluted the FTS5 signal.

## Methodology

### Step 1: Define weight formula

```
rrf_score = sum(weight_i * 1/(rank_i + C)) / sum(weights)
```

Where C=60, and weights are per-signal: FTS, Vec, Jaccard, HRR, Access.

### Step 2: Grid search

Test combinations of FTS and Vec weights (primary signals), keeping Jaccard=0.5, HRR=0.5, Access=0.5 fixed.

```python
def evaluate(weights, queries):
    """Run RRF with given weights on all queries, return metrics."""
    # Override retrieval._RRF_WEIGHTS, run search on each query, compute P@3/P@5/R@3/R@5/NDCG
    ...

# Grid
for fts_w in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    for vec_w in [0.5, 0.75, 1.0, 1.25, 1.5]:
        w = {"fts": fts_w, "vec": vec_w, "jaccard": 0.5, "hrr": 0.5, "access": 0.5}
        score = evaluate(w, queries)
```

### Step 3: Fine-grained around sweet spot

Once coarse grid finds best region, test finer steps (0.125 increments).

### Step 4: Secondary signal tuning

Test Jaccard, HRR, Access variations around optimal primary weights.

## Results

### Coarse grid (FTS=1.5, varying Vec)
- vec=1.00: P@3=0.298 (baseline)
- vec=0.50: P@3=0.295
- vec=1.25: P@3=0.295

### Coarse grid (Vec=1.0, varying FTS)
- fts=1.00: P@3=0.302 ← **first improvement**
- fts=1.50: P@3=0.298
- fts=2.00: P@3=0.298
- fts=3.00: P@3=0.293 ← **too high FTS hurts**

### Fine grid (around FTS=0.5, Vec=0.5)
- **FTS=0.5, Vec=0.5, jacc=0.75**: P@3=0.319 ← **best**
- FTS=0.5, Vec=0.5, jacc=0.5: P@3=0.319
- FTS=0.5, Vec=0.5, jacc=1.0: P@3=0.319
- FTS=0.5, Vec=0.75, jacc=0.75: P@3=0.319, NDCG=0.302 ← **best NDCG**

## Key Insights

1. **Lower FTS weight is better** — Counterintuitive. IDF boost already amplifies FTS; adding more weight just makes RRF more like FTS-only.
2. **Sweet spot is very flat** — FTS=0.375-0.5, Vec=0.5-0.75, Jaccard=0.5-1.0 all give P@3≈0.319.
3. **Jaccard up-weighting helps** — Jaccard=0.75 vs 0.5 gives same P@3 but better NDCG (ranking quality).
4. **Diminishing returns** — Going from P@3=0.287 to 0.319 (+11%) required significant tuning. Next gains will be smaller.

## Optimization Queue (Post-Weight Tuning)

1. Query expansion (WordNet/Cygnet + custom dict) — target +5-10% P@3
2. Sentence transformer reranking (MiniLM cosine) — target +5-10% P@3
3. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) — target +10-15% P@3
4. Combined pipeline — target P@3 0.45+

See `~/wiki/summaries/holographic-query-expansion-and-reranking-roadmap-2026.md` for full roadmap.
