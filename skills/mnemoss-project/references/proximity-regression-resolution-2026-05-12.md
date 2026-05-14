# Proximity Regression Resolution (2026-05-12)

## Summary

LongMemEval P@3 dropped from 0.249 (bge-m3 baseline) to 0.069 after e5-small migration. Full investigation and resolution across 4 commits (v0.4.7-v0.4.10). Final P@3=0.333, **above baseline**.

## Root Cause

Proximity multiplier 0.20 was 10-60x larger than RRF scores (0.003-0.019). Combined with substring false positives ("car" matching "Carols"), irrelevant facts with coincidental term proximity dominated rankings.

## Investigation Path

### Hypothesis 1: Vec embeddings stored incorrectly
- **Tested:** Compared fresh encode() vs stored vectors — cosine sim=1.0, identical
- **Result:** NOT the issue. Vectors stored correctly.

### Hypothesis 2: 512-token truncation
- **Tested:** Checked token length distribution across 10,960 facts
- **Result:** NOT the issue. Max fact=326 tokens, zero truncation.

### Hypothesis 3: Chinese/English routing misdirecting queries
- **Tested:** Checked _has_chinese() on English benchmark queries
- **Result:** NOT the issue. Routing works correctly.

### Hypothesis 4: BM25/IDF boost too aggressive
- **Tested:** Capped IDF boost at 1.5 (was 2.05-3.12 across all queries)
- **Result:** No measurable P@3 change. Acts as safeguard only.

### Hypothesis 5: Proximity scoring dominating (CORRECT)
- **Tested:** Manual RRF score trace showed proximity bonus 0.158 vs RRF 0.003
- **Diagnosis:** "car service" adjacent in "private car service" → proximity=1.0, bonus=0.02
- **Fix progression:**
  1. v0.4.7: Word boundary checks + multiplier 0.20→0.02 → P@3 0.069→0.113 (+64%)
  2. v0.4.8: IDF cap at 1.5 → no change
  3. v0.4.9: query: prefix for e5 → helped Chinese (+2.2%), neutral LongMemEval
  4. v0.4.10: multiplier 0.02→0.002 → **P@3 0.109→0.333**

## Why 0.02 Was Still Too High

Even at 0.02, max proximity bonus (0.02) exceeded typical RRF scores (0.007-0.015). A fact with proximity=1.0 got bonus=0.02, still larger than RRF scores of relevant facts. At 0.002, max bonus=0.002 — now smaller than RRF, so proximity acts as tiebreaker only.

## Why 0.20 Worked Before But Broke with e5-small

1. Old DB had accumulated access counts that boosted relevant facts
2. MiniLM vec distances had wider spread (better discrimination)
3. Proximity bug was masked by stronger other signals
4. Fresh DB (access_count=0) + e5-small tight vec clusters (std=0.016) meant proximity dominated

## Diagnostic Commands

```python
# Check proximity scores for expected vs top-retrieved facts
for fid in expected + top_retrieved:
    row = store._conn.execute('SELECT content FROM facts WHERE fact_id = ?', (fid,)).fetchone()
    prox = retriever._phrase_proximity(query, row['content'], proximity_terms)
    print(f'Fact #{fid}: proximity={prox:.4f}, bonus={prox*0.002:.4f}')

# Check vec candidate quality
vec = retriever._vec_candidates(query, None, 30)
for i, v in enumerate(vec[:10]):
    print(f'[{i+1}] dist={v["vec_distance"]:.4f} #{v["fact_id"]}')

# Verify stored vectors match fresh encoding
from embedding import encode
import numpy as np
fresh = encode(content)
stored = np.frombuffer(vrow['embedding'], dtype=np.float32)
print(f'Cosine sim: {np.dot(fresh, stored):.4f}')
```

## Final Parameters (v0.4.10)

- Proximity multiplier: 0.002
- Word boundary checks: enabled
- IDF boost cap: 1.5
- Query prefix: "query:" for queries, "passage:" for facts

## Benchmark Results

| Benchmark | P@3 | R@3 | NDCG@3 |
|-----------|-----|-----|--------|
| LongMemEval (500q) | 0.333 | 0.599 | 0.570 |
| Chinese CMRC (50q) | 0.327 | 0.980 | 0.965 |
