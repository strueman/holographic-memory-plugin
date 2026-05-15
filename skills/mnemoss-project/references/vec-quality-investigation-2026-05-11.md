# Vec Quality Investigation — 2026-05-11

## Context

After migrating from bge-m3 (1024d) to e5-small (384d), LongMemEval P@3 dropped from 0.249 to 0.069. Investigation to determine root cause.

## Hypotheses Tested

| Hypothesis | Result | Notes |
|-----------|--------|-------|
| 512-token truncation | ❌ Not an issue | Max fact is 326 tokens, zero truncation |
| Chinese/English misrouting | ❌ Not an issue | `_has_chinese()` regex works correctly |
| Proximity scoring bug | ✅ MAIN CULPRIT | Substring matching + 0.20 multiplier dominated RRF |
| IDF boost too aggressive | ⚠️ Partial | Capped at 1.5, no measurable P@3 change |
| Query prefix wrong | ✅ For Chinese | "query:" prefix improved Chinese P@3 0.320→0.327 |
| Vec discrimination | ⚠️ Fundamental | 384d embeddings produce tight clusters (std=0.016) |

## Key Findings

### Proximity Bug (v0.4.7 fix)

- Proximity multiplier 0.20 was 10-60x larger than RRF scores (0.003-0.019)
- Substring matching: "car" matched inside "Carols" → proximity=0.79 for irrelevant fact
- Fact #3203 about "Lessons and Carols service" scored 0.161 (proximity 0.158 + RRF 0.003)
- Expected fact #15 scored only 0.015 (RRF 0.015 + proximity 0.000)
- Fix: word boundary checks + multiplier 0.20→0.02
- Result: LongMemEval P@3 recovered from 0.069 to 0.113 (+64%)

### IDF Boost (v0.4.8 fix)

- In 10K-fact corpus, even common terms have high IDF ("car" IDF=3.72, "service" IDF=3.30)
- IDF boost formula `1 + log(1 + avg_idf)` produced 2.05-3.12 across ALL LongMemEval queries
- Effective FTS weight: 0.5 * 2.66 = 1.33, higher than vec weight 0.75
- Fix: cap IDF boost at 1.5 so FTS and vec stay at parity (0.5 * 1.5 = 0.75)
- Result: No measurable P@3 change (safeguard, not cure)

### Query Prefix (v0.4.9 fix)

- e5 models trained with "query: " prefix for queries, "passage: " for documents
- Using "passage: " for both embeds queries in wrong subspace
- `encode()` function updated to accept `prefix` parameter (default "passage")
- `_vec_candidates()` now calls `encode(query, prefix="query")`
- Result: Chinese P@3 improved 0.320→0.327, LongMemEval unaffected

### Vec Discrimination (fundamental limitation)

- Cosine similarities for top vec candidates: 0.85-0.86 (L2 distances 0.53-0.55)
- Standard deviation across 100 results: 0.016
- Gap between closest and 5th result: only 0.017 L2 distance
- This is a 384-dim embedding limitation on a 10K-fact corpus
- Token length NOT the issue (max 326 tokens, well under 512 limit)

## Remaining Gap

LongMemEval P@3: 0.109 (v0.4.9) vs 0.249 (bge-m3 baseline) = 2.3x gap

Attributed to:
1. 384-dim vec embeddings can't discriminate well across 10K facts
2. Fresh DB has access_count=0 (no accumulated signal)
3. FTS5 OR semantics too broad (719/10960 matches for one query)

## Diagnostic Commands

```python
# Check token lengths
lengths = [len(retriever._tokenize(row['content'])) for row in store._conn.execute('SELECT content FROM facts')]
print(f'Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}')

# Check vec distance spread
vec_cands = retriever._vec_candidates(query, None, 100)
dists = [vc['vec_distance'] for vc in vec_cands]
print(f'Std: {np.std(dists):.4f}, Gap: {max(dists[:5])-min(dists[:5]):.4f}')

# Check proximity scores
for fid in [3203, 1, 15]:
    row = store._conn.execute('SELECT content FROM facts WHERE fact_id=?', (fid,)).fetchone()
    prox = retriever._phrase_proximity(query, row['content'], proximity_terms)
    print(f'Fact #{fid}: proximity={prox:.4f}, bonus={prox*0.02:.4f}')

# Check IDF boost distribution
boosts = [retriever._query_idf(inst['question']) for inst in lme_instances]
print(f'Min: {min(boosts):.4f}, Max: {max(boosts):.4f}, Mean: {sum(boosts)/len(boosts):.4f}')
```
