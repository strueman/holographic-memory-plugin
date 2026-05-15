# Proximity Scoring Regression Debug (2026-05-11)

## Problem

After migrating from bge-m3 (1024d) to e5-small (384d), LongMemEval P@3 dropped from 0.249 to 0.069 (-72%).

## Diagnosis

### Step 1: Ground truth verification
- Checked `lme_instances.expected_fact_ids` map to correct content
- Expected facts #1 and #15 contained correct car detailing content
- Ground truth was correct

### Step 2: Candidate pool coverage
- Fact #1: NOT in FTS5, in vec at rank 2
- Fact #15: in FTS5 at rank 2, in vec at rank 5
- Expected facts ARE in candidate pools but not ranking high enough

### Step 3: Manual score tracing
```
Fact #1:  FTS5=None, Vec=2, Jac=0.125(r~27), RRF=0.0131, Final=0.0105
Fact #15: FTS5=2,    Vec=5, Jac=0.113(r~27), RRF=0.0186, Final=0.0149
Fact #3203: FTS5=None, Vec=None, Jac=0.088(r~28), RRF=0.0113, Final=0.0056
```

Expected fact #15 had HIGHEST RRF score (0.0186) but was NOT in top 10 results.

### Step 4: Signal dominance check
Live search results showed:
```
[0.160978 #id=3203] Lessons and Carols service...
[0.104925 #id=8143] open track day at VIRginia...
```

Fact #3203 scored 0.161 while expected facts scored ~0.015. Something was boosting #3203 massively.

### Step 5: Proximity investigation
```
Fact #3203: proximity=0.7919, prox_bonus=0.1584
Fact #1:    proximity=0.0101, prox_bonus=0.0020
Fact #15:   proximity=0.0000, prox_bonus=0.0000
```

**Root cause identified:** Proximity bonus (0.158) completely overwhelmed RRF score (0.003-0.019).

### Step 6: Substring false positive
Fact #3203 content: "Lessons and Carols service"
- Query terms: ['car', 'issue', 'service']
- "car" matched inside "Carols" (substring match)
- "service" matched as standalone word
- Span between "car" (in Carols) and "service" was tiny → proximity=0.79

## Fix

1. **Word boundary checks:** Reject substring matches where term is part of a larger word
   - Check `content[idx-1].isalnum()` (left boundary)
   - Check `content[idx+len(term)].isalnum()` (right boundary)

2. **Multiplier reduction:** 0.20 → 0.02 (10x reduction)
   - Proximity now acts as tiebreaker, not dominant signal
   - RRF scores (0.003-0.019) now dominate proximity bonus (0.0-0.02)

## Results

| Metric | Before Fix | After Fix | Change |
|--------|-----------|-----------|--------|
| P@3 | 0.069 | 0.113 | +64% |
| R@3 | 0.115 | 0.191 | +66% |
| NDCG@3 | 0.110 | 0.180 | +64% |

Chinese benchmark unaffected (P@3=0.320 both before and after).

## Lessons

1. **Signal dominance:** When one signal contributes 10x more than others, it dominates regardless of quality
2. **Substring matching:** `str.find()` matches substrings — "car" in "Carols" is a false positive
3. **Multiplier sensitivity:** A 33% increase (0.15→0.20) caused a 72% regression because proximity was already the dominant signal
4. **Debug method:** Trace individual fact scores manually to identify which signal is causing the problem
