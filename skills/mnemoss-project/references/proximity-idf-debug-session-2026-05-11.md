# Proximity + IDF Debug Session (2026-05-11)

## Context

After migrating from bge-m3 (1024d) to e5-small (384d), LongMemEval benchmark showed massive regression: P@3 0.249 -> 0.069 (-72%).

## Investigation Steps

### Step 1: Verify ground truth
- Checked `lme_instances.expected_fact_ids` map to correct fact content
- Result: Ground truth correct (fact #1 about car detailing, fact #15 about detailer)

### Step 2: Check candidate pool coverage
- Expected fact #1: in vec=True (rank 2), in FTS5=False
- Expected fact #15: in vec=True (rank 5), in FTS5=True (rank 2)
- Result: Expected facts ARE in candidate pools but not ranking high enough

### Step 3: Score expected facts manually
- Fact #1: RRF=0.0089, trust=0.8, final=0.0075
- Fact #15: RRF=0.0142, trust=0.8, final=0.0122
- Top retrieved fact #3203: score=0.161 (10x higher than expected!)
- Result: Something is massively boosting wrong facts

### Step 4: Check signal dominance
- Proximity bonus for fact #3203: 0.79 * 0.20 = 0.158
- RRF score for fact #3203: 0.003
- Result: Proximity bonus 50x larger than RRF — proximity is dominating

### Step 5: Check substring false positives
- Fact #3203 content: "Lessons and Carols service"
- Query terms: ["car", "issue", "service"]
- "car" matched inside "Carols" (substring match)
- "service" matched as standalone word
- Result: Two matches, span=7 chars, proximity=exp(-7/30)=0.79 — false positive!

## Root Causes Found

### Cause 1: Proximity multiplier too high (0.20)
- Proximity bonus range: 0.0-0.20
- RRF score range: 0.003-0.019
- Proximity 10-60x larger than RRF → proximity dominates ranking
- Fix: Reduce multiplier to 0.02 (v0.4.7)

### Cause 2: Substring matching in proximity
- `str.find()` matches substrings ("car" in "Carols")
- Fix: Add word boundary checks (v0.4.7)
- Check: `content[idx-1].isalnum()` and `content[idx+len(term)].isalnum()`

### Cause 3: IDF boost too aggressive for large corpora
- LongMemEval avg IDF boost: 2.66 (min 2.05, max 3.2)
- Effective FTS weight: 0.5 * 2.66 = 1.33 > vec weight 0.75
- Fix: Cap IDF boost at 1.5 (v0.4.8)

## Results

| Fix | LongMemEval P@3 | Chinese P@3 |
|-----|-----------------|-------------|
| Before (broken) | 0.069 | 0.320 |
| After proximity fix (v0.4.7) | 0.113 (+64%) | 0.320 |
| After IDF cap (v0.4.8) | 0.111 | 0.320 |
| bge-m3 baseline | 0.249 | 0.300 |

## Remaining Gap

P@3 0.111 vs baseline 0.249. Attributed to:
1. e5-small vec distances too close to discriminate (0.530 vs 0.532 vs 0.547)
2. Fresh DB with access_count=0 (no accumulated signal)
3. FTS5 OR semantics too broad (719/10960 matches for "car OR service OR issue")

## Key Lessons

- Always check signal dominance when benchmark regresses
- Proximity substring matching is a silent killer in large corpora
- IDF boost scales with corpus size — cap it for large DBs
- Chinese/English routing was NOT the issue (correctly detects CJK characters)
