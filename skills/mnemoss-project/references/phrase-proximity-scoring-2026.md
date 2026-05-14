# Phrase Proximity Scoring

Post-FTS5 reranking bonus: rewards facts where query terms appear closer together in the content.

## Algorithm

1. **Extract terms:** meaningful query tokens (non-stopwords, >= 3 chars)
2. **Find positions:** for each term, find all character positions in content (case-insensitive substring search)
3. **Calculate span:** `max(all_positions) - min(all_positions)`
4. **Compute score:** `e^(-span / 30)` — exponential decay, scale = 30 chars (v0.3.0)
5. **Adjacent boost:** if `span < avg_term_length`, multiply by 2.0 (capped at 1.0)
6. **Single term:** returns 0.5 (neutral — proximity undefined)
7. **Combine:** `final_score = rrf_score * trust + proximity_score * 0.20`

## Parameters (v0.3.0)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Decay scale | 30 chars | Steeper decay (was 50) — closer terms matter more |
| Weight | 0.20 | Stronger influence (was 0.15) |
| Adjacent boost | 2.0x | Stronger boost when span < avg term length (was 1.5x) |
| Single term | 0.5 | Neutral fallback |

## Implementation

Location: `retrieval.py` — `FactRetriever._phrase_proximity(query, content, terms)`

Called in `search()` pipeline after RRF fusion, before temporal decay.

## Pitfalls

- **Substring matching:** uses `str.find()` — "cat" matches "category", not just word "cat"
- **Overlap:** for terms like "cat" and "category", the shorter term's positions dominate
- **No stemming:** "running" and "run" are different terms
- **Benchmark impact:** small but meaningful on queries with clear phrase structure
- **Single term:** proximity is undefined for single-term queries — returns neutral 0.5
- **Tuning note:** v0.3.0 increased weight and reduced scale. Benchmark showed identical results on pre-linked LME DB — impact visible on new facts only.

## Example

Query: "Samsung Galaxy S22"
- Terms: ["samsung", "galaxy", "s22"]
- Fact: "Samsung Galaxy S22 is a great phone"
- Positions: samsung=0, galaxy=8, s22=16
- Span: 16 - 0 = 16
- Score: e^(-16/30) = 0.59
- Avg term length: 5.7, span (16) > avg → no boost
- Final proximity: 0.59

Fact: "I bought a samsung phone and a galaxy tablet"
- Positions: samsung=10, galaxy=30, s22=none
- Span: 30 - 10 = 20 (only 2 terms found)
- Score: e^(-20/30) = 0.51
- Final proximity: 0.51

## References

- `retrieval.py` — `_phrase_proximity()` method
- `retrieval.py` — `search()` pipeline integration
- `references/threshold-tuning-v0.3.0.md` — tuning rationale