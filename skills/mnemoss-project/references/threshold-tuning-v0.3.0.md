# Threshold Tuning — v0.3.0 (2026-05-08)

## Structural Linking Thresholds

### Changes

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `_HRR_LINK_THRESHOLD` | 0.65 | 0.55 | Production corpus HRR similarities range 0.50–0.54 (normalized) |
| `_EMB_LINK_THRESHOLD` | 0.75 | 0.70 | Embedding similarities slightly below 0.75 in production corpus |
| `_ENTITY_LINK_THRESHOLD` | 0.25 | 0.25 | No change |

### Impact

- **Benchmark (LME DB):** Identical results. The LME DB was prepared with old thresholds and already has 31,934 links baked in.
- **Production corpus:** 0 existing links (structural linking added after facts were inserted). New facts added going forward will use the lower thresholds.
- **Cross-level compression:** Lower thresholds should enable clustering on the production corpus by creating more links for BFS clustering.

### Caveats

1. **Threshold changes only affect NEW facts.** Existing facts keep their original links.
2. **To re-link existing corpus:** Call `_create_links_for_fact(fact_id)` for each fact in the DB.
3. **Lower thresholds increase link count**, which may affect BFS clustering granularity. Monitor cluster sizes after re-linking.

## Phrase Proximity Tuning

### Changes

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| Decay scale | 50.0 | 30.0 | Steeper decay — closer terms matter more |
| Weight | 0.15 | 0.20 | Stronger influence in final score |
| Adjacent boost | 1.5x | 2.0x | Stronger boost for adjacent terms |

### Impact

- **Benchmark (LME DB):** Identical results. The LME DB was prepared with old thresholds and already has 31,934 links baked in.
- **Production corpus:** New facts added going forward will use the tuned parameters.
- **Query types:** More impact expected on `single-session-assistant` queries where phrase structure is clear.

### Benchmark Evidence

v0.2.0 and v0.3.0 show identical P@3 (0.264) and all other metrics. This is expected:
- LME DB was prepared with old parameters → existing links are unchanged
- Proximity scoring only affects post-FTS5 reranking → subtle effect on top-10 candidates
- Benchmark impact is more likely to show on larger datasets (1000+ facts) where candidate ordering is more sensitive

## Recommendations

1. **Re-link production corpus** with new thresholds to enable cross-level compression
2. **Monitor** cluster sizes and consolidation results after re-linking
3. **Re-benchmark** after adding new facts to see if tuned parameters show impact
4. **Consider** further scale reduction (e.g., 20 chars) if proximity still too weak

## Linked Files

- `references/phrase-proximity-scoring-2026.md` — Full proximity algorithm details
- `references/benchmark-findings-2026-05-08.md` — Benchmark pitfall documentation