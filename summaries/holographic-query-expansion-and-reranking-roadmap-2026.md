# Holographic Query Expansion & Reranking Roadmap

> **Status:** Research phase — not implemented. Documented for future iteration.
> **Date:** 2026-05-10
> **Current P@3:** 0.319 (weighted RRF, 5 signals, 1694 facts, 181 queries)
> **Target P@3:** 0.40-0.45 ("good" zone)

## Background

Current retrieval pipeline uses weighted RRF fusion of 5 signals (FTS5=0.5, Vec=0.75, Jaccard=0.75, HRR=0.5, Access=0.5) with BM25-IDF boosting. P@3=0.319 is "acceptable" for a lightweight system (~3s latency, zero cost, local-only), but there's room to push toward 0.45+ with programmatic query expansion and lightweight reranking.

## Phase 1: Query Expansion (Thesaurus-based)

### Approach
Expand queries with synonyms before FTS5 search. Pure lookup, no LLM, no reasoning.

### Options
1. **NLTK WordNet** — Free, local, ~15MB download, English only
   - `from nltk.corpus import wordnet`
   - `wordnet.synsets("fast")` returns synonym sets
   - Pros: Zero cost, instant, no new deps
   - Cons: English only, won't have domain terms ("RRF", "HRR", "Minilm")

2. **Cygnet** — Multi-language WordNet collection, future-proof for i18n
   - Collection of WordNets from many languages (English, German, French, Spanish, etc.)
   - Pros: Multi-language support from day one, same API as WordNet
   - Cons: Larger download, may have sparser coverage per language
   - **Why it matters:** If we ever support non-English queries, Cygnet covers it without new infrastructure

3. **Custom synonym dict** — Hand-curated for domain
   - Pros: Domain-accurate, no external deps
   - Cons: Manual work, limited coverage

4. **Hybrid** — Cygnet/WordNet for common words + custom dict for domain terms
   - Best of both worlds, with multi-language readiness

### Implementation
```python
# Expand query by replacing each token with synonyms
def expand_query(query):
    tokens = tokenize(query)
    expanded = []
    for token in tokens:
        synonyms = get_synonyms(token)  # WordNet or custom dict
        expanded.extend(synonyms)
    return " ".join(expanded)

# Run FTS5 on expanded query, take top results
results = search(expand_query(query))
```

### Expected gain
- +5-10% P@3 on its own
- Negligible overhead (~10ms)
- No new dependencies (WordNet) or minimal (custom dict)

### Risks
- Over-expansion could introduce noise
- WordNet may not cover technical terms
- Requires careful filtering (stopwords, short tokens)

## Phase 2: Sentence Transformer Reranking

### Approach
Use existing MiniLM model (already in dependency chain) to rerank top 50 candidates by cosine similarity.

### Why this works
- MiniLM encodes both query and facts into same embedding space
- Cosine similarity is O(n) — instant
- Uses model already downloaded (~66MB)
- No new dependencies

### Implementation
```python
# Encode query and top 50 candidates
query_vec = model.encode(query)
candidate_vecs = model.encode([f["content"] for f in top_50])

# Cosine similarity reranking
scores = cosine_similarity(query_vec, candidate_vecs)
reranked = sort_by(scores, reverse=True)[:10]
```

### Expected gain
- +5-10% P@3
- ~50ms overhead for 50 candidates
- Uses existing model

### Risks
- MiniLM is a bi-encoder, not a cross-encoder
- May not capture fine-grained relevance
- Could conflict with existing Vec signal

## Phase 3: Cross-Encoder Reranking

### Approach
Use a purpose-built cross-encoder model to score (query, passage) pairs.

### Options
| Model | Size | Speed | Expected Gain |
|-------|------|-------|---------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 120MB | ~100ms/100 pairs | +10-15% |
| `BAAI/bge-reranker-base` | 250MB | ~200ms/100 pairs | +15-20% |
| `BAAI/bge-reranker-v2-m3` | 250MB | ~200ms/100 pairs | +15-25% |

### Implementation
```python
from cross_encoder import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score (query, passage) pairs
pairs = [(query, f["content"]) for f in top_50]
scores = model.predict(pairs)
reranked = sort_by(scores, reverse=True)[:10]
```

### Expected gain
- +10-15% P@3 (MiniLM cross-encoder)
- ~100ms overhead for 50 candidates
- Purpose-built for reranking

### Risks
- New dependency (cross-encoder)
- Model download (~120MB)
- Cross-encoder is slower than bi-encoder

## Phase 4: Combined Approach

### Pipeline
1. **Query expansion** (Phase 1) — Expand query with synonyms
2. **FTS5 + Vec retrieval** — Get top 50 candidates
3. **Sentence transformer rerank** (Phase 2) — Quick cosine similarity
4. **Cross-encoder rerank** (Phase 3) — Fine-grained scoring on top 20

### Expected total gain
- Phase 1 alone: 0.319 → ~0.35
- Phase 1 + 2: 0.319 → ~0.40
- Phase 1 + 2 + 3: 0.319 → ~0.45-0.50

### Total overhead
- ~150-250ms per query (vs current ~3s)
- Still under 1 second total

## Decision Criteria

| Factor | Weight | Notes |
|--------|--------|-------|
| P@3 improvement | High | Primary metric |
| Latency increase | Medium | Must stay under 1s total |
| Dependency cost | Medium | Prefer existing deps |
| Implementation complexity | Low | Keep it simple |
| Domain coverage | Medium | Must work for technical terms |

## Open Questions

1. **WordNet coverage:** How many of our query terms are covered by WordNet vs. custom dict?
2. **Expansion strategy:** Replace vs. append? How many synonyms per token?
3. **Rerank depth:** 20 vs. 50 candidates? Diminishing returns?
4. **Cross-encoder vs. bi-encoder:** Is the +5-10% worth the extra cost?
5. **Query expansion timing:** Expand before FTS5, or expand the query tokens and run multiple FTS5 queries?

## Next Steps

1. Measure WordNet coverage on current query set
2. Implement Phase 1 (query expansion) as a standalone experiment
3. Benchmark P@3 improvement
4. If promising, add Phase 2 (sentence transformer rerank)
5. If still not enough, evaluate Phase 3 (cross-encoder)

## Existing Bilingual PR

PR #1 already exists: "feat: bilingual search optimization with smart routing"
- Language-aware routing: English → FTS5, Chinese → LIKE+jieba, Mixed → combined
- Fixes FTS5 unicode61 tokenizer issue with Chinese (single-char tokenization)
- 32 test cases, Chinese documentation included
- **Status:** un-mergeable (built against old version), but the core idea is solid
- **Note:** This PR is about *search routing*, not query expansion. The two approaches are complementary.

## References

- NLTK WordNet: https://www.nltk.org/howto/wordnet.html
- MS MARCO MiniLM cross-encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
- BGE Reranker: https://huggingface.co/BAAI/bge-reranker-v2-m3
- vstash hybrid retrieval: https://arxiv.org/abs/2604.15484
