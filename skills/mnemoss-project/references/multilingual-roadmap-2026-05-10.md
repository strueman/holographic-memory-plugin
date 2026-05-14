# Multilingual Support Roadmap

> **Status:** Research phase — not implemented. Documented for future iteration.
> **Date:** 2026-05-10
> **Current P@3:** 0.319 (weighted RRF, 5 signals, 1694 facts, 181 queries)
> **Target P@3:** 0.45+ (with multilingual expansion)

## Background

The holographic memory plugin currently supports English-only retrieval. A roadmap has been documented for future multilingual support using a three-layer approach that builds on existing infrastructure.

## Three-Layer Approach

### Layer 1: Query Expansion (Cygnet)
- **What:** Expand queries with synonyms before FTS5 search
- **How:** Cygnet (multi-language WordNet collection) + custom domain dict
- **Why:** Recovers relevant facts when user queries in different languages
- **Cost:** ~10ms, no new dependencies (WordNet is standard)
- **Language support:** English, German, French, Spanish, etc. (Cygnet coverage)

### Layer 2: Search Routing (PR #1)
- **What:** Language-aware routing based on query content
- **How:** Detect language → route to appropriate search method
  - English → FTS5 (fast, indexed)
  - Chinese → LIKE + jieba (FTS5 unicode61 breaks Chinese)
  - Mixed → combined approach
- **Why:** FTS5's unicode61 tokenizer tokenizes Chinese into single characters
- **Status:** PR #1 exists but is un-mergeable (built against old version)
- **Cost:** ~5ms for detection, negligible for routing

### Layer 3: Cross-Encoder Reranking
- **What:** Re-rank top candidates with a cross-encoder model
- **How:** `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-v2-m3`
- **Why:** Language-agnostic relevance scoring, improves ranking quality
- **Cost:** ~100ms for 50 candidates
- **Language support:** Depends on model (ms-marco-MiniLM is English-only, BGE v2-m3 is multilingual)

## Combined Pipeline

```
Query → Layer 1 (expand with Cygnet) → Layer 2 (route to FTS5/LIKE) → 
Top 50 candidates → Layer 3 (cross-encoder rerank top 20) → Final results
```

**Expected total gain:** P@3 0.319 → 0.45+ (target)
**Expected total overhead:** ~150-250ms (vs current ~3s)

## Implementation Notes

### Cygnet
- Collection of WordNets from many languages
- Same API as NLTK WordNet
- Larger download than English-only WordNet
- Sparser coverage per language (but sufficient for expansion)

### PR #1 (Bilingual Search)
- Language detection via regex: `re.search(r'[\u4e00-\u9fff]', text)`
- Chinese entity extraction via regex for CJK character sequences
- 32 test cases covering English-only, Chinese-only, mixed
- Chinese documentation included
- **Un-mergeable:** Built against old version, needs rebasing

### Cross-Encoder Models
- **ms-marco-MiniLM-L-6-v2:** 120MB, English-only, ~100ms/100 pairs
- **BAAI/bge-reranker-v2-m3:** 250MB, multilingual, ~200ms/100 pairs
- Both use `cross_encoder` library (same as bi-encoder)

## Open Questions

1. **Cygnet coverage:** How many of our query terms are covered by Cygnet vs. custom dict?
2. **Expansion strategy:** Replace vs. append? How many synonyms per token?
3. **Rerank depth:** 20 vs. 50 candidates? Diminishing returns?
4. **Cross-encoder vs. bi-encoder:** Is the +5-10% worth the extra cost?
5. **Query expansion timing:** Expand before FTS5, or expand query tokens and run multiple FTS5 queries?

## References

- NLTK WordNet: https://www.nltk.org/howto/wordnet.html
- MS MARCO MiniLM cross-encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
- BGE Reranker: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Cygnet (multi-language WordNet): https://github.com/... (to be added)
- vstash hybrid retrieval: https://arxiv.org/abs/2604.15484
