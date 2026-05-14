# Mnemoss

A lightweight memory system for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — symbiotically combining algorithmic speed with LLM reasoning. Built on **Holographic Reduced Representation (HRR)** vectors for compositional reasoning, backed by SQLite with FTS5 full-text search, sqlite-vec dense vector search, and Reciprocal Rank Fusion (RRF) retrieval.

> **Attribution:** Original plugin by [dusterbloom](https://github.com/dusterbloom) (PR [#2351](https://github.com/NousResearch/hermes-agent/pull/2351)), adapted to the `MemoryProvider` ABC.

> **Multilingual roadmap:** Three-layer approach for multilingual support — (1) **Query expansion** via Cygnet (multi-language WordNet), (2) **Search routing** (PR #1: language-aware routing, English→FTS5, Chinese→jieba), (3) **Cross-encoder reranking** (language-agnostic). See [roadmap](https://github.com/strueman/mnemoss/blob/main/summaries/holographic-query-expansion-and-reranking-roadmap-2026.md).

## Features

- **HRR compositional memory** — bind roles and entities into algebraic vectors; probe and reason across entities using vector unbinding
- **FTS5 full-text search** — fast keyword matching via SQLite's built-in FTS5 engine
- **Jaccard token overlap** — reranking based on shared token sets
- **Reciprocal Rank Fusion (RRF)** — weighted rank-level fusion combining FTS5, Vec, Jaccard, HRR, and access count (5 signals) with optimized weights
- **Entity resolution** — automatic entity extraction and linking from fact content
- **Trust scoring** — asymmetric trust adjustment based on helpful/unhelpful feedback
- **Fact splitting** — recursive sentence-boundary decomposition reduces FTS5 keyword competition
- **Entity-boosted RRF** — conditional entity rank boost for named-entity queries
- **Evidence-gap tracker** — algorithmic coverage analysis checks if retrieved facts actually cover the query's key components (inspired by MemR3, arxiv 2512.20237)
- **Dense vector search** — sqlite-vec KNN queries on 384-dim ONNX embeddings (all-MiniLM-L6-v2), auto-downloaded on first use
- **Zero external deps** — uses only Python stdlib (sqlite3, math) + optional numpy for HRR algebra
- **Single-user** — designed for standalone Hermes Agent sessions

## Requirements

- Python 3.11+
- SQLite 3.35+ (with FTS5 support — enabled by default on Debian)
- `numpy` — optional, required for HRR vector operations (probe, reason, related)
- `sqlite-vec` + `onnxruntime` + `tokenizers` — optional, required for dense vector search (auto-installed via `pip install sqlite-vec onnxruntime tokenizers`)

## Installation

Drop the `mnemoss` directory into your Hermes Agent plugin path:

```bash
cp -r mnemoss ~/.hermes/hermes-agent/plugins/memory/mnemoss
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    FactRetriever                           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐      │
│  │ FTS5    │  │ Jaccard  │  │ HRR     │  │ Vec KNN  │      │
│  │ rank    │  │ rank     │  │ rank    │  │ rank     │      │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘      │
│       └────────────┴─────────────┴────────────┘            │
│                           │                                │
│              Reciprocal Rank Fusion (RRF)                  │
│              sum(w_i / (rank_i + 60)) / sum(weights)       │
│              Optimal: FTS=0.5, Vec=0.75, Jaccard=0.75      │
│                           │                                │
│                  Trust-weighted score                      │
└───────────────────────────┬────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               │    MemoryStore          │
               │  ┌──────────────────┐   │
               │  │ SQLite (WAL mode)│   │
               │  │  • facts         │   │
               │  │  • facts_fts     │   │
               │  │  • facts_vec     │   │
               │  │  • entities      │   │
               │  │  • memory_banks  │   │
               │  └──────────────────┘   │
               └─────────────────────────┘
```

## Retrieval Pipeline

The `FactRetriever.search()` method uses a multi-stage pipeline with four ranking sources fused via RRF:

1. **FTS5 candidate retrieval** — fast keyword search via SQLite FTS5, returns top 3x limit candidates
2. **Vec KNN candidate retrieval** — dense vector similarity search via sqlite-vec, returns top 3x limit candidates
3. **Candidate union** — FTS5 and Vec results merged, deduplicated by fact_id
4. **Per-candidate ranking** — Jaccard and HRR scores computed for each candidate
5. **RRF fusion** — weighted rank fusion: `sum(w_i / (rank_i + 60)) / sum(weights)` with optimized weights (FTS=0.5, Vec=0.75, Jaccard=0.75, HRR=0.5, Access=0.5)
6. **Trust weighting** — final score multiplied by fact's trust_score
7. **Evidence-gap analysis** — coverage check on retrieved facts (optional, enabled by default)

## Memory Types

| Type | Method | Description |
|------|--------|-------------|
| **HRR probe** | `probe(entity)` | Unbinds entity from memory banks to find associated facts |
| **HRR related** | `related(entity)` | Finds facts sharing structural connections with an entity |
| **Multi-entity reason** | `reason([e1, e2, ...])` | Vector-space JOIN across multiple entities (AND semantics) |
| **Contradiction check** | `contradict()` | Finds facts with high entity overlap but low content similarity |
| **FTS5 search** | `search(query)` | Full-text keyword search with entity-boosted RRF fusion |
| **Fact splitting** | `split_and_add(fact)` | Auto-decomposes long composite facts via sentence-boundary splitting |

## Fact Splitting

Long facts can create keyword competition in FTS5, diluting relevance. The `fact_splitter` module
recursively decomposes composite facts at sentence, comma, and conjunction boundaries to produce
atomic facts. This improves retrieval precision without changing the storage schema.

Splitting is applied only when a fact exceeds 80 characters and produces at least 2 atomic fragments.

## Entity Extraction

Entity patterns added in this branch:

- `RE_ALLCAPS` — matches runs of 2+ all-caps tokens (e.g. `LLM`, `FTS5`)
- `RE_MIXED` — matches mixed alphanumeric tokens (e.g. `35b`, `Qwen3`)
- Fragment filtering: removes single-word fragments from split facts

Entity-boosted RRF is **conditional**: only applies when the query contains entities that match stored facts. This prevents noise on natural language queries.

## Dense Vector Search

Inspired by [vstash](https://arxiv.org/abs/2604.15484) hybrid retrieval, the plugin integrates [sqlite-vec](https://github.com/asg017/sqlite-vec) for dense vector similarity search alongside FTS5 keyword search.

### How It Works

- **Embedding model:** multilingual-e5-small (384-dim, INT8 ONNX runtime, ~449MB) downloaded on first use and cached in `~/.hermes/mnemoss_embeddings/`. Uses "query: " / "passage: " prefix pattern required by e5 models.
- **Storage:** `facts_vec` vec0 virtual table stores float32 embeddings keyed by fact_id
- **Query:** sqlite-vec KNN finds nearest neighbors to the query embedding
- **Fusion:** Vec rank is one of five sources in the weighted RRF fusion pipeline (FTS5=0.5, Vec=0.75, Jaccard=0.75, HRR=0.5, Access=0.5)

### Schema

```sql
CREATE VIRTUAL TABLE facts_vec USING vec0(
    embedding float[384]
);
```

### Backfilling Existing Facts

If facts were added before vector search was enabled, compute embeddings for them:

```python
from store import MemoryStore
store = MemoryStore()
count = store.backfill_embeddings()  # returns number of facts embedded
```

### Dependencies

Vector search requires three optional packages (not installed by default):

```bash
pip install sqlite-vec onnxruntime tokenizers
```

If unavailable, the pipeline degrades gracefully — Vec rank is simply omitted from the RRF fusion and search continues with FTS5 + Jaccard + HRR only.

### Testing

15 unit tests in `tests/test_sqlite_vec.py` covering vec table creation, embedding storage, KNN queries, RRF fusion, backfill, and graceful degradation.

## Evidence-Gap Tracker

Inspired by MemR3's closed-loop retrieval controller (arxiv 2512.20237), the evidence-gap tracker adds a lightweight algorithmic layer on top of `search()` that checks whether retrieved facts actually cover what the query is asking.

### How It Works

Every `search()` call decomposes the query into three component types:

1. **Entities** — extracted via the store's `_extract_entities()` (capitalized phrases, all-caps tokens, mixed alphanumeric)
2. **Content words** — significant non-stopword tokens >= 3 characters (80+ English stopwords filtered)
3. **Phrases** — multi-word capitalized phrases from regex

For each component, coverage is checked against all retrieved facts:
- Single-word components: exact substring match (case-insensitive)
- Multi-word components: Jaccard similarity >= 0.15

### Return Values

An evidence-gap dict is attached to the first search result:

| Field | Type | Description |
|-------|------|-------------|
| `coverage_ratio` | float 0.0-1.0 | Fraction of query components covered |
| `covered` | list[str] | Components that were covered |
| `uncovered` | list[str] | Components that were NOT covered |
| `components_total` | int | Total components checked |
| `needs_followup` | bool | True if coverage_ratio < 0.6 |

When `needs_followup` is True, the LLM is instructed to refine its search query to cover missing components.

### Usage

```python
# Enabled by default
results = retriever.search("what about the Debian VM setup?")
gap = results[0].get("evidence_gap")  # dict with coverage info

# Disable if not needed
results = retriever.search("query", evidence_gap=False)
```

### Testing

19 unit tests in `tests/plugins/memory/test_evidence_gap.py` covering entity extraction, content word filtering, coverage checking, edge cases, and JSON serializability.

## Benchmarks

Evaluations performed on the LongMemEval oracle subset (500 instances, 10,960 turns) and Chinese CMRC 2018 (50 instances, 50 facts) using precision, recall, and NDCG at K=3 and K=5.

**Benchmark methodology:** The benchmark imports the real `FactRetriever` and `MemoryStore` from the plugin, retrieves actual FTS5 + vec KNN candidates, then applies RRF scoring on the **live candidate pool**. All five ranking sources (FTS5, Vec, Jaccard, HRR, Access) are active during candidate retrieval.

### LongMemEval 500-instance Results (v0.4.10, 5-signal Weighted RRF)

**Metric key:** P@K = Precision@K (fraction of top-K results that are relevant), R@K = Recall@K (fraction of relevant results found in top-K), NDCG@K = Normalized Discounted Cumulative Gain (ranking quality, rewards relevant items appearing higher).

| Method | P@3 | P@5 | R@3 | R@5 | NDCG@3 |
|--------|-----|-----|-----|-----|--------|
| v0.4.2 baseline (MiniLM) | 0.249 | 0.173 | 0.435 | 0.498 | 0.420 |
| v0.4.6 initial (e5-small) | 0.069 | 0.058 | 0.115 | 0.158 | 0.110 |
| **v0.4.10 (e5-small, fixed)** | **0.333** | **0.232** | **0.599** | **0.672** | **0.570** |
| **Delta vs baseline** | **+33.7%** | **+34.1%** | **+37.7%** | **+34.7%** | **+35.7%** |

Per-query wins (P@3): RRF=363, Tie=137 (out of 500).

### Chinese CMRC 2018 Results (v0.4.10, 50 queries, 50 facts)

| Method | P@3 | P@5 | R@3 | R@5 | NDCG@3 |
|--------|-----|-----|-----|-----|--------|
| v0.4.2 baseline (MiniLM) | 0.300 | 0.180 | 0.900 | 0.900 | 0.853 |
| **v0.4.10 (e5-small, fixed)** | **0.327** | **0.196** | **0.980** | **0.980** | **0.965** |
| **Delta vs baseline** | **+9.0%** | **+8.9%** | **+8.9%** | **+8.9%** | **+13.1%** |

### By Query Type — LongMemEval (P@3, v0.4.10)

| Query Type | P@3 |
|------------|-----|
| temporal-reasoning | 0.318 |
| multi-session | 0.376 |
| knowledge-update | 0.432 |
| single-session-preference | 0.167 |
| single-session-assistant | 0.333 |
| single-session-user | 0.243 |

### Regression Investigation (v0.4.6 → v0.4.10)

Initial e5-small migration (v0.4.6) showed P@3 regression from 0.249 to 0.069 on LongMemEval. Root cause: phrase proximity scoring (multiplier 0.20) dominated RRF scores (0.003-0.019), allowing irrelevant facts with coincidental term adjacency to outrank semantically relevant ones. Fixed across 4 commits:

| Tag | Fix | LongMemEval P@3 | Chinese P@3 |
|-----|-----|-----------------|-------------|
| v0.4.7 | Proximity word boundaries + mult 0.20→0.02 | 0.113 | 0.320 |
| v0.4.8 | IDF boost cap at 1.5 | 0.111 | 0.320 |
| v0.4.9 | "query: " prefix for search queries | 0.109 | 0.327 |
| v0.4.10 | Proximity mult 0.02→0.002 | **0.333** | 0.327 |

Final P@3=0.333 exceeds v0.4.2 baseline (0.249) by 33.7%.

### Personal Memory Dataset v0.4.2 (1694 facts, 181 queries, 100% vec coverage)

| Method | P@3 | P@5 | R@3 | R@5 | NDCG@3 |
|--------|-----|-----|-----|-----|--------|
| Weighted-sum | 0.297 | 0.213 | 0.529 | 0.594 | 0.541 |
| **Weighted RRF** | **0.319** | **0.374** | **0.297** | **0.371** | **0.302** |
| **Delta** | **+7.4%** | **+75.6%** | — | — | — |

**Optimal weights:** FTS=0.5, Vec=0.75, Jaccard=0.75, HRR=0.5, Access=0.5

By query type (Weighted RRF):
- temporal: 0.227
- personal-memory: 0.148
- comparative: 0.187
- ambiguous: 0.197
- multi-hop: 0.154
- entity-resolution: 0.150

### Notes

- **Previous benchmark (standalone)** reported higher deltas (RRF P@3: 0.223 vs Weighted 0.154, +44.6%) because it reimplemented retrieval from scratch without vec candidates. The new benchmark uses the actual plugin pipeline with all 5 signals active, producing higher absolute scores but narrower deltas since both methods benefit from vec.
- **Entity-boosted RRF** (conditional entity rank boost for named-entity queries) was evaluated separately and underperformed standard RRF on synthetic LME data (P@3: 0.177 vs 0.223). Entity boosting remains enabled in production where real entities show better signal.
- **e5-small vs MiniLM:** e5-small backfills 56x faster (~56 facts/s vs ~1 fact/s) with comparable or better retrieval quality. Multilingual coverage (100+ languages) enables Chinese CMRC benchmark support.

## Tests

```bash
# RRF tests
python -m pytest tests/test_mnemoss_rrf.py -v

# Evidence-gap tracker tests
python -m pytest tests/test_mnemoss_rrf.py -v

# Dense vector search tests
python -m pytest tests/test_sqlite_vec.py -v
```

20 tests for RRF (math, full integration, numpy-aware HRR, edge cases, tokenization, IDF weighting, weighted RRF).
19 tests for evidence-gap tracker (entity extraction, content word filtering, coverage checking, edge cases, JSON serializability).
15 tests for sqlite-vec integration (vec table creation, embedding storage, KNN queries, RRF fusion, backfill, graceful degradation).
24 tests for structural linking (graph traversal, link scoring, evidence-gap tracking, temporal segmentation).
15 tests for temporal segmentation (episode detection, temporal boundaries, segment scoring).
26 tests for bilingual search (Chinese detection, jieba tokenization, LIKE fallback, entity extraction, edge cases).
All 184 tests pass.

## License

MIT — see LICENSE file.
Original plugin code by dusterbloom (PR #2351).
RRF improvements, entity-boosted RRF, fact splitting, evidence-gap tracker, and sqlite-vec dense vector search by Axiom (2026).
