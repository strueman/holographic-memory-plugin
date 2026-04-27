# Holographic Memory Plugin

A lightweight, single-user memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) using **Holographic Reduced Representation (HRR)** vectors for compositional reasoning, backed by SQLite with FTS5 full-text search and Reciprocal Rank Fusion (RRF) retrieval.

> **Attribution:** Original plugin by [dusterbloom](https://github.com/dusterbloom) (PR [#2351](https://github.com/NousResearch/hermes-agent/pull/2351)), adapted to the `MemoryProvider` ABC.
>
> **⚠️ This is the standalone plugin repo (strueman/holographic-memory-plugin). Only push to this repo — do NOT push to NousResearch/hermes-agent.**

## Features

- **HRR compositional memory** — bind roles and entities into algebraic vectors; probe and reason across entities using vector unbinding
- **FTS5 full-text search** — fast keyword matching via SQLite's built-in FTS5 engine
- **Jaccard token overlap** — reranking based on shared token sets
- **Reciprocal Rank Fusion (RRF)** — rank-level fusion combining FTS5, Jaccard, and HRR without weight tuning
- **Entity resolution** — automatic entity extraction and linking from fact content
- **Trust scoring** — asymmetric trust adjustment based on helpful/unhelpful feedback
- **Fact splitting** — recursive sentence-boundary decomposition reduces FTS5 keyword competition
- **Entity-boosted RRF** — conditional entity rank boost for named-entity queries
- **Zero external deps** — uses only Python stdlib (sqlite3, math) + optional numpy for HRR algebra
- **Single-user** — designed for standalone Hermes Agent sessions

## Requirements

- Python 3.11+
- SQLite 3.35+ (with FTS5 support — enabled by default on Debian)
- `numpy` — optional, required for HRR vector operations (probe, reason, related)

## Installation

Drop the `holographic` directory into your Hermes Agent plugin path:

```bash
cp -r holographic ~/.hermes/hermes-agent/plugins/memory/holographic
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FactRetriever                             │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
│  │ FTS5    │  │ Jaccard  │  │ HRR     │  │ Entity Match │  │
│  │ rank    │  │ rank     │  │ rank    │  │ boost (cond) │  │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └──────┬───────┘  │
│       └────────────┴─────────────┴───────────────┘          │
│                           │                                  │
│              Reciprocal Rank Fusion (RRF)                    │
│              C / (rank + C)  [C=60]                          │
│                           │                                  │
│                  Trust-weighted score                        │
└───────────────────────────┬────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               │    MemoryStore          │
               │  ┌──────────────────┐   │
               │  │ SQLite (WAL mode)│   │
               │  │  • facts         │   │
               │  │  • facts_fts     │   │
               │  │  • entities      │   │
               │  │  • memory_banks  │   │
               │  └──────────────────┘   │
               └─────────────────────────┘
```

## Retrieval Pipeline

The `FactRetriever.search()` method uses a four-stage pipeline:

1. **FTS5 candidate retrieval** — fast keyword search via SQLite FTS5, returns top 3× limit candidates
2. **Independent ranking** — Jaccard and HRR scores computed within candidate set
3. **Entity resolution** — query entities extracted via regex patterns; conditional RRF boost applied
4. **RRF fusion** — scores combined via `60/(rank + 60)`, entity boost applied post-fusion when query contains entities

## Memory Types

| Type | Method | Description |
|------|--------|-------------|
| **Holographic probe** | `probe(entity)` | Unbinds entity from memory banks to find associated facts |
| **Holographic related** | `related(entity)` | Finds facts sharing structural connections with an entity |
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

## Benchmarks

Evaluations performed on the LongMemEval oracle subset (500 instances, 10,960 turns) using precision, recall, and NDCG at K=3.

### LongMemEval 500-instance Results (Entity-Boosted RRF)

| Method | P@3 | R@3 | NDCG@3 |
|--------|-----|-----|--------|
| Standard Weighted-sum | 0.185 | 0.267 | 0.240 |
| RRF (C=60) | 0.248 | 0.359 | 0.366 |
| **Entity-Boosted RRF** | **0.293** | **0.399** | **0.381** |

Entity-boosted RRF outperforms standard RRF by +18.1% P@3 and +11.2% R@3 on the 500-instance subset.
Entity RRF outperforms weighted-sum by +58.4% P@3 and +49.4% R@3.

## Tests

```bash
python -m pytest tests/plugins/memory/test_holographic_rrf.py -v
```

20 tests covering RRF math, full integration, numpy-aware HRR, edge cases, and tokenization.

## License

MIT — see LICENSE file.
Original plugin code by dusterbloom (PR #2351).
RRF improvements, entity-boosted RRF, and fact splitting by Axiom (2026).
