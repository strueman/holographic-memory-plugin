# Holographic Memory Plugin

A lightweight, single-user memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) using **Holographic Reduced Representation (HRR)** vectors for compositional reasoning, backed by SQLite with FTS5 full-text search and Reciprocal Rank Fusion (RRF) retrieval.

> **Attribution:** Original plugin by [dusterbloom](https://github.com/dusterbloom) (PR [#2351](https://github.com/NousResearch/hermes-agent/pull/2351)), adapted to the `MemoryProvider` ABC.

## Features

- **HRR compositional memory** — bind roles and entities into algebraic vectors; probe and reason across entities using vector unbinding
- **FTS5 full-text search** — fast keyword matching via SQLite's built-in FTS5 engine
- **Jaccard token overlap** — reranking based on shared token sets
- **Reciprocal Rank Fusion (RRF)** — rank-level fusion combining FTS5, Jaccard, and HRR without weight tuning
- **Entity resolution** — automatic entity extraction and linking from fact content
- **Trust scoring** — asymmetric trust adjustment based on helpful/unhelpful feedback
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
┌─────────────────────────────────────────────────┐
│                 FactRetriever                    │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐ │
│  │ FTS5    │  │ Jaccard  │  │ HRR (optional) │ │
│  │ rank    │  │ rank     │  │ rank           │ │
│  └────┬────┘  └────┬─────┘  └───────┬────────┘ │
│       └────────────┴────────────────┘           │
│                   │                             │
│           Reciprocal Rank Fusion (RRF)          │
│           C / (rank + C)                        │
│                   │                             │
│          Trust-weighted score                   │
└───────────────────┬─────────────────────────────┘
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

The `FactRetriever.search()` method uses a three-stage pipeline:

1. **FTS5 candidate retrieval** — fast keyword search via SQLite FTS5, returns top 3× limit candidates
2. **Independent ranking** — Jaccard and HRR scores computed within candidate set
3. **RRF fusion** — scores combined via `60/(rank + 60)`, no weight tuning needed

## Memory Types

| Type | Method | Description |
|------|--------|-------------|
| **Holographic probe** | `probe(entity)` | Unbinds entity from memory banks to find associated facts |
| **Holographic related** | `related(entity)` | Finds facts sharing structural connections with an entity |
| **Multi-entity reason** | `reason([e1, e2, ...])` | Vector-space JOIN across multiple entities (AND semantics) |
| **Contradiction check** | `contradict()` | Finds facts with high entity overlap but low content similarity |
| **FTS5 search** | `search(query)` | Full-text keyword search with RRF fusion |

## Tests

```bash
python -m pytest tests/plugins/memory/test_holographic_rrf.py -v
```

20 tests covering RRF math, full integration, numpy-aware HRR, edge cases, and tokenization.

## License

MIT — see LICENSE file.
Original plugin code by dusterbloom (PR #2351).
RRF improvements by Axiom (2026).
