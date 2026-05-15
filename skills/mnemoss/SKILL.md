---
name: mnemoss
description: "How to use the holographic memory plugin: fact_store tool actions, entity reasoning, evidence-gap tracking, temporal episodes. Use when storing facts, searching memory, probing entities, or managing structured memory."
version: 0.4.9
author: Hermes Agent
license: MIT
---

# Holographic Memory Plugin — Usage

Structured fact storage with entity resolution, trust scoring, and HRR-based retrieval. Installed as a MemoryProvider plugin.

## What It Does

- Stores durable structured facts about people, projects, preferences, decisions
- Extracts entities automatically from fact content
- Computes trust scores that adjust based on feedback
- Provides algebraic reasoning over entities (probe, reason, contradict)
- Tracks temporal episodes (session clusters)
- Tracks evidence gaps in search results
- Reranks by phrase proximity (closer query terms = higher relevance). Word-boundary aware (v0.4.7 — "car" no longer matches "Carols"). Multiplier reduced to 0.02 (tiebreaker, not dominant).
- Boosts frequently-retrieved and highly-rated facts via access count
- **IDF-weighted FTS5 boosting (v0.4.0):** rare query terms get amplified FTS5 signal
- **Weighted RRF (v0.4.2):** Per-signal weights (FTS=0.5, Vec=0.75, Jaccard=0.75, HRR=0.5, Access=0.5) prevent vec noise from diluting strong FTS5 signals. Formula: `sum(w_i * 1/(rank_i + 60)) / sum(weights)`
- **e5-small multilingual embeddings (v0.4.6):** 384-dim INT8 ONNX model (Teradata export) replaces bge-m3 for better size-to-accuracy tradeoff. Uses pre-pooled `sentence_embedding` output (no manual mean-pooling). "passage: " prefix for facts, "query: " prefix for search queries (v0.4.9 — required for proper e5 retrieval subspace). 512-token max — long facts truncated at end (preserves opening). ONNX runtime emits benign "invalid expand shape" errors — these do NOT affect output. See `references/e5-small-embedding-2026-05-10.md`.
- **bge-m3 INT8 embeddings (v0.4.4):** 1024-dim multilingual ONNX model (Teradata export). 100+ languages including Chinese. Instruction prefix required. See `references/bge-m3-int8-onnx-2026-05-10.md` and `references/chinese-language-support-2026-05-10.md`. (Archived — replaced by e5-small in v0.4.6)
- **Chinese language support (v0.4.3):** Smart query routing — English→FTS5, Chinese→FTS5+LIKE fallback. jieba optional. Zero English regression. e5-small embeddings now handle Chinese natively.
- **Grid search for RRF weights:** Systematically test weight combinations on benchmark to find optimal fusion. See `references/rrf-weight-tuning-2026-05-10.md` for methodology.

## System Prompt

When active, the plugin injects this into the system prompt:

```
# Holographic Memory
Active. N facts stored with entity resolution and trust scoring.
Use fact_store to search, probe entities, reason across entities, or add facts.
Use fact_feedback to rate facts after using them (trains trust scores).
```

When empty:
```
# Holographic Memory
Active. Empty fact store — proactively add facts the user would expect you to remember.
Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.
Use fact_feedback to rate facts after using them (trains trust scores).
```

## Tool: fact_store

### Actions

| Action | Required Args | Description |
|--------|--------------|-------------|
| `add` | `content` | Store a fact. Optional: `category` (user_pref/project/tool/general), `tags` (comma-separated) |
| `search` | `query` | Search facts. Optional: `category`, `min_trust` (default 0.3), `limit` (default 10), `episode_id` |
| `probe` | `entity` | Get ALL facts about a person/thing. Optional: `category`, `limit` |
| `related` | `entity` | Get facts connected to an entity via structural links. Optional: `category`, `limit` |
| `linked` | `fact_id` | Get fact-to-fact structural links (HRR/entity/embedding similarity). Optional: `link_type`, `min_strength`, `limit` |
| `reason` | `entities` (list) | Find facts connected to MULTIPLE entities simultaneously |
| `contradict` | — | Find facts making conflicting claims. Optional: `category`, `limit` |
| `update` | `fact_id` | Update fact content or trust. Optional: `content`, `trust_delta`, `tags`, `category` |
| `remove` | `fact_id` | Remove a fact (also cleans up links) |
| `list` | — | List facts. Optional: `category`, `min_trust`, `limit` |
| `episode` | `episode_id` | Get details of a temporal episode |
| `list_episodes` | — | Browse episodes. Optional: `limit` (default 50), `order` (newest/oldest) |
| `episode_facts` | `episode_id` | Get all facts in an episode. Optional: `limit` (default 50) |
| `segment` | — | Run temporal segmentation on unassigned facts. Optional: `gap_minutes` (default 60) |
| `consolidate` | — | Consolidate related L0 facts into L1 observations. Optional: `category` |

### Usage Patterns

**Store a fact:**
```json
{"action": "add", "content": "Simon prefers concise responses", "category": "user_pref", "tags": "communication, preference"}
```

**Search for facts:**
```json
{"action": "search", "query": "editor config", "min_trust": 0.3, "limit": 10}
```

**Probe an entity:**
```json
{"action": "probe", "entity": "Simon", "limit": 20}
```

**Reason across entities:**
```json
{"action": "reason", "entities": ["Simon", "Debian"], "limit": 10}
```

**Rate a fact:**
```json
{"action": "helpful", "fact_id": 42}
{"action": "unhelpful", "fact_id": 42}
```

## Evidence-Gap Tracking

Every search returns an `evidence_gap` field on the first result:

```json
{
  "evidence_gap": {
    "coverage_ratio": 0.75,
    "covered": ["debian", "stable"],
    "uncovered": ["kernel", "version"],
    "needs_followup": false
  }
}
```

**Rule:** If `needs_followup` is True, refine the search with different terms or broader queries.

## Temporal Episodes

Facts are automatically grouped into episodes — temporal clusters representing sessions or work periods.

- Episode title: top 3 entities joined by " → " (fallback: category)
- Episode summary: fact count, category, entities, avg trust, time span
- Gap threshold: 60 minutes between facts starts a new episode (configurable)

Use `episode_facts` to get all facts in an episode, or `search` with `episode_id` to filter to a single episode.

## When to Use

- **Always run `mnemoss-sync.sh check` first** before modifying the plugin (see [[mnemoss-workflow-2026]])
- When storing durable structured facts the user would expect you to remember
- When answering questions about the user — ALWAYS probe or reason first
- When checking evidence_gap after search: if uncovered components exist, do a follow-up search
- When the user mentions preferences, decisions, project details, or facts worth remembering long-term

## Auto-Extract (on_session_end)

The `on_session_end` hook is registered in `plugin.yaml` but **auto_extract defaults to `false`** in config. When disabled, the hook fires but returns immediately. The built-in regex patterns are narrow (5 patterns matching "I prefer/like/love/use/want/need", "my favorite/preferred/default", "I always/never/usually", "we decided/agreed/chose", "the project uses/needs/requires"). Enable via `hermes config set plugins.memory.holographic.auto_extract true` or replace with the summary-based extraction pipeline ([[small-llm-fact-extraction]]).

## Integration Points

1. **Prefetch:** `prefetch_all(user_message)` runs per-turn search, appends results to user message (not system prompt)
2. **Auto-extract:** Can be wired via `on_session_end()` to auto-extract facts from conversations
3. **Memory mirroring:** `on_memory_write()` mirrors built-in memory writes as facts

## Agent Init Hangs on First Message

**Symptom:** Hermes hangs at "initializing agent" when the first chat message is sent. Works with `--skip-memory` or a different profile.

**Root cause (original):** The plugin loader at `plugins/memory/__init__.py` line 240 globs all `*.py` files in a provider directory and imports them as submodules. If `setup.py` is present, `from setuptools import setup` triggers setuptools' internal argparse which reads `sys.argv` — but `sys.argv` contains `chat` (the CLI command), causing `error: invalid command 'chat'` which hangs the init.

**Root cause (secondary — FIXED 2026-05-11):** `__init__.py` passed `hrr_weight` to `FactRetriever()` but the constructor doesn't accept it (replaced by `_RRF_WEIGHTS` dict in v0.4.2). This caused `TypeError` on agent init. Fix: removed `hrr_weight` from config read and FactRetriever call.

**Fix (applied 2026-05-10):**
1. `setup.py` renamed to `setup_standalone.py` in the plugin repo and agent checkout
2. Plugin loader patched to skip `setup.py`, `setup_standalone.py`, and `conftest.py`

**Fix (applied 2026-05-11):**
3. `hrr_weight` removed from `HolographicMemoryProvider.initialize()` in `__init__.py`

**If this recurs:** Check for any `setup.py` in a memory provider directory. Rename to `setup_standalone.py`. Clear `__pycache__` in the provider directory. Also check that `FactRetriever()` is not being passed params it doesn't accept.

## Debugging "pre-wrapped context; stripped" Warnings

**Symptom:** `agent.log` shows repeated `WARNING agent.memory_manager: memory provider returned pre-wrapped context; stripped`.

**Root cause:** A fact's `content` contains literal `<memory-context>` tags or `[System note: ...]` text. The framework's sanitiser strips these and fires the warning — a false positive.

**Fix:** Update the offending fact to remove literal angle-bracket tags. E.g. replace `fenced <memory-context> block` with `fenced memory-context block` (no angle brackets).

**Prevention:** The ingestion sanitiser in `store.py` (`_sanitize_fact_content()`) filters framework markup on every `add_fact()` and `update_fact()` call.

## Chinese Language Support (v0.4.3)

**Status:** ✅ Merged to main. e5-small embeddings (v0.4.6) now handle Chinese natively (100+ languages).

**Design Principles:**
- **Additive only:** English path completely unchanged — zero regression, zero overhead
- **jieba is optional:** All Chinese functionality falls back gracefully if jieba not installed
- **Programmatic, not LLM-based:** Regex detection + jieba segmentation, no external API calls
- **Lightweight:** Pre-compiled regex (~0.01ms), cached jieba imports

**Pipeline:**
```
Query → _has_chinese() → English? → Standard FTS5 path (unchanged)
                  → Chinese? → FTS5 first → 0 results? → LIKE fallback
```

**Attribution:** Inspired by PR #1 by @kelongyan. Co-author trailer added. PR #1 closed with credit comment.

**See:** `references/chinese-language-support-2026-05-10.md` for Chinese benchmark results.

## Consumer Copy Staleness (Developer Note)

The consumer copy at `~/.hermes/hermes-agent/plugins/memory/mnemoss/` can be severely out of sync with the plugin repo. When this happens, vec/embedding code may be missing.

**Fix:** Run `mnemoss-sync.sh sync-to-agent`, then backfill embeddings. See [[mnemoss-workflow-2026]] for the full workflow.

**Sync script location:** `~/.hermes/skills/mnemoss/scripts/mnemoss-sync.sh`

## Schema Migration (v0.4.6)

**EMBEDDING_DIM = 384** (e5-small multilingual). Model file: `multilingual-e5-small-int8.onnx` in `~/.hermes/mnemoss_embeddings/`.

**⚠️ ONNX runtime noise:** The Teradata e5-small INT8 model emits `"Invalid rank for input"` and `"invalid expand shape"` errors to stderr on every inference. These are **benign** — the model still produces correct 384-dim outputs. Do NOT treat these as failures.

**⚠️ 512-token limit:** e5-small max position embeddings = 512. Facts exceeding this are truncated at the END (opening preserved, which is usually the key content). ~30% of LME facts exceed 512 tokens (~2048 chars).

**⚠️ Model inputs:** The Teradata export expects 2D input `[batch, seq_len]` (NOT 3D) and does NOT require `token_type_ids`. Outputs: `[token_embeddings, sentence_embedding]` where `sentence_embedding` is already mean-pooled.

**⚠️ Pitfall:** When changing EMBEDDING_DIM, the old embeddings become incompatible with the new vec0 table schema. The `_init_vec0()` method recreates the table but **does not migrate old embeddings** — they are lost. Always run `backfill_embeddings()` after a dimension change, or the vec search will return 0 candidates for every query.

**⚠️ Backfill performance:** e5-small backfill is ~56 facts/s on CPU (faster than bge-m3's ~1 fact/s). Full 10,960-fact backfill takes ~3-4 minutes. No batch encoding needed — model handles single inputs efficiently.

**⚠️ Memory bank aggregation pitfall (2026-05-10):** The bank aggregation script (which averages all fact embeddings into a single bank vector) should NEVER re-encode facts from scratch. It must read embeddings directly from `facts_vec` (the vec0 table) instead. Re-encoding 10,960 facts one-by-one for bank aggregation took 400+ seconds — the same embeddings were already stored during backfill. The bank vector is just `mean(facts_vec.vector)` — no model inference needed. Always use `SELECT vector FROM facts_vec` for bank aggregation, never `encode(content)` for every fact.

**⚠️ Benchmark DB migration pitfall (2026-05-10):** When changing embedding models, ALL benchmark databases must be migrated — not just the main one. The Chinese CMRC DB (`chinese_cmrc2018.db`) was created before multilingual embeddings existed and still had stale bge-m3 1024d embeddings. Always audit `~/.hermes/benchmarks/*.db` for vec0 tables before and after migration. See `references/benchmark-db-migration-e5-small-2026-05-10.md` and `references/chinese-cmrc-e5-small-migration-2026-05-10.md`.

## Related Skills

- **mnemoss-project** — Development workflow, repo structure, testing, benchmarking, versioning. Use when modifying store.py, adding features, or managing the plugin repo.
- **small-llm-fact-extraction** — Two-pass pipeline for extracting and deduplicating structured facts from conversation sessions using lightweight models.

## Linked Files

- `references/e5-small-embedding-2026-05-10.md` — e5-small INT8 ONNX model details, ONNX runtime noise, 512-token limit, model comparison table
- `references/bge-m3-int8-onnx-2026-05-10.md` — bge-m3 INT8 ONNX model details, Teradata vs Xenova calibration, schema migration, benchmark impact
- `references/bge-m3-backfill-performance-2026-05-10.md` — Backfill performance analysis: why it's ~1 fact/s (not 64ms/fact), batch encoding workaround, partial coverage impact
- `references/chinese-language-support-2026-05-10.md` — Chinese language support implementation, benchmark results (pre/post bge-m3), embedding model decision
- `references/rrf-weight-tuning-2026-05-10.md` — Weighted RRF tuning methodology and results
- `references/cygnet-multilingual-wordnet-2026-05-10.md` — Multilingual query expansion design
- `references/sqlite-vec-integration-2026-05-05.md` — sqlite-vec integration details
- `references/fts5-vec-debug-2026-05-05.md` — FTS5 + vec debugging notes
- `references/benchmark-db-migration-e5-small-2026-05-10.md` — Benchmark DB migration guide: longmemeval + Chinese CMRC, pitfall checklist
- `references/chinese-cmrc-e5-small-migration-2026-05-10.md` — Chinese CMRC 2018 DB migration from bge-m3 1024d to e5-small 384d