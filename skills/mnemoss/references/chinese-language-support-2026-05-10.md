# Chinese Language Support — Implementation Notes

> Date: 2026-05-10
> Status: Implemented on `chinese-support` branch, merged to main.

## Problem

FTS5's unicode61 tokenizer tokenizes Chinese characters individually (single-char tokens), making keyword search completely ineffective for Chinese queries. FTS5 MATCH on "中国菜" returns 0 results because no fact contains the single character "中" as a standalone keyword.

## Solution: Smart Query Routing

```
Query → _has_chinese() → English? → Standard FTS5 path (unchanged)
                  → Chinese? → FTS5 first → 0 results? → LIKE fallback
```

### Key Methods

**retrieval.py:**
- `_RE_CHINESE = re.compile(r'[\u4e00-\u9fff]')` — pre-compiled CJK regex
- `_has_chinese(text)` — static method, ~0.01ms hot path
- `_chinese_tokenize(text)` — jieba segmentation with character-level fallback
- `_chinese_candidates(query)` — LIKE-based search with jieba tokenization
- `_fts_candidates_with_chinese_fallback()` — wraps `_fts_candidates`, falls back to LIKE
- `_escape_like(text)` — escapes `%` and `_` for SQL LIKE safety

**store.py:**
- `_extract_chinese_entities(text)` — jieba POS tagging (nr/ns/nt/nz) with fallback

### Design Principles
- **Additive only:** English path completely unchanged
- **jieba optional:** All Chinese functionality falls back gracefully
- **Signal reuse:** vec/Jaccard/HRR signals still active for Chinese queries (not LIKE-only)

## Pitfalls

### Double-escaping in patch tool
When patching `_escape_like`, the patch tool can double-escape backslashes. The correct implementation uses raw strings:
```python
def _escape_like(text: str) -> str:
    return text.replace('%', r'\\%').replace('_', r'\\_')
```
Do NOT write `'\\\\\\\\%'` — the patch tool will double-escape it to `'\\\\\\\\\\\\%'`.

### Subagent test file persistence
Subagents may create test files that don't persist to the plugin clone. Always verify:
```bash
ls ~/.hermes/mnemoss/tests/test_bilingual.py
```
If missing, recreate the file directly.

### Sync script invocation
The sync script is not on PATH. Use:
```bash
bash ~/.hermes/skills/mnemoss/scripts/mnemoss-sync.sh full
```
Not `mnemoss-sync.sh full`.

## Benchmark Results

### English (No Regression)

| Metric | Before (v0.4.2) | After (v0.4.3) | Delta |
|--------|-----------------|----------------|-------|
| P@3 (weighted) | 0.319 | 0.327 | +2.5% (noise) |
| Total tests | 158 | 184 | +26 bilingual |

No English regression confirmed.

### Chinese (CMRC 2018 — 50 facts, 50 queries)

**Before (no embeddings, 0% vec coverage):**
| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | 0.000 | 0.140 | +14.0% |
| P@5 | 0.000 | 0.140 | +14.0% |
| R@3 | 0.000 | 0.140 | +14.0% |
| R@5 | 0.000 | 0.140 | +14.0% |
| NDCG@3 | 0.000 | 0.140 | +14.0% |

Per-query wins (P@3): RRF=7, Weighted=0, Tie=43

**After (bge-m3 INT8, 1024-dim, 100% vec coverage):**
| Metric | Weighted | RRF | Delta |
|--------|----------|-----|-------|
| P@3 | 0.000 | **0.300** | **+30.0%** |
| P@5 | 0.000 | **0.180** | **+18.0%** |
| R@3 | 0.000 | **0.900** | **+90.0%** |
| R@5 | 0.000 | **0.900** | **+90.0%** |
| NDCG@3 | 0.000 | **0.853** | **+85.3%** |

Per-query wins (P@3): RRF=45, Weighted=0, Tie=5

**By query type (after bge-m3):**
| Type | Count | P@3 |
|------|-------|-----|
| entity | 38 | 0.298 |
| general | 9 | 0.296 |
| numeric | 1 | 0.333 |
| reasoning | 2 | 0.333 |

**Key findings:**
- bge-m3 embeddings dramatically improve Chinese retrieval: P@3 0.140→0.300 (+114%), R@3 0.140→0.900 (+543%)
- RRF found 45/50 queries with P@3 win (vs 7/50 without embeddings)
- Entity queries: P@3=0.298, general: P@3=0.296 — consistent across types
- Chinese P@3 (0.300) now comparable to Personal Memory English (0.319) despite much smaller corpus
- FTS5 returns 0 for pure Chinese — LIKE fallback + bge-m3 vec fills the gap

## Embedding Model Decision (2026-05-10, updated)

**Selected: BAAI/bge-m3 INT8 ONNX (Teradata export)**

- **Why Teradata over Xenova:** Teradata INT8 produces well-calibrated similarities (0.74 unrelated vs 0.88 related). Xenova INT8 produces inflated similarities (0.96 unrelated vs 0.97 related), causing false positive structural links
- **Why INT8 over Q4:** No true Q4 ONNX exists for bge-m3 (the "Q4" ONNX is 1.25GB Transformers.js format). INT8 is 543-568MB, 99%+ accuracy, 2x CPU speedup with AVX512-VNNI support
- **Model:** Teradata/bge-m3 `onnx/model_int8.onnx` (543MB, 1024d)
- **License:** MIT
- **Languages:** 100+ (Chinese + English + cross-lingual)
- **Tradeoff:** 1024d requires pipeline change from current 384d, ~50-100ms/query vs ~10ms for MiniLM
- **Instruction prefix required:** "Represent this sentence for searching relevant passages: " + text

## PR #1 Context

PR #1 (kelongyan/main) exists but is un-mergeable — built against old version. This implementation is a fresh rewrite rebased on v0.4.2 with proper integration into the weighted RRF pipeline. Credit @kelongyan by closing PR #1 with thank-you comment and co-authoring the commit.