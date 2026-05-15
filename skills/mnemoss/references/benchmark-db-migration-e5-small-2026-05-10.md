# Benchmark DB Migration: bge-m3 1024-dim → e5-small 384-dim

## Context

The LongMemEval benchmark DB (`~/.hermes/benchmarks/longmemeval_benchmark.db`) was migrated from bge-m3 (1024-dim) to multilingual-e5-small (384-dim) embeddings.

## State Before Migration

- Facts table: 10,960 rows
- `facts_vec` (vec0): 2,801 of 10,960 had 1024-dim bge-m3 embeddings; 8,159 missing
- Old model: bge-m3 INT8 ONNX, 1024-dim, Teradata export
- Memory bank: 1024-dim

## Migration Steps

1. **Backup:** `cp longmemeval_benchmark.db longmemeval_benchmark.db.e5-backup`
2. **Drop old vec0:** `DROP TABLE facts_vec`
3. **Recreate vec0:** `CREATE VIRTUAL TABLE facts_vec USING vec0(content_embedding(float[384]))`
4. **Backfill:** Run `backfill_e5_small.py` — encodes all 10,960 facts via e5-small ONNX and inserts into `facts_vec`
5. **Memory bank:** Average all 10,960 embeddings, L2-normalize, write to `memory_banks`

## Pitfall: Bank Aggregation Re-encoding (2026-05-10)

The memory bank aggregation process took 400+ seconds because it re-encoded all 10,960 facts one-by-one. The embeddings were already in `facts_vec` from the backfill. Bank vector is just `mean(facts_vec.vector)` — no model inference needed.

**Correct approach:** `SELECT vector FROM facts_vec` → numpy average → normalize → write. See `mnemoss` skill for the pitfall note.

## Model Details

- **Model:** `multilingual-e5-small-int8.onnx` (Teradata export)
- **Size:** 449 MB
- **Dimensions:** 384
- **Token limit:** 512 (truncates at end, ~30% of facts affected)
- **Input:** 2D `[batch, seq_len]`, no `token_type_ids`
- **Output:** `[token_embeddings, sentence_embedding]` — sentence_embedding is pre-pooled
- **ONNX noise:** "Invalid rank for input" and "invalid expand shape" errors on stderr — benign

## Files

| Path | Role |
|---|---|
| `~/.hermes/benchmarks/longmemeval_benchmark.db` | Target DB (post-migration) |
| `~/.hermes/benchmarks/longmemeval_benchmark.db.e5-backup` | Pre-migration backup |
| `~/.hermes/scripts/backfill_e5_small.py` | Migration script |
| `~/.hermes/mnemoss/embedding.py` | e5-small ONNX wrapper (384-dim, 512-token truncation) |
| `~/.hermes/mnemoss_embeddings/multilingual-e5-small-int8.onnx` | Model file |

## Chinese CMRC 2018 DB Migration (2026-05-10)

Same process, different DB:
- DB: `~/.hermes/benchmarks/chinese_cmrc2018.db`
- Before: 50 facts, 1024-dim bge-m3 embeddings
- After: 50 facts, 384-dim e5-small embeddings
- Backup: `.e5-backup`
- Script: `~/.hermes/scripts/backfill_chinese_e5.py`
- Expected: better Chinese retrieval (e5-small explicitly optimized for multilingual retrieval tasks)

## Pitfall: Check ALL Benchmark DBs

When changing EMBEDDING_DIM or switching embedding models, **every** benchmark DB must be migrated — not just the main one. The Chinese CMRC DB was created before multilingual embeddings existed and still had stale bge-m3 1024d embeddings. Always audit `~/.hermes/benchmarks/*.db` for vec0 tables before and after migration.

## Post-Migration Checklist

- [ ] Verify `facts_vec` has correct row count at new dim
- [ ] Complete memory bank aggregation (reading from `facts_vec`, not re-encoding)
- [ ] Re-run ALL benchmark suites (LongMemEval, Chinese CMRC, Personal Memory, Golden)
- [ ] Update wiki benchmark tracker
- [ ] Commit `embedding.py` changes and backfill scripts to plugin repo
