# Chinese CMRC 2018 DB — e5-small 384d Migration

> Date: 2026-05-10
> DB: `~/.hermes/benchmarks/chinese_cmrc2018.db`

## Before

- 50 facts, 50 queries (CMRC 2018 trial subset)
- Embeddings: bge-m3 INT8 ONNX, 1024-dim, 100% coverage
- Created before multilingual embeddings existed

## Migration

1. Backup: `cp chinese_cmrc2018.db chinese_cmrc2018.db.e5-backup`
2. Drop old vec0: `DROP TABLE facts_vec`
3. Recreate: `CREATE VIRTUAL TABLE facts_vec USING vec0(embedding float[384])`
4. Backfill: `python3 ~/.hermes/scripts/backfill_chinese_e5.py` — 50/50 facts embedded, 0 failures
5. Embedding size: 1536 bytes (384-dim float32)

## Expected Improvement

e5-small is explicitly optimized for multilingual retrieval tasks (trained on MS MARCO + NLI + cross-lingual pairs). Expect better P@3/P@5 on Chinese queries vs bge-m3, which was primarily an English model with multilingual support as a secondary feature.

## Files

| Path | Role |
|---|---|
| `~/.hermes/benchmarks/chinese_cmrc2018.db` | Target DB (post-migration) |
| `~/.hermes/benchmarks/chinese_cmrc2018.db.e5-backup` | Pre-migration backup |
| `~/.hermes/scripts/backfill_chinese_e5.py` | Migration script |
| `~/.hermes/mnemoss/embedding.py` | e5-small ONNX wrapper |