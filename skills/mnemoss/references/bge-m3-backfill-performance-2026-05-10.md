# bge-m3 Backfill Performance

> Date: 2026-05-10
> Status: Documented — backfill is slower than expected

## Problem

After upgrading from MiniLM (384d) to bge-m3 (1024d), the LongMemEval DB had 0 embeddings because the schema migration recreated the vec0 table with 1024-dim but dropped the old 384-dim embeddings. Backfilling took much longer than expected.

## Performance

| Metric | Theoretical | Actual |
|--------|-------------|--------|
| Encoding time | ~64ms/fact | ~64ms/fact (correct) |
| Backfill rate | ~15 facts/s | ~1 fact/s |
| Total time (10,960 facts) | ~12 minutes | ~2-3 hours |

## Root Cause

The backfill rate is ~1 fact/s, not ~15 facts/s, due to:
1. **SQLite I/O overhead** — Each fact requires DELETE + INSERT per fact
2. **Python-GIL contention** — ONNX calls block the GIL
3. **Batch size** — Small batches (20-50) don't amortize ONNX overhead

## Solution

Use batch encoding (50 texts per ONNX call) instead of sequential encoding:

```python
# Batch encode 50 texts at once
encoded_ids = [tok.encode(t).ids for t in texts]
max_len = max(len(ids) for ids in encoded_ids)

input_ids = np.array([ids + [0] * (max_len - len(ids)) for ids in encoded_ids], dtype=np.int64)
attention_mask = np.array([[1] * len(ids) + [0] * (max_len - len(ids)) for ids in encoded_ids], dtype=np.int64)

outputs = sess.run(None, {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
})
```

This reduces per-fact overhead but the rate is still ~1 fact/s due to SQLite I/O.

## Partial Coverage Impact

With 23.4% vec coverage (after 3 minutes of backfill):
- bge-m3 P@3=0.259 vs MiniLM P@3=0.264 (nearly tied)
- RRF=46 wins, Weighted=16 wins, Tie=438

Full coverage should push bge-m3 ahead of MiniLM on the full 500-query benchmark.

## Scripts

- `benchmarks/backfill_lme_batch.py` — Batch backfill script (50 texts/batch)
- `benchmarks/backfill_chinese_embeddings.py` — Backfill for Chinese benchmark DB (50 facts, faster)
