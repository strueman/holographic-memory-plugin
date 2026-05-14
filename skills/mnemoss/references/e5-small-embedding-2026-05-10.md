# e5-small Multilingual Embedding Model

**Date:** 2026-05-10
**Replaces:** bge-m3 1024-dim INT8 (v0.4.4)
**Model:** intfloat/multilingual-e5-small INT8 ONNX (Teradata export)
**Dimension:** 384
**Cache:** `~/.hermes/mnemoss_embeddings/multilingual-e5-small-int8.onnx` (449MB)

## Why e5-small over bge-m3

| Factor | bge-m3 1024d | e5-small 384d |
|--------|-------------|---------------|
| Model size | 543MB | 449MB |
| Embedding dim | 1024 | 384 |
| Backfill speed | ~1 fact/s | ~56 facts/s |
| Chinese quality | Excellent | Good |
| English LME P@3 | 0.249 | TBD (needs benchmark) |
| Disk I/O per fact | 4096 bytes | 1536 bytes |
| 512-token limit | Yes | Yes |

Tradeoff: smaller embeddings = faster I/O, less disk, but potentially lower quality on English retrieval. Chinese support is good (e5-small trained on 100+ languages).

## Model Details

- **URL:** `https://huggingface.co/Teradata/multilingual-e5-small/resolve/main/onnx/model.onnx`
- **Tokenizer:** `https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json`
- **Prefix:** "passage: " for facts, "query: " for search
- **Input:** 2D `[batch, seq_len]`, `input_ids` + `attention_mask` (NO `token_type_ids`)
- **Output:** `[token_embeddings (batch, seq, 384), sentence_embedding (batch, 384)]`
- **Pooling:** Pre-pooled — `sentence_embedding` already mean-pooled
- **Normalization:** L2-normalized in embedding.py
- **Max tokens:** 512 (truncation at end for longer facts)

## ONNX Runtime Noise

The model emits benign errors to stderr on every inference:
```
[E:onnxruntime:, sequential_executor.cc:614 ExecuteKernel] Non-zero status code returned while running Expand node. Name:'/0/auto_model/Expand' Status Message: invalid expand shape
```

**These do NOT affect output.** The model still produces correct 384-dim embeddings. Do NOT treat these as failures — they're ONNX runtime compatibility warnings for certain operators in the INT8 model.

## Embedding Code

File: `~/.hermes/mnemoss/embedding.py`

Key functions:
- `encode(text)` → `np.ndarray` (384,) L2-normalized
- `encode_batch(texts)` → `list[np.ndarray | None]`
- `is_available()` → bool
- `EMBEDDING_DIM = 384`

## Migration Notes

When switching from bge-m3 (1024d) to e5-small (384d):
1. Drop old `facts_vec` table (1024-dim)
2. Recreate with `float[384]`
3. Backfill all facts with `backfill_embeddings()`
4. Update `memory_banks` embedding (mean-pool all fact embeddings)
5. Run benchmark to verify quality

See `~/.hermes/scripts/backfill_e5_small.py` for full migration script.
