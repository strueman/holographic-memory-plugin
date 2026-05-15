# bge-m3 INT8 ONNX — Model Selection Notes

> Date: 2026-05-10
> Status: Replaced all-MiniLM-L6-v2 (384d) with BAAI/bge-m3 INT8 ONNX (1024d)

## Model Comparison

| Model | Dims | Languages | ONNX Size | License |
|-------|------|-----------|-----------|---------|
| all-MiniLM-L6-v2 | 384 | English-only | 223MB | Apache 2.0 |
| bge-m3 INT8 | 1024 | 100+ | 543MB | MIT |

## Teradata vs Xenova INT8 ONNX

**Selected: Teradata/bge-m3 `onnx/model_int8.onnx`**

| Metric | Xenova INT8 | Teradata INT8 |
|--------|-------------|---------------|
| Unrelated text similarity | 0.96 | **0.74** |
| Related text similarity | 0.97 | **0.88** |
| Self-similarity | 1.00 | 1.00 |

Xenova's INT8 produces inflated similarities (0.96 for unrelated texts), causing false positive structural links. Teradata's is well-calibrated.

### Why Teradata
- Better similarity calibration (0.74 vs 0.96 for unrelated)
- Proper separation between related (0.88) and unrelated (0.74) texts
- Allows embedding link threshold at 0.75 (vs 0.70 which was too low for Xenova)

## Usage

### Instruction Prefix Required
bge-m3 requires an instruction prefix for proper embeddings:
```
"Represent this sentence for searching relevant passages: " + text
```

Without it, mean-pooled embeddings produce inflated similarities.

### API
```python
from embedding import encode, is_available, EMBEDDING_DIM

# encode() automatically prepends instruction prefix
vec = encode("Simon uses Debian Trixie")
# Returns 1024-dim L2-normalized np.float32 array
```

### Schema Migration
- EMBEDDING_DIM: 384 → 1024
- `_init_vec0()` auto-recreates `facts_vec` table if dimension mismatch detected
- Run `store.backfill_embeddings()` after upgrade to regenerate all embeddings

## Benchmark Impact

### LongMemEval (50 queries, 10,960 facts)
| Metric | MiniLM (v0.4.0) | bge-m3 (v0.4.4) | Delta |
|--------|-----------------|-----------------|-------|
| P@3 RRF | 0.380 | **0.400** | **+16.3%** |
| R@3 RRF | 0.557 | **0.587** | **+23.1%** |
| NDCG@3 RRF | 0.538 | **0.581** | **+20.8%** |

### Chinese CMRC 2018 (50 facts, 50 queries)
| Metric | Before (no vec) | After (bge-m3) | Delta |
|--------|-----------------|----------------|-------|
| P@3 RRF | 0.140 | **0.300** | **+114%** |
| R@3 RRF | 0.140 | **0.900** | **+543%** |
| NDCG@3 RRF | 0.140 | **0.853** | **+509%** |

Chinese P@3 (0.300) now comparable to Personal Memory English (0.319) despite smaller corpus.

## Model Files
- **ONNX:** `~/.hermes/mnemoss_embeddings/bge-m3-int8.onnx` (543MB)
- **Tokenizer:** `~/.hermes/mnemoss_embeddings/bge-m3-tokenizer.json`
- **Download:** Teradata/bge-m3 on HuggingFace

## Performance

- Model size: 543MB (INT8)
- RAM during inference: ~1-1.5GB
- CPU speed on Ryzen 9 8945HS: ~64ms/query (vs ~10ms for MiniLM) — **6x slower**
- First encoding includes model load (~2s), subsequent encodings ~64ms each
- Full 10,960-fact backfill takes ~12 minutes on CPU
- AVX512-VNNI optimized via onnxruntime CPUExecutionProvider

### CPU Performance Implications

bge-m3 is significantly slower than MiniLM on CPU. This matters for:
- **Initial model load:** ~2s first encoding (one-time cost)
- **Per-query latency:** ~64ms vs ~10ms for MiniLM
- **Batch backfill:** ~1 fact/s (not 64ms/fact) — SQLite I/O overhead dominates. Full 10,960-fact backfill takes ~2-3 hours, not 12 minutes.
- **Embedding storage:** 1024-dim floats = 4KB per embedding (vs 1.5KB for 384-dim)

### Backfill Performance (2026-05-10)

**Reality check:** The theoretical 64ms/fact assumes in-memory batch processing. Actual backfill is ~1 fact/s due to:
- SQLite I/O overhead (INSERT/DELETE per fact)
- ONNX model loading per batch
- Python-GIL contention

**Workaround:** Use batch encoding (50 texts per ONNX call) instead of sequential encoding. Still slow (~1 fact/s) but more efficient than single-encoding.

**Partial coverage impact:** With 23.4% vec coverage, bge-m3 P@3=0.259 vs MiniLM P@3=0.264 (nearly tied). Full coverage should push bge-m3 ahead.

### Benchmark: Full 500-query LongMemEval (v0.4.5)

| Model | P@3 | P@5 | R@3 | NDCG@3 |
|-------|-----|-----|-----|--------|
| MiniLM v0.1.0 | 0.264 | 0.173 | 0.462 | 0.443 |
| bge-m3 v0.4.5 | 0.249 | 0.166 | 0.435 | 0.420 |

**Note:** The 50-query curated subset showed bge-m3 ahead (P@3 0.400 vs 0.380), but the full 500-query benchmark shows MiniLM slightly ahead. The subset was not representative — use full benchmark for accurate comparison.

### Schema Migration Pitfall

When changing EMBEDDING_DIM, the old embeddings become incompatible with the new vec0 table schema. The `_init_vec0()` method recreates the table but **does not migrate old embeddings** — they are lost. Always run `backfill_embeddings()` after a dimension change.
