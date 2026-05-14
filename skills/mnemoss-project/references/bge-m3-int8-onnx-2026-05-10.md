# bge-m3 INT8 ONNX Integration (v0.4.4)

## Model Details

**Model:** BAAI/bge-m3 INT8 ONNX (Teradata export)
- 1024-dim embeddings (was 384 with all-MiniLM-L6-v2)
- 543MB model, ~1-1.5GB RAM during inference
- 100+ languages (Chinese + English + everything)
- MIT license
- Cache: `~/.hermes/mnemoss_embeddings/bge-m3-int8.onnx`
- Tokenizer: `~/.hermes/mnemoss_embeddings/bge-m3-tokenizer.json`

## ONNX Specifics

**Inputs:** `input_ids` (int64), `attention_mask` (int64) — NO `token_type_ids`
**Output:** `last_hidden_state` (batch, seq_len, 1024)
**Instruction prefix required:** `"Represent this sentence for searching relevant passages: " + text` — without it, embeddings are poorly calibrated (0.96 similarity for unrelated texts)

## Teradata vs Xenova INT8 Comparison

| Metric | Xenova INT8 | Teradata INT8 |
|--------|-------------|---------------|
| Unrelated text similarity | 0.96 | **0.74** |
| Related text similarity | 0.97 | **0.88** |
| Self-similarity | 1.00 | 1.00 |
| Model size | 568MB | 543MB |
| Calibration | Poor (inflated) | Good |

**Why Teradata wins:** Xenova INT8 produces 0.96 similarity for completely unrelated texts (sourdough bread vs quantum entanglement), causing false positive structural links. Teradata gives 0.74 for unrelated, 0.88 for related — proper separation.

**Embedding link threshold raised from 0.70 to 0.75** to accommodate bge-m3's similarity distribution.

## Schema Migration

**EMBEDDING_DIM changed from 384 to 1024.** The `_init_vec0()` method handles migration:

1. Checks if `facts_vec` table exists
2. Attempts a test insert with 1024-dim zero vector
3. If it fails (dimension mismatch), drops and recreates the table
4. Existing embeddings are lost — call `store.backfill_embeddings()` to regenerate

## Benchmark Results

**LongMemEval (50 queries, 10,960 facts):**

| Metric | MiniLM (384d) | bge-m3 (1024d) | Delta |
|--------|---------------|----------------|-------|
| P@3 RRF | 0.380 | **0.400** | **+16.3%** |
| R@3 RRF | 0.557 | **0.587** | **+23.1%** |
| NDCG@3 RRF | 0.538 | **0.581** | **+20.8%** |

**Chinese CMRC 2018 (50 facts, 50 queries):**

| Metric | Before (no vec) | After (bge-m3) | Delta |
|--------|-----------------|----------------|-------|
| P@3 RRF | 0.140 | **0.300** | **+114%** |
| R@3 RRF | 0.140 | **0.900** | **+543%** |
| NDCG@3 RRF | 0.140 | **0.853** | **+509%** |

## Quantization Research

**Q4 vs INT8 for CPU:**
- No true Q4 ONNX exists for bge-m3. The only "Q4" ONNX file (Xenova's `model_q4.onnx`) is 1.25GB — larger than INT8.
- INT8 provides 99%+ accuracy retention with ~2x CPU speedup via AVX512-VNNI.
- True Q4 benefits require GGUF + llama.cpp, not ONNX.
- INT8 is the correct choice for ONNX CPU inference.

## Implementation Notes

**embedding.py changes:**
- Removed `token_type_ids` input (bge-m3 doesn't use it)
- Added instruction prefix to all encode() calls
- Updated model/tokenizer URLs to Teradata export
- Updated cache filenames to `bge-m3-int8.onnx` and `bge-m3-tokenizer.json`

**store.py changes:**
- EMBEDDING_DIM: 384 → 1024
- _EMB_LINK_THRESHOLD: 0.70 → 0.75
- _init_vec0() migration logic for dimension mismatch
