# sqlite-vec Integration Session (2026-05-05)

## Implementation Summary

Added dense vector search via sqlite-vec as a 4th ranking source in the RRF fusion pipeline.

### Files Changed
- `store.py` — `_HAS_SQLITE_VEC` flag, `_init_vec0()`, `_compute_embedding()`, `backfill_embeddings()`
- `retrieval.py` — `_vec_candidates()`, RRF fusion replacing weighted-sum in `search()`
- `embedding.py` — NEW: ONNX embedding module (all-MiniLM-L6-v2, 384-dim)
- `embeddings/.gitignore` — NEW: excludes model files from git
- `tests/test_sqlite_vec.py` — NEW: 15 tests

### Key Decisions

1. **Model not bundled in git** — 87MB ONNX model too large. Auto-downloaded on first use to `~/.hermes/mnemoss_embeddings/`
2. **ONNX over PyTorch** — PyTorch install failed due to disk space (18GB VM). ONNX runtime + tokenizers = ~20MB vs ~2GB for torch
3. **Mean pooling + L2 normalization** — Standard sentence embedding approach, matches all-MiniLM-L6-v2 convention
4. **RRF C=60** — Same constant as existing FTS5 RRF, keeps scoring consistent
5. **Graceful degradation** — If sqlite-vec/onnxruntime unavailable, vec search silently disabled, pipeline falls back to 3-source RRF

### Pitfalls Discovered

1. **sqlite-vec KNN requires `k = ?` not `LIMIT`** — vec0 virtual table throws `OperationalError: A LIMIT or 'k = ?' constraint is required on vec0 knn queries` if you use LIMIT. Must use `WHERE embedding MATCH ? AND k = ?`

2. **Relative import fails when running module directly** — `from . import embedding` fails with "attempted relative import with no known parent package" when running `python3 -c` from the plugin dir. Fix: try/except with fallback to absolute import.

3. **Disk space critical on 18GB VM** — PyTorch install consumed all available space. Always check `df -h /` before large installs. Clean huggingface cache (`~/.cache/huggingface`) and pip cache to free space.

4. **sqlite-vec must be loaded per-connection** — `conn.enable_load_extension(True)` + `sqlite_vec.load(conn)` needed before any vec0 query. The store's `_init_vec0()` handles this, but `_vec_candidates()` also loads it defensively.

### vstash Paper Insights (arxiv 2604.15484)

- 74.5% of queries produce disagreement between vector-heavy and FTS-heavy search — this is a free training signal
- Post-RRF scoring (frequency+decay, cross-encoder reranking) FAILED to improve NDCG — keep RRF simple
- 20.9ms median latency at 50K chunks — sqlite-vec is fast enough for real-time use
- Adaptive per-query IDF weighting improves results — future enhancement candidate

### Test Results

All 15 tests pass:
- vec table creation, embedding storage, dimension verification
- vec candidates with/without category filter
- RRF fusion scoring, relevance ranking
- Backfill idempotency
- Evidence-gap compatibility
- Graceful degradation when vec unavailable

### Disk Space Management

```bash
# Check space
df -h /

# Clean caches (frees ~6GB)
rm -rf ~/.cache/huggingface ~/.cache/pip

# Remove partial torch installs
rm -rf .venv/lib/python3.11/site-packages/{torch,nvidia-*,triton,cuda-*}
```
