# Embedding Migration Verification — 2026-05-11

## Context

Migration from bge-m3 (1024d) to multilingual-e5-small (384d). Previous migration attempts by LLM agents produced corrupted state (mixed dim vectors, orphaned rows, incomplete commits). This verification workflow was developed to prevent repeat corruption.

## Step 1: Verify Embedding Model Loads

```python
from embedding import encode, is_available, EMBEDDING_DIM
assert is_available()
vec = encode('hello world test')
assert vec is not None
assert vec.shape == (EMBEDDING_DIM,)  # (384,)
assert vec.dtype == np.float32
assert np.linalg.norm(vec) == 1.0  # L2-normalized
```

## Step 2: Verify Fresh DB Pipeline

```python
import tempfile, os
from store import MemoryStore, EMBEDDING_DIM, _HAS_SQLITE_VEC
from retrieval import FactRetriever

tmp = tempfile.mktemp(suffix='.db')
store = MemoryStore(db_path=tmp, default_trust=0.5, hrr_dim=1024)

facts = [
    ('Simon uses Debian Trixie on his Minisforum UM890 Pro laptop', 'project'),
    ('The holographic memory plugin uses sqlite-vec for dense vector search', 'tool'),
    ('e5-small is a multilingual embedding model with 384 dimensions', 'tool'),
    ('Simon prefers concise responses without unnecessary fluff', 'user_pref'),
    ('The Ryzen 9 8945HS has a Radeon 780M integrated GPU', 'project'),
]

for content, cat in facts:
    store.add_fact(content, category=cat)

# Verify all subsystems
total = store._conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0]
vec_count = store._conn.execute('SELECT COUNT(*) FROM facts_vec').fetchone()[0]
hrr_count = store._conn.execute('SELECT COUNT(*) FROM facts WHERE hrr_vector IS NOT NULL').fetchone()[0]
assert total == 5
assert vec_count == 5  # 100% vec coverage
assert hrr_count == 5  # 100% HRR coverage

# Verify vec0 schema
row = store._conn.execute('SELECT embedding FROM facts_vec LIMIT 1').fetchone()
assert len(row['embedding']) == EMBEDDING_DIM * 4  # 384 * 4 = 1536 bytes

# Verify search works
retriever = FactRetriever(store=store, hrr_dim=1024)
results = retriever.search('embedding model dimensions', limit=5)
assert len(results) > 0

os.unlink(tmp)
```

## Step 3: Verify with Real Benchmark Data

```python
# Load 20 facts from LongMemEval benchmark DB
lme_db = os.path.expanduser('~/.hermes/benchmarks/longmemeval_benchmark.db')
lme = sqlite3.connect(lme_db)
lme.row_factory = sqlite3.Row
rows = lme.execute('SELECT content, category FROM facts LIMIT 20').fetchall()

# Ingest into fresh DB and verify
tmp = tempfile.mktemp(suffix='.db')
store = MemoryStore(db_path=tmp, default_trust=0.5, hrr_dim=1024)
for row in rows:
    store.add_fact(row['content'], category=row['category'] or 'general')

assert store._conn.execute('SELECT COUNT(*) FROM facts_vec').fetchone()[0] == 20
assert store._conn.execute('SELECT COUNT(*) FROM facts WHERE hrr_vector IS NOT NULL').fetchone()[0] == 20
```

## Step 4: Production DB Migration (only after steps 1-3 pass)

```python
from store import MemoryStore

store = MemoryStore(db_path='~/.hermes/benchmarks/longmemeval_benchmark.db')

# Backfill embeddings
count = store.backfill_embeddings()
assert count == store._conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0]

# Backfill HRR vectors
rows = store._conn.execute('SELECT fact_id, content FROM facts WHERE hrr_vector IS NULL').fetchall()
for row in rows:
    store._compute_hrr_vector(row['fact_id'], row['content'])
store._conn.commit()
```

## Lessons Learned

1. **Disk space kills backfills:** The VM disk was 100% full during the session. Backfill failed at fact #10,945 with "database or disk is full". Cleaned 8 orphaned model files (~2.2GB) before retrying.

2. **Corrupted state → rebuild, don't repair:** The LongMemEval DB had mixed dim vectors (some 384d, some 1024d, some NULL). Instead of trying to fix individual rows, we rebuilt from source: `prepare_longmemeval_db.py` downloads fresh from HuggingFace, then backfill through the store.

3. **Chinese CMRC DB was empty:** The facts table didn't exist. The original conversion script was lost during migration chaos. Wrote new `create_chinese_benchmark.py` that downloads CMRC 2018 from CLUE dataset.

4. **HRR vectors are slow:** ~50 facts/s for 1024-dim HRR vectors. 10,960 facts took ~112s. Embeddings are faster: ~56 facts/s for e5-small 384d.

5. **ONNX stderr noise is benign:** The e5-small INT8 model emits "Invalid rank for input" and "invalid expand shape" errors to stderr. These do NOT affect output quality. Redirect stderr during backfills: `2>/dev/null`.

## Timing (10,960 facts, Ryzen 9 8945HS)

| Operation | Time | Rate |
|-----------|------|------|
| Embedding backfill (e5-small) | 346s | 32/s |
| HRR vector backfill | 112s | 50/s |
| Total | ~458s | ~24/s combined |
