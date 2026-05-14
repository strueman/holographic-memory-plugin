# Profiling Commands for Holographic Memory

Ready-to-run commands for profiling the retrieval pipeline.

## Warm Query Cost Breakdown

```bash
python3 -c "
import cProfile, pstats, io
from retrieval import FactRetriever
from store import MemoryStore

store = MemoryStore('path/to/db')
retriever = FactRetriever(store)

# Warm up ONNX
retriever.search('test query', limit=10)

# Profile 10 warm queries
profiler = cProfile.Profile()
profiler.enable()
for _ in range(10):
    retriever.search('test query', limit=10)
profiler.disable()

s = io.StringIO()
pstats.Stats(profiler, stream=s).sort_stats('cumulative').print_stats(20)
print(s.getvalue())
"
```

## Distinguish Explicit vs FTS5 Internal SQLite Calls

```bash
python3 -c "
from store import MemoryStore
from retrieval import FactRetriever

store = MemoryStore('path/to/db')
calls = []
store._conn.set_trace_callback(
    lambda sql, *a: calls.append(sql[:200])
    if not sql.strip().startswith('--') else None
)

retriever = FactRetriever(store)
retriever.search('test query', limit=10)

print(f'Explicit SQL calls: {len(calls)}')
for c in calls:
    print(c)
"
```

## Benchmark-Specific Profiling

Profile a specific query type:

```bash
python3 -c "
import time, sqlite3
from retrieval import FactRetriever
from store import MemoryStore

store = MemoryStore('path/to/db')
retriever = FactRetriever(store)

# Load queries from LME DB
conn = sqlite3.connect('path/to/lme.db')
queries = conn.execute('SELECT question FROM lme_instances LIMIT 50').fetchall()
conn.close()

# Warm up
for q in queries[:5]:
    retriever.search(q[0], limit=10)

# Measure
start = time.perf_counter()
for q in queries[5:]:
    retriever.search(q[0], limit=10)
elapsed = time.perf_counter() - start

print(f'45 queries: {elapsed*1000:.1f}ms total')
print(f'Per query: {elapsed/45*1000:.1f}ms')
print(f'QPS: {45/elapsed:.1f}')
"
```

## Optimization Comparison

Measure with and without an optimization:

```bash
python3 -c "
import time, sqlite3
from retrieval import FactRetriever
from store import MemoryStore

store = MemoryStore('path/to/db')

# Load queries
conn = sqlite3.connect('path/to/lme.db')
queries = conn.execute('SELECT question FROM lme_instances LIMIT 50').fetchall()
conn.close()

# Test WITH optimization
retriever = FactRetriever(store)
for q in queries[:5]:
    retriever.search(q[0], limit=10)
start = time.perf_counter()
for q in queries[5:]:
    retriever.search(q[0], limit=10)
elapsed_with = time.perf_counter() - start

# Test WITHOUT optimization (new retriever)
retriever2 = FactRetriever(store)
for q in queries[:5]:
    retriever2.search(q[0], limit=10)
start = time.perf_counter()
for q in queries[5:]:
    retriever2.search(q[0], limit=10)
elapsed_without = time.perf_counter() - start

print(f'WITH:   {elapsed_with*1000:.1f}ms ({elapsed_with/45*1000:.1f}ms/query)')
print(f'WITHOUT: {elapsed_without*1000:.1f}ms ({elapsed_without/45*1000:.1f}ms/query)')
print(f'Delta: {elapsed_without/elapsed_with:.2f}x')
"
```
