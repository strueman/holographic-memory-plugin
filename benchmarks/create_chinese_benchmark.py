"""Create Chinese CMRC 2018 benchmark database from source dataset.

Downloads CMRC 2018 trial data, extracts facts (paragraph contexts) and
queries (questions), maps ground truth, and creates a benchmark DB with
the full MemoryStore schema.

Usage:
    python3 benchmarks/create_chinese_benchmark.py
    python3 benchmarks/create_chinese_benchmark.py --facts 100 --queries 100
"""

import argparse
import json
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

DATASET_URL = "https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip"
DEFAULT_DB = Path(__file__).parent / "chinese_cmrc2018.db"

# Schema matching the Mnemoss store (simplified for benchmark)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    tags TEXT DEFAULT '',
    trust_score REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    content, tags, content='facts', content_rowid='fact_id'
);

CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags) VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_d AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags) VALUES('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_u AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags) VALUES('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags) VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT DEFAULT 'general',
    expected TEXT NOT NULL
);
"""


def download_dataset(output_dir: Path) -> Path:
    """Download and extract CMRC 2018 dataset."""
    if not HAS_REQUESTS:
        print("Error: requests not installed", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "cmrc2018_public.zip"

    if not zip_path.exists():
        print(f"Downloading {DATASET_URL} ...")
        r = requests.get(DATASET_URL, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        print(f"Downloaded ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)

    return output_dir / "trial.json"


def classify_query_type(question: str) -> str:
    """Classify Chinese query into type based on content patterns."""
    # Entity queries: ask about specific entities/names
    entity_patterns = ['谁', '什么', '哪', '哪个', '哪里', '何', '称为', '名字']
    # Numeric queries: ask about numbers/dates
    numeric_patterns = ['多少', '几', '年', '日期', '时间', '数量']
    # Reasoning queries: require inference
    reasoning_patterns = ['因为', '为什么', '原因', '关系', '区别']

    for p in entity_patterns:
        if p in question:
            return 'entity'
    for p in numeric_patterns:
        if p in question:
            return 'numeric'
    for p in reasoning_patterns:
        if p in question:
            return 'reasoning'
    return 'general'


def extract_facts_and_queries(trial_path: Path, max_facts: int, max_queries: int):
    """Extract facts and queries from CMRC 2018 trial data."""
    with open(trial_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    facts = []  # (content, section_title)
    queries = []  # (query_id, question, query_type, expected_fact_ids)
    fact_id_map = {}  # paragraph_id -> fact_id

    fact_counter = 0
    query_counter = 0

    for section in data.get('data', []):
        if fact_counter >= max_facts:
            break

        section_title = section.get('title', '')
        for para in section.get('paragraphs', []):
            if fact_counter >= max_facts:
                break

            context = para.get('context', '').strip()
            if not context:
                continue

            fact_counter += 1
            para_id = para.get('id', f'para_{fact_counter}')
            fact_id_map[para_id] = fact_counter

            facts.append((context, section_title))

            # Extract queries for this paragraph
            for qa in para.get('qas', []):
                if query_counter >= max_queries:
                    break

                question = qa.get('question', '').strip()
                if not question:
                    continue

                query_counter += 1
                qid = qa.get('id', f'q_{query_counter}')
                qtype = classify_query_type(question)

                # Ground truth: map answer spans back to this paragraph's fact_id
                answers = qa.get('answers', [])
                expected_ids = json.dumps([fact_counter])

                queries.append((qid, question, qtype, expected_ids))

    return facts, queries


def create_benchmark_db(db_path: Path, facts, queries):
    """Create the benchmark database and populate it."""
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)

    # Insert facts
    for content, section_title in facts:
        tags = f"chinese,cmrc2018,{section_title}"
        conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, 'chinese', ?, ?)",
            (content, tags, 0.7),
        )
    conn.commit()

    # Insert queries
    for qid, question, qtype, expected_ids in queries:
        conn.execute(
            "INSERT INTO queries (query_id, query, query_type, expected) VALUES (?, ?, ?, ?)",
            (qid, question, qtype, expected_ids),
        )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    qcount = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    conn.close()

    return count, qcount


def main():
    parser = argparse.ArgumentParser(description="Create Chinese CMRC 2018 benchmark DB")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--facts", type=int, default=50, help="Max facts to extract")
    parser.add_argument("--queries", type=int, default=50, help="Max queries to extract")
    args = parser.parse_args()

    db_path = Path(args.output)
    tmp_dir = Path("/tmp/cmrc2018_download")

    print(f"Downloading CMRC 2018 dataset...")
    trial_path = download_dataset(tmp_dir)

    print(f"Extracting facts and queries...")
    facts, queries = extract_facts_and_queries(trial_path, args.facts, args.queries)

    print(f"Creating benchmark DB at: {db_path}")
    fact_count, query_count = create_benchmark_db(db_path, facts, queries)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"CHINESE CMRC 2018 BENCHMARK DB")
    print(f"{'=' * 50}")
    print(f"  Facts:     {fact_count}")
    print(f"  Queries:   {query_count}")

    # Query type breakdown
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    type_counts = conn.execute(
        "SELECT query_type, COUNT(*) as cnt FROM queries GROUP BY query_type"
    ).fetchall()
    print(f"  Query types:")
    for row in type_counts:
        print(f"    {row['query_type']}: {row['cnt']}")
    conn.close()

    print(f"{'=' * 50}")
    print(f"\nNext: backfill embeddings and HRR vectors")
    print(f"  cd ~/.hermes/mnemoss")
    print(f"  python3 -c \"\"\"")
    print(f"  from store import MemoryStore")
    print(f"  store = MemoryStore('{db_path}')")
    print(f"  store.backfill_embeddings()")
    print(f"  \"\"\"")


if __name__ == "__main__":
    main()
