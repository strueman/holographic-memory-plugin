"""Load LongMemEval data into a dedicated benchmark SQLite database.

Creates a memory_store.db-compatible schema and populates it with
turns from LongMemEval. The benchmark script can then run against
this isolated database.

Usage:
    python prepare_longmemeval_db.py                         # default (oracle, single-session)
    python prepare_longmemeval_db.py --dataset longmemeval_s
    python prepare_longmemeval_db.py --output ~/benchmark.db
    python prepare_longmemeval_db.py --subset 50             # first 50 instances
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_ALLCAPS      = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b')
_RE_MIXED        = re.compile(r'\b([A-Z]{2,}\s+\d+\s+[A-Z]{2,})\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)

_FRAGMENT_WORDS = frozenset({
    'm happy', 's known', 's prefetch', 's always', 's great',
    's specific', 's even more', 's appearance',
    'what did', 'what are', 'what is', 'what gpu',
})
_MAX_ENTITY_WORDS = 4


def _is_bad_entity(name: str) -> bool:
    """Filter out noise extracted as entity candidates."""
    if len(name) < 4 or len(name.split()) > _MAX_ENTITY_WORDS:
        return True
    lower = name.lower()
    if lower in _FRAGMENT_WORDS:
        return True
    if name.count(' ') >= 3:
        words = name.split()
        if any(w in ('what', 'how', 'why', 'when', 'where', 'which', 'did', 'does', 'can', 'do') for w in words):
            return True
    return False


def _extract_entities(text: str) -> List[str]:
    """Extract entity candidates using multiple regex strategies."""
    seen: set = set()
    candidates: List[str] = []

    def _add(name: str) -> None:
        stripped = name.strip()
        if stripped and stripped.lower() not in seen and not _is_bad_entity(stripped):
            seen.add(stripped.lower())
            candidates.append(stripped)

    for m in _RE_CAPITALIZED.finditer(text):
        _add(m.group(1))

    for m in _RE_ALLCAPS.finditer(text):
        _add(m.group(1))

    for m in _RE_MIXED.finditer(text):
        _add(m.group(1))

    for m in _RE_DOUBLE_QUOTE.finditer(text):
        _add(m.group(1))

    for m in _RE_SINGLE_QUOTE.finditer(text):
        _add(m.group(1))

    for m in _RE_AKA.finditer(text):
        _add(m.group(1))
        _add(m.group(2))

    return candidates


def _resolve_entity(conn: sqlite3.Connection, name: str, cache: dict, entity_inserts: list, name_to_id: dict) -> int:
    """Find existing entity or create new one, using in-memory cache."""
    name_lower = name.lower()
    if name_lower in cache:
        return cache[name_lower]
    # Check existing DB
    row = conn.execute(
        "SELECT entity_id, name FROM entities WHERE name LIKE ?", (name,)
    ).fetchone()
    if row is not None:
        cache[name_lower] = row["entity_id"]
        return int(row["entity_id"])
    # New entity
    new_id = len(name_to_id) + 1000  # offset from DB IDs
    cache[name_lower] = new_id
    name_to_id[name_lower] = name
    entity_inserts.append((name,))
    return new_id


class EntityIndexer:
    """Batch entity extraction and linking for fast ingestion."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cache: dict = {}  # name_lower -> real_entity_id
        self.name_to_id: dict = {}  # name_lower -> name (for DB insert)
        self.next_id: int = 1000
        self._pending_links: list = []  # list of (fact_id, [entity_ids])
        self._load_existing(conn)

    def _load_existing(self, conn: sqlite3.Connection) -> None:
        for row in conn.execute("SELECT entity_id, name FROM entities").fetchall():
            self.cache[row["name"].lower()] = row["entity_id"]
        # Set next_id to avoid collision with existing DB IDs
        max_id = conn.execute("SELECT COALESCE(MAX(entity_id), 0) FROM entities").fetchone()[0]
        self.next_id = max(max_id + 1, 1000)

    def extract_and_resolve(self, text: str, fact_id: int) -> None:
        """Extract entities from text, resolve IDs, collect for batch insert."""
        entities = _extract_entities(text)
        entity_names = []
        for ename in entities:
            name_lower = ename.lower()
            if name_lower not in self.cache:
                self.cache[name_lower] = self.next_id
                self.name_to_id[name_lower] = ename
                self.next_id += 1
            entity_names.append(ename)

        # Collect links (temporarily use cached IDs)
        entity_ids = [self.cache[ename.lower()] for ename in entity_names]
        self._pending_links.append((fact_id, entity_ids))

    def flush(self) -> None:
        """Insert all collected entities and links into DB."""
        if self._pending_links:
            # Insert all new entities
            new_entities = [(self.name_to_id[nl],) for nl in self.name_to_id]
            self.conn.executemany("INSERT OR IGNORE INTO entities (name) VALUES (?)", new_entities)
            self.conn.commit()

            # Build name_lower -> real_entity_id mapping
            entity_names = [self.name_to_id[nl] for nl in self.name_to_id]
            placeholders = ",".join("?" * len(entity_names))
            real_id_map: dict = {}
            for row in self.conn.execute(
                f"SELECT entity_id, name FROM entities WHERE name IN ({placeholders})",
                entity_names,
            ).fetchall():
                real_id_map[row["name"].lower()] = row["entity_id"]

            # Build all fact-entity links with real IDs
            all_links = []
            for fact_id, entity_ids in self._pending_links:
                for eid in entity_ids:
                    # Find which name_lower maps to this cached ID
                    for nl, cached_eid in self.cache.items():
                        if cached_eid == eid and nl in real_id_map:
                            all_links.append((fact_id, real_id_map[nl]))
                            break

            if all_links:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                    all_links,
                )
                self.conn.commit()

        self._pending_links.clear()
        self.name_to_id.clear()

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Schema (matches holographic memory store)
# ---------------------------------------------------------------------------

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
    content,
    tags,
    content='facts',
    content_rowid='fact_id'
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
"""

# Default DB path in the benchmarks directory
DEFAULT_DB = Path(__file__).parent / "longmemeval_benchmark.db"

# HuggingFace dataset URL
DATASET_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{filename}"


def create_db(db_path: Path) -> sqlite3.Connection:
    """Create (or recreate) the benchmark database with the correct schema."""
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def download_dataset(filename: str) -> dict:
    """Download LongMemEval dataset from HuggingFace."""
    if not HAS_REQUESTS:
        print("Error: requests not installed. Install with: python3 -m pip install requests", file=sys.stderr)
        sys.exit(1)

    url = DATASET_URL.format(filename=filename)
    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    print(f"Downloaded {len(data)} instances")
    return data


def ingest_turns(
    conn: sqlite3.Connection,
    instances: List[dict],
    subset: Optional[int] = None,
    source: str = "longmemeval_oracle.json",
) -> dict:
    """Ingest turns from LongMemEval instances into the facts table.

    Also creates an lme_instances table with question text, type, and
    expected answer fact IDs for benchmark query loading.

    Returns stats dict with counts.
    """
    if subset:
        instances = instances[:subset]

    conn.execute("DELETE FROM facts")  # clear any existing data

    # Create metadata table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lme_instances (
            question_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            question_type TEXT,
            expected_fact_ids TEXT
        )
    """)
    conn.execute("DELETE FROM lme_instances")  # clear any existing metadata

    # Create entity tables (for entity-indexed retrieval)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            entity_type TEXT DEFAULT 'unknown',
            aliases     TEXT DEFAULT '',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("DELETE FROM entities")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_entities (
            fact_id   INTEGER REFERENCES facts(fact_id),
            entity_id INTEGER REFERENCES entities(entity_id),
            PRIMARY KEY (fact_id, entity_id)
        )
    """)
    conn.execute("DELETE FROM fact_entities")

    total_turns = 0
    total_answers = 0
    total_sessions = 0
    instance_count = 0

    # Batch entity indexer
    indexer = EntityIndexer(conn)

    for inst in instances:
        qtype = inst.get("question_type", "unknown")
        question_id = inst.get("question_id", "unknown")
        question = inst.get("question", "")
        sessions = inst.get("haystack_sessions", [])

        # Find answer turn content and their fact_ids
        answer_contents = []
        for session in sessions:
            total_sessions += 1
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                content = turn.get("content", "").strip()
                if not content:
                    continue

                role = turn.get("role", "user")
                is_answer = turn.get("has_answer", False)

                if role == "user":
                    fact_content = f"[user] {content}"
                else:
                    fact_content = f"[assistant] {content}"

                tags = f"longmemeval,{qtype},{role}"
                if is_answer:
                    tags += ",has_answer"

                trust = 0.8 if is_answer else 0.5

                cur = conn.execute(
                    "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, 'longmemeval', ?, ?)",
                    (fact_content, tags, trust),
                )
                fact_id = cur.lastrowid
                total_turns += 1

                # Extract and link entities for this fact
                indexer.extract_and_resolve(fact_content, fact_id)

                if is_answer:
                    total_answers += 1
                    answer_contents.append(fact_content)

        # Store instance metadata with expected fact IDs
        # We'll populate expected_fact_ids after committing facts
        conn.execute(
            "INSERT OR REPLACE INTO lme_instances (question_id, question, question_type, expected_fact_ids) VALUES (?, ?, ?, ?)",
            (question_id, question, qtype, json.dumps(answer_contents)),  # store as content for now
        )
        instance_count += 1

    indexer.flush()
    conn.commit()

    # Now populate expected_fact_ids with actual fact_ids
    # Build content → fact_id mapping
    content_to_id = {}
    for row in conn.execute("SELECT fact_id, content FROM facts"):
        content_to_id[row[1]] = row[0]

    # Update lme_instances with real fact IDs
    for row in conn.execute("SELECT question_id, expected_fact_ids FROM lme_instances"):
        answer_contents = json.loads(row[1])
        expected_ids = [content_to_id.get(c) for c in answer_contents if c in content_to_id]
        conn.execute(
            "UPDATE lme_instances SET expected_fact_ids = ? WHERE question_id = ?",
            (json.dumps(expected_ids), row[0]),
        )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    return {
        "total_turns": total_turns,
        "answer_turns": total_answers,
        "total_sessions": total_sessions,
        "total_instances": len(instances),
        "stored_facts": count,
    }


def print_summary(stats: dict):
    """Print ingestion summary."""
    print("\n" + "=" * 60)
    print("LONGMEMEVAL BENCHMARK DB — INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Instances loaded:   {stats['total_instances']}")
    print(f"  Total turns:        {stats['total_turns']}")
    print(f"  Answer turns:       {stats['answer_turns']}")
    print(f"  Total sessions:     {stats['total_sessions']}")
    print(f"  Facts stored:       {stats['stored_facts']}")
    print(f"  Answer density:     {stats['answer_turns']/stats['total_turns']*100:.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load LongMemEval into benchmark SQLite DB")
    parser.add_argument(
        "--dataset",
        choices=["s", "m", "oracle"],
        default="oracle",
        help="Dataset version: s (40 sessions), m (500 sessions), oracle (evidence only)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_DB),
        help="Output database path",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of instances to load (default: all)",
    )
    args = parser.parse_args()

    db_path = Path(args.output)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating benchmark DB at: {db_path}")
    conn = create_db(db_path)

    print(f"Loading dataset: longmemeval_{args.dataset}.json")
    data = download_dataset(f"longmemeval_{args.dataset}.json")

    stats = ingest_turns(conn, data, subset=args.subset)
    print_summary(stats)
    conn.close()

    print(f"\nReady to run benchmark:")
    print(f"  python {Path(__file__).name.split('.')[0]}.py --db {db_path}")
