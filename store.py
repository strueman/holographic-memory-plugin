"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

try:
    import sqlite_vec
    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

# Embedding config
EMBEDDING_DIM = 384

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_links (
    fact_id_a  INTEGER REFERENCES facts(fact_id),
    fact_id_b  INTEGER REFERENCES facts(fact_id),
    link_type  TEXT NOT NULL,
    strength   REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id_a, fact_id_b, link_type)
);

CREATE INDEX IF NOT EXISTS idx_fact_links_a ON fact_links(fact_id_a);
CREATE INDEX IF NOT EXISTS idx_fact_links_b ON fact_links(fact_id_b);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    start_time  TIMESTAMP NOT NULL,
    end_time    TIMESTAMP NOT NULL,
    fact_count  INTEGER DEFAULT 0,
    summary     TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS episode_facts (
    episode_id INTEGER REFERENCES episodes(episode_id),
    fact_id    INTEGER REFERENCES facts(fact_id),
    PRIMARY KEY (episode_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_start ON episodes(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_episode_facts_fact ON episode_facts(fact_id);
"""

# Structural linking thresholds
_HRR_LINK_THRESHOLD    = 0.65  # Normalized HRR sim; raw ~0.3 (above random ~0.0)
_ENTITY_LINK_THRESHOLD = 0.25  # Jaccard overlap on entity sets
_EMB_LINK_THRESHOLD    = 0.75  # Cosine similarity for embeddings
_LINK_COMPARE_LIMIT    = 100   # Max facts to compare against per new fact
_LINKS_PER_FACT_MAX    = 20    # Max outgoing links per fact

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Temporal segmentation (episodes)
_EPISODE_GAP_MINUTES = 60   # gap threshold for episode boundaries
_EPISODE_MAX_SIZE    = 100  # cap episode size to prevent massive clusters

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")

        # Migrate: create vec0 virtual table for dense vector search (sqlite-vec)
        self._init_vec0()

        self._conn.commit()

    def _init_vec0(self) -> None:
        """Create vec0 virtual table for dense vector search if sqlite-vec is available."""
        if not _HAS_SQLITE_VEC:
            return
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            # Check if vec0 table already exists
            tables = [r["name"] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "facts_vec" not in tables:
                self._conn.execute(f"""
                    CREATE VIRTUAL TABLE facts_vec USING vec0(
                        embedding float[{EMBEDDING_DIM}]
                    )
                """)
                self._conn.commit()
        except Exception:
            # sqlite-vec unavailable or failed — vector search will be disabled
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — return existing id
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"])

            # Entity extraction and linking
            for name in self._extract_entities(content):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            # Compute dense embedding for vector search
            self._compute_embedding(fact_id, content)
            # Structural linking: compare against recent facts
            self._create_links_for_fact(fact_id)
            self._rebuild_bank(category)

            # Auto-segment into episode
            self._auto_segment_fact(fact_id)

            return fact_id

    def backfill_embeddings(self) -> int:
        """Compute embeddings for all facts that don't have one yet.

        Returns the number of facts that were backfilled.
        """
        with self._lock:
            try:
                from . import embedding as emb_mod
            except ImportError:
                import embedding as emb_mod  # type: ignore[no-redef]

            if not emb_mod.is_available():
                return 0

            # Get facts without embeddings
            rows = self._conn.execute("""
                SELECT f.fact_id, f.content FROM facts f
                LEFT JOIN facts_vec fv ON fv.rowid = f.fact_id
                WHERE fv.rowid IS NULL
            """).fetchall()

            if not rows:
                return 0

            count = 0
            for row in rows:
                try:
                    vec = emb_mod.encode(row["content"])
                    if vec is not None:
                        self._conn.execute(
                            "INSERT INTO facts_vec(rowid, embedding) VALUES (?, ?)",
                            (row["fact_id"], vec.tobytes()),
                        )
                        count += 1
                except Exception:
                    continue
            self._conn.commit()
            return count

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            params: list = [query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()

            # If content changed, re-extract entities
            if content is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
            # Recompute embedding if content changed
            if content is not None:
                self._compute_embedding(fact_id, content)
            # Rebuild bank for relevant category
            cat = category or self._conn.execute(
                "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()["category"]
            self._rebuild_bank(cat)

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact, its entity links, structural links, and episode links. Returns True if the row existed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            self._conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute(
                "DELETE FROM fact_links WHERE fact_id_a = ? OR fact_id_b = ?", (fact_id, fact_id)
            )
            # Clean up episode links — save episode IDs before deleting
            ep_rows = self._conn.execute(
                "SELECT episode_id FROM episode_facts WHERE fact_id = ?", (fact_id,)
            ).fetchall()
            episode_ids = [r[0] for r in ep_rows]

            self._conn.execute(
                "DELETE FROM episode_facts WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))

            # Update fact_count for affected episodes
            for eid in episode_ids:
                self._conn.execute(
                    """
                    UPDATE episodes SET fact_count = (
                        SELECT COUNT(*) FROM episode_facts WHERE episode_id = ?
                    )
                    WHERE episode_id = ?
                    """,
                    (eid, eid),
                )
            self._conn.commit()
            self._rebuild_bank(row["category"])
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._conn.commit()

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Structural linking
    # ------------------------------------------------------------------

    def _create_links_for_fact(self, fact_id: int) -> None:
        """Compare a new fact against recent existing facts and create links.

        Three link signals:
        1. HRR vector similarity (structural/semantic overlap)
        2. Entity set overlap (shared entities)
        3. Dense embedding similarity (semantic similarity)

        Only creates links when similarity exceeds configured thresholds.
        Caps comparisons at _LINK_COMPARE_LIMIT facts and outgoing links
        at _LINKS_PER_FACT_MAX to avoid O(n) blowup.
        """
        with self._lock:
            # Check current link count for this fact
            existing = self._conn.execute(
                "SELECT COUNT(*) FROM fact_links WHERE fact_id_a = ?", (fact_id,)
            ).fetchone()[0]
            remaining = _LINKS_PER_FACT_MAX - existing
            if remaining <= 0:
                return

            # Get recent facts with vectors (most recent first, capped)
            rows = self._conn.execute(
                """
                SELECT fact_id, content, hrr_vector FROM facts
                WHERE fact_id != ? AND hrr_vector IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (fact_id, _LINK_COMPARE_LIMIT),
            ).fetchall()

            if not rows:
                return

            # Get the new fact's HRR vector
            new_row = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if not new_row or not new_row["hrr_vector"]:
                return
            new_vec = hrr.bytes_to_phases(new_row["hrr_vector"])

            # Get the new fact's entities
            new_entities = self._get_fact_entities(fact_id)

            # Get the new fact's embedding
            new_emb = self._get_fact_embedding(fact_id)

            links_to_create: list[tuple[int, int, str, float]] = []

            for row in rows:
                if remaining <= 0:
                    break

                other_id = row["fact_id"]
                other_vec = hrr.bytes_to_phases(row["hrr_vector"])

                # Signal 1: HRR similarity
                if self._hrr_available and remaining > 0:
                    sim = hrr.similarity(new_vec, other_vec)
                    # Normalize from [-1, 1] to [0, 1]
                    norm_sim = (sim + 1.0) / 2.0
                    if norm_sim > _HRR_LINK_THRESHOLD:
                        links_to_create.append(
                            (fact_id, other_id, "hrr_similarity", round(norm_sim, 4))
                        )
                        remaining -= 1

                # Signal 2: Entity overlap (Jaccard)
                if remaining > 0:
                    other_entities = self._get_fact_entities(other_id)
                    if new_entities and other_entities:
                        union = len(new_entities | other_entities)
                        if union > 0:
                            entity_jaccard = len(new_entities & other_entities) / union
                            if entity_jaccard > _ENTITY_LINK_THRESHOLD:
                                links_to_create.append(
                                    (fact_id, other_id, "entity_overlap", round(entity_jaccard, 4))
                                )
                                remaining -= 1

                # Signal 3: Embedding similarity
                if remaining > 0 and new_emb is not None:
                    other_emb = self._get_fact_embedding(other_id)
                    if other_emb is not None:
                        emb_sim = self._cosine_similarity(new_emb, other_emb)
                        if emb_sim > _EMB_LINK_THRESHOLD:
                            links_to_create.append(
                                (fact_id, other_id, "embedding_similarity", round(emb_sim, 4))
                            )
                            remaining -= 1

            # Insert links (INSERT OR IGNORE handles duplicates)
            if links_to_create:
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength)
                    VALUES (?, ?, ?, ?)
                    """,
                    links_to_create,
                )
                self._conn.commit()

    def _get_fact_entities(self, fact_id: int) -> set[str]:
        """Get entity names linked to a fact."""
        rows = self._conn.execute(
            """
            SELECT e.name FROM entities e
            JOIN fact_entities fe ON fe.entity_id = e.entity_id
            WHERE fe.fact_id = ?
            """,
            (fact_id,),
        ).fetchall()
        return {row["name"].lower() for row in rows}

    def _get_fact_embedding(self, fact_id: int) -> "bytes | None":
        """Get the dense embedding for a fact. Returns raw bytes or None."""
        if not _HAS_SQLITE_VEC:
            return None
        try:
            row = self._conn.execute(
                "SELECT embedding FROM facts_vec WHERE rowid = ?", (fact_id,)
            ).fetchone()
            return row["embedding"] if row else None
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: bytes, b: bytes) -> float:
        """Compute cosine similarity between two byte-encoded float32 vectors."""
        # Decode bytes to lists of floats (avoid numpy dependency here)
        import struct
        fmt = f"<{len(a) // 4}f"
        vec_a = struct.unpack(fmt, a)
        vec_b = struct.unpack(fmt, b)

        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = sum(x * x for x in vec_a) ** 0.5
        norm_b = sum(x * x for x in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_linked_facts(
        self,
        fact_id: int,
        link_type: str | None = None,
        min_strength: float = 0.0,
        limit: int = 10,
    ) -> list[dict]:
        """Retrieve facts structurally linked to a given fact.

        Returns a list of dicts with fact data + 'links' field containing
        the link details (link_type, strength).

        Arguments:
            fact_id: The fact to find links for.
            link_type: Filter by link type ('hrr_similarity', 'entity_overlap',
                       'embedding_similarity'). None = all types.
            min_strength: Minimum link strength to include.
            limit: Maximum number of linked facts to return.
        """
        with self._lock:
            # Verify fact exists
            row = self._conn.execute(
                "SELECT fact_id FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return []

            # Build query - look for links in both directions (A->B or B->A)
            where_parts: list[str] = [
                "fl.strength >= ?",
            ]
            params: list = [min_strength]

            if link_type is not None:
                where_parts.append("fl.link_type = ?")
                params.append(link_type)

            sql = f"""
                SELECT DISTINCT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at,
                       fl.link_type, fl.strength
                FROM fact_links fl
                JOIN facts f ON (
                    (fl.fact_id_a = ? AND f.fact_id = fl.fact_id_b) OR
                    (fl.fact_id_b = ? AND f.fact_id = fl.fact_id_a)
                )
                WHERE {" AND ".join(where_parts)}
                ORDER BY fl.strength DESC
                LIMIT ?
            """
            # Params: fact_id (JOIN a), fact_id (JOIN b), min_strength, [link_type], limit
            params.insert(0, fact_id)
            params.insert(1, fact_id)
            params.append(limit)

            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Temporal segmentation (episodes)
    # ------------------------------------------------------------------

    def segment_episodes(self, gap_minutes: int | None = None) -> dict:
        """Segment facts into episodes based on temporal gaps.

        Facts whose created_at timestamps are within `gap_minutes` of each other
        belong to the same episode. When the gap exceeds the threshold, a new
        episode starts.

        Only segments facts that are not yet assigned to an episode.
        Idempotent — calling twice produces the same result.

        Returns stats: {episodes_created, facts_segmented, gaps_skipped}
        """
        with self._lock:
            gap = gap_minutes if gap_minutes is not None else _EPISODE_GAP_MINUTES
            gap_seconds = gap * 60

            # Get unassigned facts ordered by created_at
            unassigned = self._conn.execute(
                """
                SELECT f.fact_id, f.created_at
                FROM facts f
                WHERE f.fact_id NOT IN (SELECT fact_id FROM episode_facts)
                ORDER BY f.created_at
                """
            ).fetchall()

            if not unassigned:
                return {"episodes_created": 0, "facts_segmented": 0, "gaps_skipped": 0}

            episodes_created = 0
            gaps_skipped = 0

            current_episode_id = None
            current_facts: list[tuple[int, str]] = []
            current_end: str | None = None

            def _flush_episode() -> None:
                nonlocal current_episode_id, episodes_created, current_facts
                if not current_facts:
                    return
                # Create episode
                title = self._generate_episode_title(
                    [fid for fid, _ in current_facts]
                )
                summary = self._generate_episode_summary(
                    [fid for fid, _ in current_facts]
                )
                cur = self._conn.execute(
                    """
                    INSERT INTO episodes (title, start_time, end_time, fact_count, summary)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (title, current_facts[0][1], current_end, len(current_facts), summary),
                )
                current_episode_id = cur.lastrowid  # type: ignore[assignment]
                episodes_created += 1

                # Link facts to episode
                for fid, _ in current_facts:
                    self._conn.execute(
                        "INSERT INTO episode_facts (episode_id, fact_id) VALUES (?, ?)",
                        (current_episode_id, fid),
                    )
                self._conn.commit()
                current_facts = []

            for row in unassigned:
                fid, ts = row[0], row[1]

                if not current_facts:
                    # First fact starts a new episode
                    current_facts.append((fid, ts))
                    current_end = ts
                    continue

                # Check gap from last fact in current episode
                try:
                    last_ts = datetime.fromisoformat(current_end.replace("Z", "+00:00"))  # type: ignore[union-attr]
                    this_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    gap_sec = abs((this_ts - last_ts).total_seconds())
                except (ValueError, TypeError):
                    gap_sec = gap_seconds  # on parse error, start new episode

                if gap_sec > gap_seconds or len(current_facts) >= _EPISODE_MAX_SIZE:
                    # Gap too large or episode full — flush and start new
                    _flush_episode()
                    if gap_sec > gap_seconds:
                        gaps_skipped += 1
                    current_facts.append((fid, ts))
                    current_end = ts
                else:
                    # Same episode
                    current_facts.append((fid, ts))
                    current_end = ts

            # Flush remaining
            _flush_episode()

            return {
                "episodes_created": episodes_created,
                "facts_segmented": self._conn.execute(
                    "SELECT COUNT(DISTINCT fact_id) FROM episode_facts"
                ).fetchone()[0],
                "gaps_skipped": gaps_skipped,
            }

    def _auto_segment_fact(self, fact_id: int) -> None:
        """Assign a newly inserted fact to an existing episode or create one.

        Called from add_fact() after the fact is inserted.
        Checks if the fact falls within gap_minutes of the most recent episode's
        end_time. If so, appends to that episode. Otherwise creates a new one.
        """
        # Get the most recent episode
        latest = self._conn.execute(
            """
            SELECT episode_id, end_time
            FROM episodes
            ORDER BY end_time DESC
            LIMIT 1
            """
        ).fetchone()

        fact_row = self._conn.execute(
            "SELECT created_at FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if not fact_row:
            return
        fact_ts = fact_row[0]

        try:
            fact_dt = datetime.fromisoformat(str(fact_ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return

        if latest:
            try:
                end_dt = datetime.fromisoformat(str(latest[1]).replace("Z", "+00:00"))
                gap_sec = abs((fact_dt - end_dt).total_seconds())
            except (ValueError, TypeError):
                gap_sec = _EPISODE_GAP_MINUTES * 60 + 1

            if gap_sec <= _EPISODE_GAP_MINUTES * 60:
                # Append to existing episode
                self._conn.execute(
                    "INSERT INTO episode_facts (episode_id, fact_id) VALUES (?, ?)",
                    (latest[0], fact_id),
                )
                self._conn.execute(
                    """
                    UPDATE episodes
                    SET end_time = ?, fact_count = fact_count + 1
                    WHERE episode_id = ?
                    """,
                    (fact_ts, latest[0]),
                )
                # Refresh title + summary for updated episode
                title = self._generate_episode_title([fact_id])
                self._conn.execute(
                    "UPDATE episodes SET title = ? WHERE episode_id = ?",
                    (title, latest[0]),
                )
                self._conn.commit()
                return

        # Create new episode for this fact
        title = self._generate_episode_title([fact_id])
        summary = self._generate_episode_summary([fact_id])
        self._conn.execute(
            """
            INSERT INTO episodes (title, start_time, end_time, fact_count, summary)
            VALUES (?, ?, ?, 1, ?)
            """,
            (title, fact_ts, fact_ts, summary),
        )
        new_ep = self._conn.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]
        self._conn.execute(
            "INSERT INTO episode_facts (episode_id, fact_id) VALUES (?, ?)",
            (new_ep, fact_id),
        )
        self._conn.commit()

    def _next_episode_num(self) -> int:
        """Return the next sequential episode number."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM episodes"
        ).fetchone()
        return (row[0] if row else 0) + 1

    def _generate_episode_title(self, fact_ids: list[int]) -> str:
        """Generate a title for an episode from its facts' dominant entities."""
        if not fact_ids:
            return "Untitled Episode"

        placeholders = ",".join("?" * len(fact_ids))
        rows = self._conn.execute(
            f"""
            SELECT e.name, COUNT(*) as cnt
            FROM entities e
            JOIN fact_entities fe ON fe.entity_id = e.entity_id
            WHERE fe.fact_id IN ({placeholders})
            GROUP BY e.name
            ORDER BY cnt DESC
            LIMIT 3
            """,
            fact_ids,
        ).fetchall()

        if rows:
            entities = [row[0] for row in rows]
            return " → ".join(entities)

        # Fallback: use first fact's category
        cat_row = self._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (fact_ids[0],)
        ).fetchone()
        return cat_row[0] if cat_row else "Untitled Episode"

    def _generate_episode_summary(self, fact_ids: list[int]) -> str:
        """Generate an algorithmic summary of an episode."""
        if not fact_ids:
            return ""

        placeholders = ",".join("?" * len(fact_ids))

        # Dominant category
        cat_row = self._conn.execute(
            f"""
            SELECT category, COUNT(*) as cnt
            FROM facts WHERE fact_id IN ({placeholders})
            GROUP BY category ORDER BY cnt DESC LIMIT 1
            """,
            fact_ids,
        ).fetchone()
        category = cat_row[0] if cat_row else "general"

        # All entities
        ent_rows = self._conn.execute(
            f"""
            SELECT DISTINCT e.name
            FROM entities e
            JOIN fact_entities fe ON fe.entity_id = e.entity_id
            WHERE fe.fact_id IN ({placeholders})
            ORDER BY e.name
            """,
            fact_ids,
        ).fetchall()
        entities = [row[0] for row in ent_rows]

        # Average trust
        trust_row = self._conn.execute(
            f"SELECT AVG(trust_score) FROM facts WHERE fact_id IN ({placeholders})",
            fact_ids,
        ).fetchone()
        avg_trust = trust_row[0] if trust_row else 0.5

        # Timestamps
        time_rows = self._conn.execute(
            f"SELECT MIN(created_at), MAX(created_at) FROM facts WHERE fact_id IN ({placeholders})",
            fact_ids,
        ).fetchone()

        parts = [f"[EPISODE] {len(fact_ids)} facts in '{category}'"]
        if entities:
            parts.append(f"Entities: {', '.join(entities)}")
        parts.append(f"Avg trust: {avg_trust:.2f}")
        if time_rows:
            parts.append(f"Span: {time_rows[0]} → {time_rows[1]}")

        return ". ".join(parts)

    def get_episode(self, episode_id: int) -> dict | None:
        """Get episode details including fact count and time range."""
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_episodes(self, limit: int = 50, order: str = "newest") -> list[dict]:
        """List episodes ordered by time.

        Args:
            limit: Maximum number of episodes to return.
            order: 'newest' (default) or 'oldest'.
        """
        direction = "DESC" if order == "newest" else "ASC"
        rows = self._conn.execute(
            f"SELECT * FROM episodes ORDER BY start_time {direction} LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_episode_facts(self, episode_id: int, limit: int = 50) -> list[dict]:
        """Get all facts belonging to an episode."""
        placeholders = ",".join("?" * 1)
        rows = self._conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags,
                   f.trust_score, f.retrieval_count, f.helpful_count,
                   f.created_at, f.updated_at
            FROM facts f
            JOIN episode_facts ef ON ef.fact_id = f.fact_id
            WHERE ef.episode_id = ?
            ORDER BY f.created_at
            LIMIT ?
            """,
            (episode_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if stripped and stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _compute_embedding(self, fact_id: int, content: str) -> None:
        """Compute and store dense embedding for a fact. No-op if unavailable."""
        with self._lock:
            try:
                from . import embedding as emb_mod
            except ImportError:
                import embedding as emb_mod  # type: ignore[no-redef]

            if not emb_mod.is_available():
                return

            vec = emb_mod.encode(content)
            if vec is None:
                return

            # Store as float32 bytes
            vec_bytes = vec.tobytes()
            self._conn.execute(
                "INSERT OR REPLACE INTO facts_vec(rowid, embedding) VALUES (?, ?)",
                (fact_id, vec_bytes),
            )
            self._conn.commit()

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"]) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
