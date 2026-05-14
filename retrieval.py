"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import hrr
except ImportError:
    import hrr  # type: ignore[no-redef]

# RRF constant — exposed for testing
_RRF_C = 60

# Pre-compiled regex for Chinese character detection (hot path optimization)
# CJK Unified Ideographs block: U+4E00–U+9FFF
_RE_CHINESE = re.compile(r'[\u4e00-\u9fff]')

# Cached jieba module to avoid repeated import attempts
_JIEBA_MODULE = None
_JIEBA_IMPORT_FAILED = False


def _get_jieba():
    """Get jieba module with caching. Returns None if not available."""
    global _JIEBA_MODULE, _JIEBA_IMPORT_FAILED
    if _JIEBA_IMPORT_FAILED:
        return None
    if _JIEBA_MODULE is not None:
        return _JIEBA_MODULE
    try:
        import jieba
        _JIEBA_MODULE = jieba
        return jieba
    except ImportError:
        _JIEBA_IMPORT_FAILED = True
        return None


# Default signal weights for weighted RRF
# Optimized via grid search on personal memory benchmark (v0.4.1)
# FTS is down-weighted relative to equal weighting to let vec/Jaccard help
# where FTS is ambiguous; jaccard is up-weighted for semantic overlap
_RRF_WEIGHTS = {
    "fts": 0.5,       # FTS5 gets 0.5x (down-weighted — IDF boost already amplifies)
    "vec": 0.75,      # Vec gets 0.75x (essential for multi-hop)
    "jaccard": 0.75,  # Jaccard gets 0.75x (up-weighted for semantic overlap)
    "hrr": 0.5,       # HRR gets 0.5x
    "access": 0.5,    # Access count gets 0.5x
}


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Compute IDF weights from FTS5 metadata (cached per-retriever)
        self._idf_cache: dict[str, float] | None = None
        self._idf_computed = False
        try:
            self._compute_idf()
        except Exception:
            # FTS5 metadata may not exist (e.g., empty DB) — IDF will be 1.0
            self._idf_computed = False

    def _compute_idf(self) -> None:
        """Compute IDF weights by querying FTS5 document frequency per term.

        Since FTS5's _meta table isn't available in all SQLite versions,
        we compute IDF by counting how many facts contain each term
        (document frequency) via FTS5 MATCH queries.

        Caches results in self._idf_cache for reuse across queries.
        """
        conn = self.store._conn
        total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        if total == 0:
            self._idf_cache = {}
            self._idf_computed = True
            return

        # Get all unique terms from the FTS5 index by querying _data table
        # FTS5 stores terms in a flat index; we extract unique terms from it
        try:
            # Try the internal table approach first (SQLite 3.35+)
            rows = conn.execute(
                "SELECT term FROM facts_fts('internal')"
            ).fetchall()
            unique_terms = set()
            for row in rows:
                # Internal table returns content bytes; decode and tokenize
                content = row[0] if isinstance(row[0], str) else row[0].decode('utf-8', errors='replace')
                for token in content.split():
                    unique_terms.add(token.lower())
        except Exception:
            # Fallback: scan all fact content to extract terms
            fact_rows = conn.execute("SELECT content FROM facts").fetchall()
            unique_terms = set()
            for row in fact_rows:
                for token in self._tokenize(row["content"]):
                    unique_terms.add(token)

        # Compute IDF for each term
        self._idf_cache = {}
        for term in unique_terms:
            # Count documents containing this term
            try:
                df = conn.execute(
                    "SELECT COUNT(*) FROM facts_fts JOIN facts ON facts.fact_id = facts_fts.rowid WHERE facts_fts MATCH ?",
                    (term,),
                ).fetchone()[0]
            except Exception:
                df = 0

            if df > 0 and df < total:
                # Standard IDF: log(N / df), clamped to [0, 10]
                idf = math.log(total / df)
                self._idf_cache[term] = min(idf, 10.0)
            else:
                self._idf_cache[term] = 0.0
        self._idf_computed = True

    def _query_idf(self, query: str) -> float:
        """Compute average IDF for meaningful query terms.

        Returns a boost factor in [1.0, 1.5]. A query with only common terms
        (low IDF) returns ~1.0 (no boost). A query with rare terms (high IDF)
        returns a larger boost.

        This amplifies the FTS5 contribution for queries that contain rare,
        discriminative terms. Capped at 1.5 to prevent FTS5 from dominating
        vec in large corpora where even common terms have high IDF.
        """
        if not self._idf_computed or not self._idf_cache:
            return 1.0

        tokens = self._tokenize(query)
        idf_values = [
            self._idf_cache.get(t, 0.0)
            for t in tokens
            if t.lower() not in self._STOP_WORDS and len(t) >= 3
        ]

        if not idf_values:
            return 1.0

        avg_idf = sum(idf_values) / len(idf_values)
        # Dampen: 1 + log(1 + avg_idf) gives [1.0, ~2.3] for typical corpora
        # Cap at 1.5 to prevent FTS5 from overwhelming vec in large corpora
        return min(1.5, 1.0 + math.log(1.0 + avg_idf))

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        evidence_gap: bool = True,
        episode_id: int | None = None,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. IDF boost: Compute query-level IDF from FTS5 metadata, amplify FTS5 RRF
        3. Vec KNN: Get limit*3 candidates from dense vector search (sqlite-vec)
        4. RRF fusion: Combine FTS5 (IDF-weighted), Vec, Jaccard, HRR, access count
        5. Phrase proximity: Bonus for query terms appearing close together
        6. Trust weighting: final_score = relevance * trust_score
        7. Temporal decay (optional): decay = 0.5^(age_days / half_life)
        8. Evidence-gap analysis (optional): check query coverage

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        If evidence_gap is True, the first result includes an 'evidence_gap' dict.
        If episode_id is provided, results are filtered to that episode.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        # Uses Chinese LIKE fallback when FTS5 returns 0 for Chinese queries
        candidates = self._fts_candidates_with_chinese_fallback(
            query, category, min_trust, limit * 3, episode_id
        )

        # Stage 1b: Get vec candidates
        vec_candidates = self._vec_candidates(query, category, limit * 3, episode_id)

        # Stage 2: Union candidates (deduplicate by fact_id)
        seen_ids: set[int] = set()
        all_candidates: list[dict] = []
        for fact in candidates:
            if fact["fact_id"] not in seen_ids:
                seen_ids.add(fact["fact_id"])
                all_candidates.append(fact)
        for fact in vec_candidates:
            if fact["fact_id"] not in seen_ids:
                seen_ids.add(fact["fact_id"])
                all_candidates.append(fact)

        if not all_candidates:
            return []

        # Build rank maps: fact_id -> rank for each source
        fts_rank_map: dict[int, int] = {}
        for rank, fact in enumerate(candidates, start=1):
            fts_rank_map[fact["fact_id"]] = rank

        vec_rank_map: dict[int, int] = {}
        for rank, fact in enumerate(vec_candidates, start=1):
            vec_rank_map[fact["fact_id"]] = rank

        # Stage 3: Compute per-candidate scores using RRF fusion + proximity
        query_tokens = self._tokenize(query)
        scored = []

        # Extract meaningful query terms for proximity scoring
        proximity_terms = [t for t in query_tokens
                          if t.lower() not in self._STOP_WORDS and len(t) >= 3]

        # Compute IDF boost for this query (amplifies rare terms)
        idf_boost = self._query_idf(query)

        for fact in all_candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            # Jaccard similarity
            jaccard = self._jaccard_similarity(query_tokens, all_tokens)

            # HRR similarity
            if hrr._HAS_NUMPY and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
            else:
                hrr_sim = 0.5  # neutral

            # RRF fusion: combine ranks from multiple sources with weights
            # Weighted: sum(weight_i * 1/(rank_i + C)) / sum(weight_i)
            # This prevents noisy signals from diluting strong signals
            rrf_score = 0.0
            weight_sum = 0.0

            # FTS5 rank contribution — weighted by query-level IDF
            # Rare query terms amplify the FTS5 signal; common terms don't
            if fact["fact_id"] in fts_rank_map:
                fts_term = 1.0 / (fts_rank_map[fact["fact_id"]] + _RRF_C)
                rrf_score += fts_term * idf_boost * _RRF_WEIGHTS["fts"]
                weight_sum += _RRF_WEIGHTS["fts"]

            # Vec rank contribution
            if fact["fact_id"] in vec_rank_map:
                rrf_score += 1.0 / (vec_rank_map[fact["fact_id"]] + _RRF_C) * _RRF_WEIGHTS["vec"]
                weight_sum += _RRF_WEIGHTS["vec"]

            # Jaccard rank contribution (compute rank from similarity)
            # Higher jaccard = lower rank number
            jaccard_rank = max(1, int((1.0 - jaccard) * len(all_candidates)) + 1)
            rrf_score += 1.0 / (jaccard_rank + _RRF_C) * _RRF_WEIGHTS["jaccard"]
            weight_sum += _RRF_WEIGHTS["jaccard"]

            # HRR rank contribution
            hrr_rank = max(1, int((1.0 - hrr_sim) * len(all_candidates)) + 1)
            rrf_score += 1.0 / (hrr_rank + _RRF_C) * _RRF_WEIGHTS["hrr"]
            weight_sum += _RRF_WEIGHTS["hrr"]

            # Access count boost: retrieval_count + helpful_count as 4th signal
            retrieval_count = fact.get("retrieval_count", 0) or 0
            helpful_count = fact.get("helpful_count", 0) or 0
            _MAX_ACCESS = 10  # cap for normalization
            combined = retrieval_count * 0.3 + helpful_count * 0.7
            access_score = min(1.0, combined / _MAX_ACCESS)
            access_rank = max(1, int((1.0 - access_score) * len(all_candidates)) + 1)
            rrf_score += 1.0 / (access_rank + _RRF_C) * _RRF_WEIGHTS["access"]
            weight_sum += _RRF_WEIGHTS["access"]

            # Normalize by sum of weights (not count of sources)
            if weight_sum > 0:
                rrf_score /= weight_sum

            # Phrase proximity bonus: closer query terms in content = higher score
            proximity_score = 0.0
            if proximity_terms:
                proximity_score = self._phrase_proximity(query, fact["content"], proximity_terms)

            # Combine RRF + proximity
            score = rrf_score * fact["trust_score"] + proximity_score * 0.002

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
            fact.pop("vec_distance", None)

        # Stage 5: Evidence-gap analysis
        if evidence_gap and results:
            gap = self._analyze_evidence_gap(query, results)
            results[0]["evidence_gap"] = gap

        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query using HRR algebra.

        Unbinds entity from memory bank to extract associated content.
        This is NOT keyword search — it uses algebraic structure to find facts
        where the entity plays a structural role.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            # Fallback to keyword search on entity name
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as role-bound vector
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        # Try category-specific bank first, then all facts
        if category:
            bank_name = f"cat:{category}"
            bank_row = conn.execute(
                "SELECT vector FROM memory_banks WHERE bank_name = ?",
                (bank_name,),
            ).fetchone()
            if bank_row:
                bank_vec = hrr.bytes_to_phases(bank_row["vector"])
                extracted = hrr.unbind(bank_vec, probe_key)
                # Use extracted signal to score individual facts
                return self._score_facts_by_vector(
                    extracted, category=category, limit=limit
                )

        # Score against individual fact vectors directly
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            # Final fallback: keyword search
            return self.search(entity, category=category, limit=limit)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            # Unbind probe key from fact to see if entity is structurally present
            residual = hrr.unbind(fact_vec, probe_key)
            # Compare residual against content signal
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
            content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
            sim = hrr.similarity(residual, content_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as a bare atom (not role-bound — we want ANY structural match)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Score each fact by how much the entity's atom appears in its vector
        # This catches both role-bound entity matches AND content word matches
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact
            role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

            entity_role_sim = hrr.similarity(residual, role_entity)
            content_role_sim = hrr.similarity(residual, role_content)
            # Take the max — entity could appear in either role
            best_sim = max(entity_role_sim, content_role_sim)

            fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — vector-space JOIN.

        Given multiple entities, algebraically intersects their structural
        connections to find facts related to ALL of them simultaneously.
        This is compositional reasoning that no embedding DB can do.

        Example: reason(["peppi", "backend"]) finds facts where peppi AND
        backend both play structural roles — without keyword matching.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY or not entities:
            # Fallback: search with all entities as keywords
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        conn = self.store._conn
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)

        # For each entity, compute what the bank "remembers" about it
        # by unbinding entity+role from each fact vector
        entity_residuals = []
        for entity in entities:
            entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
            probe_key = hrr.bind(entity_vec, role_entity)
            entity_residuals.append(probe_key)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        # Score each fact by how much EACH entity is structurally present.
        # A fact scores high only if ALL entities have structural presence
        # (AND semantics via min, vs OR which would use mean/max).
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            entity_scores = []
            for probe_key in entity_residuals:
                residual = hrr.unbind(fact_vec, probe_key)
                sim = hrr.similarity(residual, role_content)
                entity_scores.append(sim)

            min_sim = min(entity_scores)
            fact["score"] = (min_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND f.category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {where}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact
        fact_entities: dict[int, set[str]] = {}
        for row in rows:
            fid = row["fact_id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fid,),
            ).fetchall()
            fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"])
                v2 = hrr.bytes_to_phases(f2["hrr_vector"])
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _vec_candidates(
        self,
        query: str,
        category: str | None,
        limit: int,
        episode_id: int | None = None,
    ) -> list[dict]:
        """Get candidates from dense vector search (sqlite-vec KNN).

        Returns list of fact dicts with 'vec_rank' field.
        Returns empty list if sqlite-vec or embedding model unavailable.
        """
        try:
            import sqlite_vec
        except ImportError:
            return []

        try:
            from . import embedding as emb_mod
        except ImportError:
            try:
                import embedding as emb_mod  # type: ignore[no-redef]
            except ImportError:
                return []

        if not emb_mod.is_available():
            return []

        # Encode query with "query:" prefix required by e5 models
        query_vec = emb_mod.encode(query, prefix="query")
        if query_vec is None:
            return []

        conn = self.store._conn
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            return []

        # Check if facts_vec table exists
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "facts_vec" not in tables:
            return []

        # Build filters
        category_clause = ""
        episode_clause = ""
        params = [query_vec.tobytes(), limit * 3]
        if category is not None:
            category_clause = "AND f.category = ?"
            params.append(category)
        if episode_id is not None:
            episode_clause = "AND ef.episode_id = ?"
            params.append(episode_id)

        # KNN query joining facts_vec with facts table
        episode_join = "JOIN episode_facts ef ON ef.fact_id = f.fact_id" if episode_id is not None else ""
        sql = f"""
            SELECT f.fact_id, f.content, f.category, f.tags,
                   f.trust_score, f.retrieval_count, f.helpful_count,
                   f.created_at, f.updated_at, f.hrr_vector,
                   fv.distance as vec_distance
            FROM facts_vec fv
            JOIN facts f ON f.fact_id = fv.rowid
            {episode_join}
            WHERE fv.embedding MATCH ?
              AND k = ?
              AND f.trust_score >= 0.3
              {category_clause}
              {episode_clause}
            ORDER BY fv.distance
        """

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []

        results = []
        for rank, row in enumerate(rows, start=1):
            fact = dict(row)
            fact["vec_rank"] = rank
            results.append(fact)

        return results

    def _prepare_fts5_query(self, query: str) -> str:
        """Preprocess a natural-language query for FTS5 MATCH.

        FTS5 MATCH uses implicit AND between terms, so long queries
        like "What is the total cost of the new food bowl" return 0
        results because no single fact contains all those words.

        This method:
        1. Extracts multi-word capitalized phrases (kept as phrase matches)
        2. Strips stop words, keeps content words >= 3 chars
        3. Joins content words with OR for recall
        4. Falls back to the original query if processing yields nothing
        """
        # Extract capitalized multi-word phrases (e.g. "Samsung Galaxy S22")
        phrases = []
        for m in self._RE_CAPITALIZED.finditer(query):
            phrase = m.group(1).strip()
            if len(phrase.split()) >= 2:
                phrases.append(f'"{phrase}"')

        # Content words: strip stop words, keep >= 3 chars
        tokens = self._tokenize(query)
        content = [t for t in tokens
                   if t.lower() not in self._STOP_WORDS and len(t) >= 3]

        # Build FTS5 query: phrases OR content words
        parts = phrases + content
        if parts:
            return " OR ".join(parts)

        # Fallback: return original query if processing yielded nothing
        return query

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
        episode_id: int | None = None,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Preprocess query for FTS5: strip stop words, use OR for recall
        fts5_query = self._prepare_fts5_query(query)

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        params.append(fts5_query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        if episode_id is not None:
            where_clauses.append("ef.episode_id = ?")
            params.append(episode_id)

        where_sql = " AND ".join(where_clauses)

        episode_join = "JOIN episode_facts ef ON ef.fact_id = f.fact_id" if episode_id is not None else ""
        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            {episode_join}
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: rank is negative, lower = better
        # Convert to positive score in [0, 1] range
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    # ------------------------------------------------------------------
    # Chinese language support
    # ------------------------------------------------------------------

    @staticmethod
    def _has_chinese(text: str) -> bool:
        """Check if text contains Chinese characters (CJK Unified Ideographs).

        Uses pre-compiled regex for hot-path efficiency (~0.01ms).
        Returns False for empty/None input.
        """
        if not text:
            return False
        return bool(_RE_CHINESE.search(text))

    @staticmethod
    def _chinese_tokenize(text: str) -> list[str]:
        """Tokenize Chinese text using jieba segmentation.

        Falls back to character-level tokenization if jieba is unavailable.
        Returns a list of meaningful tokens (filtered for length >= 2).
        """
        jieba_mod = _get_jieba()
        if jieba_mod:
            tokens = []
            for word in jieba_mod.cut(text):
                # Filter: keep words with >= 2 chars (skip single punctuation)
                if len(word) >= 2:
                    tokens.append(word)
            return tokens

        # Fallback: character-level tokenization
        # Extract Chinese character sequences and bigrams
        chars = _RE_CHINESE.findall(text)
        tokens = list(set(chars))  # unique characters
        # Add bigrams for phrase matching
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])
        return tokens

    @staticmethod
    def _escape_like(text: str) -> str:
        """Escape special LIKE characters in a string for safe SQL LIKE usage."""
        return text.replace('%', r'\%').replace('_', r'\_')

    def _chinese_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
        episode_id: int | None = None,
    ) -> list[dict]:
        """Get Chinese LIKE-based candidates from the store.

        FTS5 unicode61 tokenizer tokenizes Chinese into single characters,
        making keyword search completely ineffective. This method uses
        LIKE-based pattern matching with jieba tokenization instead.

        Returns list of fact dicts with 'fts_rank' field (for RRF fusion).
        """
        conn = self.store._conn

        # Tokenize query using jieba
        tokens = self._chinese_tokenize(query)
        if not tokens:
            return []

        # Build LIKE query with AND semantics (must match all keywords)
        like_clauses = []
        params: list = []
        for token in tokens:
            like_clauses.append("(f.content LIKE ? OR f.tags LIKE ?)")
            params.append(f"%{self._escape_like(token)}%")
            params.append(f"%{self._escape_like(token)}%")

        where_clauses = like_clauses
        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        if episode_id is not None:
            where_clauses.append("ef.episode_id = ?")
            params.append(episode_id)

        where_sql = " AND ".join(where_clauses)
        episode_join = "JOIN episode_facts ef ON ef.fact_id = f.fact_id" if episode_id is not None else ""

        sql = f"""
            SELECT f.*, 0 as fts_rank_raw
            FROM facts f
            {episode_join}
            WHERE {where_sql}
            LIMIT ?
        """
        params.append(limit * 3)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        results = []
        for rank, row in enumerate(rows, start=1):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = rank  # Use rank order directly for Chinese LIKE
            results.append(fact)

        return results

    def _fts_candidates_with_chinese_fallback(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
        episode_id: int | None = None,
    ) -> list[dict]:
        """Get FTS5 candidates with Chinese LIKE fallback.

        For Chinese queries, FTS5 unicode61 tokenizer produces single-character
        tokens that are ineffective for keyword search. This method:
        1. Tries FTS5 first (works for English, mixed queries)
        2. Falls back to LIKE-based search if FTS5 returns 0 results for Chinese
        3. Returns empty list for pure English (standard FTS5 path)

        Returns list of fact dicts with 'fts_rank' field (for RRF fusion).
        """
        has_chinese = self._has_chinese(query)

        if not has_chinese:
            # Pure English: use standard FTS5 (unchanged path)
            return self._fts_candidates(query, category, min_trust, limit, episode_id)

        # Has Chinese: try FTS5 first, fall back to LIKE
        fts_results = self._fts_candidates(query, category, min_trust, limit, episode_id)

        if fts_results:
            # FTS5 returned results — good enough (e.g., mixed EN+CN query)
            return fts_results

        # FTS5 returned 0 results — fall back to LIKE for Chinese
        return self._chinese_candidates(query, category, min_trust, limit, episode_id)

    # ------------------------------------------------------------------
    # Evidence-gap analysis
    # ------------------------------------------------------------------

    # Terms that carry little semantic weight for coverage checking
    _STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "need", "dare", "ought", "used", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how",
        "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "about", "what", "which",
        "that", "this", "these", "those", "it", "its", "i", "me",
        "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their", "also", "any",
        "much", "many", "get", "got", "like", "one", "two",
        "first", "new", "use", "used", "using", "make", "made",
        "way", "like", "well", "even", "still", "now", "back",
        "up", "down", "out", "go", "going", "come", "take",
        "see", "know", "think", "say", "said", "goes",
    }

    # Entity extraction regex (mirrors store._RE_CAPITALIZED to avoid circular imports)
    _RE_CAPITALIZED = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')

    def _extract_query_components(self, query: str) -> dict:
        """Decompose a query into coverable evidence components.

        Returns a dict with:
          - entities: list of extracted entity names
          - content_words: list of significant non-stopword tokens
          - phrases: list of multi-word capitalized phrases
        """
        # Extract entities using the store's method if available, else our own
        if hasattr(self.store, '_extract_entities'):
            entities = self.store._extract_entities(query)
        else:
            entities = []

        tokens = self._tokenize(query)
        content_words = [t for t in tokens
                        if t.lower() not in self._STOP_WORDS and len(t) >= 3]

        # Multi-word capitalized phrases
        phrases = []
        for m in self._RE_CAPITALIZED.finditer(query):
            phrase = m.group(1).strip()
            if len(phrase.split()) >= 2:
                phrases.append(phrase)

        return {
            "entities": entities,
            "content_words": content_words,
            "phrases": phrases,
        }

    def _fact_covers_component(self, fact_content: str, component: str) -> bool:
        """Check if a fact's content covers a query component.

        Uses token-level Jaccard similarity with a threshold of 0.15.
        For short components (single words), requires exact substring match
        (case-insensitive) to avoid false positives.
        """
        component_tokens = self._tokenize(component)
        fact_tokens = self._tokenize(fact_content)

        if not component_tokens:
            return False

        # Single-token components use substring matching to avoid
        # matching "gpu" against "computer" via shared stopwords
        if len(component_tokens) == 1:
            return component.lower() in fact_content.lower()

        jaccard = self._jaccard_similarity(component_tokens, fact_tokens)
        return jaccard >= 0.15

    def _analyze_evidence_gap(
        self,
        query: str,
        results: list[dict],
    ) -> dict:
        """Analyze whether retrieved facts cover the query's key components.

        Returns a dict with:
          - coverage_ratio: float 0.0-1.0 (fraction of components covered)
          - covered: list of component strings that were covered
          - uncovered: list of component strings that were NOT covered
          - components_total: int total number of components checked
          - needs_followup: bool True if coverage_ratio < 0.6
        """
        components = self._extract_query_components(query)

        # Flatten all components into a single deduplicated list
        all_components: list[str] = []
        seen: set[str] = set()

        for item in components["entities"]:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                all_components.append(item)

        for item in components["phrases"]:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                all_components.append(item)

        for item in components["content_words"]:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                all_components.append(item)

        if not all_components:
            return {
                "coverage_ratio": 1.0,
                "covered": [],
                "uncovered": [],
                "components_total": 0,
                "needs_followup": False,
            }

        # Check each component against all retrieved facts
        covered: list[str] = []
        uncovered: list[str] = []

        for component in all_components:
            is_covered = False
            for fact in results:
                if self._fact_covers_component(fact["content"], component):
                    is_covered = True
                    break
            if is_covered:
                covered.append(component)
            else:
                uncovered.append(component)

        coverage_ratio = len(covered) / len(all_components) if all_components else 1.0

        return {
            "coverage_ratio": round(coverage_ratio, 2),
            "covered": covered,
            "uncovered": uncovered,
            "components_total": len(all_components),
            "needs_followup": coverage_ratio < 0.6,
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _phrase_proximity(self, query: str, content: str, terms: list[str]) -> float:
        """Calculate phrase proximity score for query terms in content.

        Finds the physical span (distance between first and last occurrence)
        of query terms in the content. Closer terms = higher relevance.

        Returns a score in [0, 1] where 1.0 means all terms are adjacent.

        For a single term, returns 0.5 (neutral — proximity is undefined).
        """
        if not terms:
            return 0.0

        content_lower = content.lower()
        positions: list[int] = []

        for term in terms:
            # Find all occurrences of this term in content
            start = 0
            while True:
                idx = content_lower.find(term, start)
                if idx == -1:
                    break
                # Word boundary check: ensure term isn't part of a larger word
                if idx > 0 and content_lower[idx - 1].isalnum():
                    start = idx + 1
                    continue
                end = idx + len(term)
                if end < len(content_lower) and content_lower[end].isalnum():
                    start = idx + 1
                    continue
                positions.append(idx)
                start = idx + 1

        if not positions:
            return 0.0

        # If only one term found, neutral score
        if len(positions) <= 1:
            return 0.5

        # Calculate span: distance between first and last occurrence
        min_pos = min(positions)
        max_pos = max(positions)
        span = max_pos - min_pos

        # Normalize: smaller span = higher score
        # Use exponential decay: score = e^(-span / scale)
        # Scale = 50 chars means span of 50 → score ~0.37
        score = math.exp(-span / 30.0)

        # Boost if terms are actually adjacent (span < avg term length)
        avg_term_len = sum(len(t) for t in terms) / len(terms)
        if span < avg_term_len:
            score = min(1.0, score * 2.0)

        return score

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
