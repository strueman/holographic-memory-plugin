"""Hybrid keyword/BM25 retrieval for the memory store.

Combines FTS5 full-text search with Jaccard and HRR similarity via
Reciprocal Rank Fusion (RRF) — a scale-invariant rank aggregation method
that eliminates weight tuning.

Ported from KIK memory_agent.py; RRF fusion inspired by Carbonell & Gale
(1992) and Van Rijsbergen & Shaw (1994).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

# RRF constant. Larger = smoother; 60 is the classic default.
_RRF_C = 60


class FactRetriever:
    """Multi-strategy fact retrieval with RRF-based fusion scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,       # deprecated — ignored
        jaccard_weight: float = 0.3,   # deprecated — ignored
        hrr_weight: float = 0.3,       # deprecated — ignored
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Weight args are now legacy — RRF handles fusion automatically.
        # No need to check hrr._HAS_NUMPY; we just skip HRR scoring if absent.
        del fts_weight, jaccard_weight, hrr_weight  # suppress unused-var warnings

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) retrieval.

        Combines FTS5, Jaccard, and HRR similarity via rank-level fusion.
        No weight tuning needed — each method contributes via rank position.

        Pipeline:
        1. FTS5 candidate retrieval (limit*3 for headroom)
        2. Independent ranking by Jaccard and HRR within candidate set
        3. RRF fusion: score = sum(60 / (rank_m + 60) for m in methods) * trust
        4. Temporal decay (optional)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.

        Inspired by Carbonell & Gale (1992), Van Rijsbergen & Shaw (1994).
        """
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            return []

        # Assign 1-based FTS5 rank from retrieval order.
        # _fts_candidates returns results ordered by rank (best first).
        for i, fact in enumerate(candidates):
            fact["fts_rank_1based"] = i + 1

        query_tokens = self._tokenize(query)
        n = len(candidates)

        # --- Compute Jaccard scores for all candidates ---
        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens
            fact["jaccard_score"] = self._jaccard_similarity(query_tokens, all_tokens)

        # Rank candidates by Jaccard (descending score → ascending rank)
        jaccard_ranked = sorted(
            candidates, key=lambda f: f["jaccard_score"], reverse=True
        )
        for i, fact in enumerate(jaccard_ranked):
            fact["jaccard_rank_1based"] = i + 1

        # --- Compute HRR similarity for all candidates ---
        # HRR requires numpy. Skip if unavailable — FTS5 + Jaccard still work.
        numpy_available = hasattr(hrr, "_HAS_NUMPY") and hrr._HAS_NUMPY
        has_vectors = any(f.get("hrr_vector") for f in candidates)

        if numpy_available and has_vectors:
            query_vec = hrr.encode_text(query, self.hrr_dim)
            for fact in candidates:
                if fact.get("hrr_vector"):
                    fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                    fact["hrr_score"] = hrr.similarity(query_vec, fact_vec)
                else:
                    fact["hrr_score"] = 0.0  # no vector → lowest score

            # Rank by HRR (descending score → ascending rank)
            hrr_ranked = sorted(
                candidates, key=lambda f: f["hrr_score"], reverse=True
            )
            for i, fact in enumerate(hrr_ranked):
                fact["hrr_rank_1based"] = i + 1

        # --- RRF fusion ---
        c = _RRF_C
        scored = []
        for fact in candidates:
            fts_rank = fact["fts_rank_1based"]
            j_rank = fact["jaccard_rank_1based"]
            score = (c / (fts_rank + c)) + (c / (j_rank + c))

            # Add HRR contribution if vectors are available
            if numpy_available and has_vectors and "hrr_rank_1based" in fact:
                score += c / (fact["hrr_rank_1based"] + c)

            score *= fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(
                    fact.get("updated_at") or fact.get("created_at")
                )

            fact["score"] = score
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]

        # Clean up internal ranking fields and strip raw HRR bytes
        for fact in results:
            fact.pop("hrr_vector", None)
            fact.pop("fts_rank_1based", None)
            fact.pop("jaccard_rank_1based", None)
            fact.pop("hrr_rank_1based", None)
            fact.pop("jaccard_score", None)
            fact.pop("hrr_score", None)
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

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.

        For Chinese queries, uses LIKE fallback since FTS5 unicode61
        tokenizer filters out Chinese characters by default.
        """
        import re
        conn = self.store._conn

        # Detect Chinese characters
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        if has_chinese:
            # FTS5 unicode61 filters out Chinese chars, use LIKE fallback
            # Use jieba to extract keywords for better matching
            keywords = []
            try:
                import jieba
                tokens = list(jieba.cut(query))
                keywords = [t.strip() for t in tokens if len(t.strip()) > 1 and not t.isspace()]
            except ImportError:
                # Fallback: use original query
                keywords = [query]
            
            if not keywords:
                keywords = [query]
            
            # Build LIKE query with AND semantics (must match all keywords)
            params: list = []
            like_clauses = []
            for kw in keywords:
                like_clauses.append("f.content LIKE ?")
                params.append(f"%{kw}%")
            
            where_clauses = [" AND ".join(like_clauses)]
            
            if category:
                where_clauses.append("f.category = ?")
                params.append(category)
            
            where_clauses.append("f.trust_score >= ?")
            params.append(min_trust)
            
            where_sql = " AND ".join(where_clauses)
            
            # Use LIKE-based query for Chinese
            sql = f"""
                SELECT f.*, 1.0 as fts_rank
                FROM facts f
                WHERE {where_sql}
                ORDER BY f.trust_score DESC
                LIMIT ?
            """
            params.append(limit)
            
            try:
                rows = conn.execute(sql, params).fetchall()
            except Exception:
                return []
            
            if not rows:
                return []
            
            # Like results get uniform rank
            results = []
            for row in rows:
                fact = dict(row)
                fact.pop("fts_rank", None)
                fact["fts_rank"] = 1.0  # uniform rank for LIKE matches
                results.append(fact)
            
            return results
        
        # Non-Chinese: use standard FTS5 query
        fts_query = query
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        params.append(fts_query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
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

    @staticmethod
    def _has_chinese(text: str) -> bool:
        """Check if text contains Chinese characters."""
        import re
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    @staticmethod
    def _extract_english_tokens(text: str) -> set[str]:
        """Extract English tokens from mixed-language text."""
        import re
        tokens = set()
        # Split on Chinese/non-Chinese boundaries to isolate English words
        segments = re.split(r'([\u4e00-\u9fff]+)', text)
        for seg in segments:
            if not seg or FactRetriever._has_chinese(seg):
                continue
            for word in seg.lower().split():
                cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
                if cleaned:
                    tokens.add(cleaned)
        return tokens

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenization with Chinese support via jieba.

        For Chinese text, uses jieba segmentation.
        For English text, uses simple whitespace splitting.
        Strips common punctuation. No stemming/lemmatization.
        """
        import re
        
        if not text:
            return set()
        
        # Detect Chinese characters (CJK Unified Ideographs)
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        if has_chinese:
            try:
                import jieba
                # Use jieba for Chinese segmentation
                tokens = set()
                for word in jieba.cut(text):
                    cleaned = word.strip(".,;:!?\"'()[]{}#@<>「」『』""''")
                    if cleaned and len(cleaned) > 1:  # filter single chars
                        tokens.add(cleaned)
                return tokens
            except ImportError:
                # Fallback: character-level tokenization for Chinese
                # Split into 2-4 character ngrams for better matching
                chars = [c for c in text if c.strip()]
                tokens = set()
                # Add individual meaningful characters (skip punctuation)
                for c in chars:
                    if '\u4e00' <= c <= '\u9fff':
                        tokens.add(c)
                # Add bigrams for phrase matching
                for i in range(len(chars) - 1):
                    if '\u4e00' <= chars[i] <= '\u9fff' and '\u4e00' <= chars[i+1] <= '\u9fff':
                        tokens.add(chars[i] + chars[i+1])
                return tokens
        else:
            # English: original whitespace tokenization
            tokens = set()
            for word in text.lower().split():
                cleaned = word.strip(".,;:!?\"'()[]{}#@<>>")
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
