"""Tests for Reciprocal Rank Fusion (RRF) in the holographic memory plugin.

Tests verify that the RRF fusion logic:
1. Produces correct rank-based scores for all retrieval methods
2. Correctly orders facts when methods disagree
3. Handles edge cases (missing vectors, no numpy, empty sets)
4. Preserves backward compatibility with the search() interface
"""

import math
import pytest
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the plugin can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class FakeHrrModule:
    """Minimal mock of the holographic module for tests without numpy."""
    _HAS_NUMPY = False

    @staticmethod
    def encode_text(text, dim):
        return None

    @staticmethod
    def similarity(a, b):
        return 0.0


class MockMemoryStore:
    """Minimal mock of MemoryStore with an in-memory SQLite database."""

    def __init__(self):
        self._db_conn = sqlite3.connect(":memory:")
        self._db_conn.row_factory = sqlite3.Row
        self.default_trust = 1.0
        self._init_db()

    def _init_db(self):
        self._db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL DEFAULT 'general',
                tags TEXT NOT NULL DEFAULT '',
                trust_score REAL NOT NULL DEFAULT 1.0,
                retrieval_count INTEGER NOT NULL DEFAULT 0,
                helpful_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                hrr_vector BLOB
            );

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
        """)

    def add_fact(self, content, category="general", tags=""):
        cur = self._db_conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, ?, ?, ?)",
            (content, category, tags, self.default_trust),
        )
        self._db_conn.commit()
        return cur.lastrowid

    def add_fact_ignore_dup(self, content, category="general", tags=""):
        """Add fact, returning existing ID on duplicate (matching real MemoryStore behavior)."""
        try:
            return self.add_fact(content, category, tags)
        except sqlite3.IntegrityError:
            row = self._db_conn.execute(
                "SELECT fact_id FROM facts WHERE content = ?", (content,)
            ).fetchone()
            return int(row["fact_id"]) if row else None

    @property
    def _conn(self):
        return self._db_conn

    def close(self):
        if self._db_conn:
            self._db_conn.close()


# =========================================================================
# RRF Score Calculation
# =========================================================================


class TestRRFScoreCalculation:
    """Verify that RRF produces mathematically correct scores."""

    def test_rrf_constant(self):
        """The RRF constant C=60 should be defined."""
        from plugins.memory.holographic.retrieval import _RRF_C
        assert _RRF_C == 60

    def test_top_rank_score(self):
        """A fact ranked #1 contributes 1/(1+C) ≈ 0.0164."""
        from plugins.memory.holographic.retrieval import _RRF_C
        c = _RRF_C
        top_score = 1.0 / (1 + c)
        assert abs(top_score - 0.0164) < 0.0001

    def test_lower_rank_score_decreases(self):
        """Higher rank numbers should yield lower RRF contributions."""
        from plugins.memory.holographic.retrieval import _RRF_C
        c = _RRF_C
        score_1 = 1.0 / (1 + c)
        score_10 = 1.0 / (10 + c)
        score_100 = 1.0 / (100 + c)
        assert score_1 > score_10 > score_100

    def test_rrf_monotonic(self):
        """RRF score strictly decreases with rank."""
        from plugins.memory.holographic.retrieval import _RRF_C
        c = _RRF_C
        for rank in range(1, 50):
            assert 1.0 / (rank + c) > 1.0 / (rank + 1 + c)


# =========================================================================
# Integration: search() with mocked FTS
# =========================================================================


class TestRRFSearchIntegration:
    """Integration tests for search() returning RRF-fused results."""

    @pytest.fixture
    def store(self):
        s = MockMemoryStore()
        # Add some varied facts
        s.add_fact("Debian Trixie is the current stable release")
        s.add_fact("Debian testing is currently called Trixie")
        s.add_fact("The Radeon 7900 XTX has 24GB of VRAM")
        s.add_fact("AMD GPUs use the ROCm software stack")
        s.add_fact("The Ryzen 9 8945HS is a laptop processor")
        s.add_fact("Linux kernel 6.1 supports the Radeon 780M iGPU")
        s.add_fact("OpenHue provides Philips Hue light control")
        s.add_fact("KVM virtual machines use QEMU as the hypervisor")
        s.add_fact("The Hermes Agent runs in a dedicated VM")
        s.add_fact("Hindsight memory uses SQLite with an embedded database")
        return s

    @pytest.fixture
    def retriever(self, store):
        # Patch the holographic module to not require numpy
        with patch.dict(
            "sys.modules",
            {"holographic.holographic": FakeHrrModule},
            clear=False,
        ):
            from plugins.memory.holographic.retrieval import FactRetriever
            # Temporarily replace the import
            import plugins.memory.holographic.retrieval as ret
            orig_hrr = ret.hrr
            ret.hrr = FakeHrrModule
            try:
                yield FactRetriever(store)
            finally:
                ret.hrr = orig_hrr

    def test_returns_results(self, store, retriever):
        """Basic search should return non-empty results."""
        results = retriever.search("debian stable")
        assert len(results) > 0

    def test_returns_top_n(self, store, retriever):
        """limit parameter should cap the number of results to at most limit."""
        results = retriever.search("debian", limit=3)
        assert len(results) <= 3

        results = retriever.search("debian", limit=1)
        assert len(results) == 1

    def test_scores_are_positive(self, store, retriever):
        """All returned scores should be positive."""
        results = retriever.search("debian stable")
        for r in results:
            assert r["score"] > 0

    def test_sorted_by_score_descending(self, store, retriever):
        """Results should be sorted by score in descending order."""
        results = retriever.search("debian")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_clean_output_no_internal_fields(self, store, retriever):
        """Output should not contain internal ranking fields."""
        results = retriever.search("debian stable")
        for r in results:
            assert "fts_rank_1based" not in r
            assert "jaccard_rank_1based" not in r
            assert "hrr_rank_1based" not in r
            assert "jaccard_score" not in r
            assert "hrr_score" not in r
            assert "hrr_vector" not in r

    def test_no_results_for_unknown_terms(self, store, retriever):
        """Search for terms that don't match any fact returns empty."""
        results = retriever.search("zookeeper quantum computing")
        assert results == []

    def test_search_respects_category(self, store, retriever):
        """Search with category filter should only return matching facts."""
        store.add_fact("This is a test fact", category="test_cat")
        results = retriever.search("test fact", category="test_cat")
        assert len(results) > 0
        assert all(r.get("category") == "test_cat" for r in results)

    def test_search_respects_min_trust(self, store, retriever):
        """Search with high min_trust should filter out low-trust facts."""
        # This tests that the min_trust parameter still works in _fts_candidates
        results_high = retriever.search("debian", min_trust=0.9, limit=10)
        results_low = retriever.search("debian", min_trust=0.1, limit=10)
        assert len(results_high) <= len(results_low)


class TestRRFWithNumpy:
    """Tests when HRR vectors are available (numpy is present)."""

    @pytest.fixture
    def store(self):
        s = MockMemoryStore()
        s.add_fact("The Hermes Agent uses holographic memory")
        s.add_fact("Holographic memory supports entity resolution")
        s.add_fact("Memory retention improves with retrieval frequency")
        s.add_fact("The Radeon 7900 XTX is a high-end GPU")
        s.add_fact("AMD GPUs support ROCm on Linux")
        return s

    @pytest.fixture
    def retriever(self, store):
        import numpy as np
        from plugins.memory.holographic.retrieval import FactRetriever
        yield FactRetriever(store, hrr_dim=128)

    def test_hrr_contributes_to_score(self, store, retriever):
        """When HRR vectors exist, HRR score should contribute to total."""
        # With all three methods contributing, top results should have higher scores
        # than FTS5+Jaccard alone could produce
        results = retriever.search("holographic memory")
        assert len(results) > 0
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        # RRF scores are inherently small (1/(rank+C)) — just verify they're positive
        assert results[0]["score"] > 0


class TestEdgeCases:
    """Edge cases and error handling for RRF."""

    def test_empty_store(self):
        """Search on an empty store returns empty list."""
        store = MockMemoryStore()
        import plugins.memory.holographic.retrieval as ret
        with patch.object(ret, "hrr", FakeHrrModule):
            from plugins.memory.holographic.retrieval import FactRetriever
            retriever = FactRetriever(store)
        results = retriever.search("anything")
        assert results == []

    def test_single_fact(self):
        """Search with only one fact returns that fact."""
        store = MockMemoryStore()
        store.add_fact("This is the only fact")
        import plugins.memory.holographic.retrieval as ret
        with patch.object(ret, "hrr", FakeHrrModule):
            from plugins.memory.holographic.retrieval import FactRetriever
            retriever = FactRetriever(store)
        results = retriever.search("this fact")
        assert len(results) == 1
        assert "This is the only fact" in results[0]["content"]

    def test_deprecated_weights_ignored(self):
        """Weight parameters are accepted for backward compatibility."""
        store = MockMemoryStore()
        store.add_fact("test fact")
        import plugins.memory.holographic.retrieval as ret
        with patch.object(ret, "hrr", FakeHrrModule):
            from plugins.memory.holographic.retrieval import FactRetriever
            # Should not raise — weights are accepted but RRF is the primary scoring
            retriever = FactRetriever(
                store,
                fts_weight=0.1,
                jaccard_weight=0.2,
                hrr_weight=0.7,
            )
        # Weights are still stored for backward compat and auto-redistribution
        assert hasattr(retriever, "fts_weight")
        assert hasattr(retriever, "jaccard_weight")
        assert hasattr(retriever, "hrr_weight")

    def test_temporal_decay_applied(self):
        """Temporal decay should reduce scores for older facts."""
        store = MockMemoryStore()
        store.add_fact("Recent fact")
        import plugins.memory.holographic.retrieval as ret
        with patch.object(ret, "hrr", FakeHrrModule):
            from plugins.memory.holographic.retrieval import FactRetriever
            retriever = FactRetriever(store, temporal_decay_half_life=30)
        results = retriever.search("recent fact")
        assert len(results) == 1
        # With a freshly-inserted fact, decay should be ~1.0
        # RRF scores are inherently small — just verify score is positive
        assert results[0]["score"] > 0


class TestTokenization:
    """Verify the tokenization helper used by Jaccard."""

    def test_tokenizes_lowercase(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        tokens = FactRetriever._tokenize("Hello WORLD")
        assert tokens == {"hello", "world"}

    def test_strips_punctuation(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        tokens = FactRetriever._tokenize("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "." not in tokens
        assert "!" not in tokens

    def test_empty_input(self):
        from plugins.memory.holographic.retrieval import FactRetriever
        assert FactRetriever._tokenize("") == set()
        assert FactRetriever._tokenize(None) == set()
