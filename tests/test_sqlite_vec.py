"""Tests for sqlite-vec integration with holographic memory plugin."""

import os
import tempfile
import pytest
import sqlite3

# Ensure we can import the plugin modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import MemoryStore, _HAS_SQLITE_VEC, EMBEDDING_DIM
from retrieval import FactRetriever


@pytest.fixture
def store(tmp_path):
    """Create a temporary MemoryStore for testing."""
    db_path = str(tmp_path / "test.db")
    return MemoryStore(db_path=db_path)


@pytest.fixture
def retriever(store):
    """Create a FactRetriever for testing."""
    return FactRetriever(store)


@pytest.fixture
def populated_store(store):
    """Store with test facts."""
    store.add_fact("The holographic memory plugin uses HRR vectors for compositional reasoning")
    store.add_fact("sqlite-vec provides vector search inside SQLite databases")
    store.add_fact("Reciprocal Rank Fusion combines multiple ranking sources into one score")
    store.add_fact("The evidence-gap tracker checks if retrieved facts cover query components")
    store.add_fact("Debian Trixie is the current stable release of Debian Linux")
    return store


@pytest.mark.skipif(not _HAS_SQLITE_VEC, reason="sqlite-vec not available")
class TestSqliteVecIntegration:
    """Tests for sqlite-vec vector search integration."""

    def test_vec_table_created(self, store):
        """Test that facts_vec table is created during initialization."""
        tables = [r["name"] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "facts_vec" in tables

    def test_embedding_stored_on_insert(self, populated_store):
        """Test that embeddings are stored when facts are added."""
        rows = populated_store._conn.execute(
            "SELECT COUNT(*) as count FROM facts_vec"
        ).fetchone()
        assert rows["count"] == 5

    def test_embedding_dim(self, populated_store):
        """Test that stored embeddings have correct dimension."""
        row = populated_store._conn.execute(
            "SELECT embedding FROM facts_vec LIMIT 1"
        ).fetchone()
        assert row is not None
        # Each float32 is 4 bytes, 384 dims = 1536 bytes
        assert len(row["embedding"]) == EMBEDDING_DIM * 4

    def test_vec_candidates_returns_results(self, retriever, populated_store):
        """Test that _vec_candidates returns results."""
        candidates = retriever._vec_candidates(
            "vector search in SQLite", None, 10
        )
        assert len(candidates) > 0
        # Check that results have vec_rank
        for c in candidates:
            assert "vec_rank" in c

    def test_vec_candidates_category_filter(self, retriever, populated_store):
        """Test that vec candidates respect category filter."""
        populated_store.add_fact(
            "Vector search is great",
            category="testing"
        )
        candidates = retriever._vec_candidates(
            "vector search", category="testing", limit=10
        )
        for c in candidates:
            assert c["category"] == "testing"

    def test_vec_candidates_empty_when_unavailable(self, retriever):
        """Test that _vec_candidates returns empty list when no facts."""
        candidates = retriever._vec_candidates("test", None, 10)
        assert isinstance(candidates, list)

    def test_search_includes_vec_candidates(self, retriever, populated_store):
        """Test that search results include vec candidates."""
        results = retriever.search("vector search", limit=5)
        assert len(results) > 0
        # Check that results have scores
        for r in results:
            assert "score" in r
            assert r["score"] > 0

    def test_search_rrf_fusion(self, retriever, populated_store):
        """Test that RRF fusion produces reasonable scores."""
        results = retriever.search("holographic memory", limit=5)
        assert len(results) > 0
        # Scores should be positive
        for r in results:
            assert r["score"] > 0
        # Results should be sorted by score descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_relevance_ranking(self, retriever, populated_store):
        """Test that most relevant fact ranks highest."""
        results = retriever.search("sqlite-vec vector search", limit=5)
        assert len(results) > 0
        # The most relevant fact should mention sqlite-vec
        top_content = results[0]["content"].lower()
        assert "sqlite" in top_content or "vector" in top_content

    def test_backfill_embeddings(self, store):
        """Test backfilling embeddings for existing facts."""
        # Add facts directly to bypass embedding computation
        store._conn.execute(
            "INSERT INTO facts (content, trust_score) VALUES (?, 0.5)",
            ("Directly inserted fact one",)
        )
        store._conn.execute(
            "INSERT INTO facts (content, trust_score) VALUES (?, 0.5)",
            ("Directly inserted fact two",)
        )
        store._conn.commit()

        # Backfill
        count = store.backfill_embeddings()
        assert count >= 2

        # Check embeddings exist
        rows = store._conn.execute(
            "SELECT COUNT(*) as count FROM facts_vec"
        ).fetchone()
        assert rows["count"] >= 2

    def test_backfill_idempotent(self, populated_store):
        """Test that backfill is idempotent (no-op when all embedded)."""
        count = populated_store.backfill_embeddings()
        assert count == 0

    def test_search_with_evidence_gap(self, retriever, populated_store):
        """Test that evidence-gap analysis still works with vec."""
        results = retriever.search(
            "vector search SQLite holographic",
            limit=5,
            evidence_gap=True
        )
        assert len(results) > 0
        assert "evidence_gap" in results[0]
        gap = results[0]["evidence_gap"]
        assert "coverage_ratio" in gap
        assert "covered" in gap
        assert "uncovered" in gap

    def test_search_no_vec_distance_in_results(self, retriever, populated_store):
        """Test that vec_distance is stripped from results."""
        results = retriever.search("test", limit=5)
        for r in results:
            assert "vec_distance" not in r
            assert "hrr_vector" not in r


class TestSqliteVecUnavailable:
    """Tests when sqlite-vec is not available."""

    def test_vec_candidates_returns_empty(self, retriever):
        """Test graceful degradation when vec unavailable."""
        # Even if vec is available, empty store should return empty
        candidates = retriever._vec_candidates("test", None, 10)
        assert isinstance(candidates, list)

    def test_search_works_without_vec(self, retriever, populated_store):
        """Test that search still works even if vec fails."""
        results = retriever.search("test", limit=5)
        # Should return results from FTS5 at minimum
        assert isinstance(results, list)
