"""Tests for graceful degradation when sqlite-vec is unavailable.

These tests explicitly disable vec to verify the fallback path works correctly.
This is critical: if vec fails in production (e.g., missing extension, corrupted DB),
the system must continue functioning with FTS5-only search.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestVecDisabledFallback:
    """Test that the system works correctly when vec is unavailable."""

    @pytest.fixture(autouse=True)
    def _disable_vec(self, monkeypatch):
        """Temporarily disable vec for all tests in this class."""
        import store as store_mod
        monkeypatch.setattr(store_mod, "_HAS_SQLITE_VEC", False)

    def test_vec_table_not_created(self):
        """When vec is disabled, facts_vec table should not exist."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            tables = [r[0] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            assert "facts_vec" not in tables

    def test_add_fact_works_without_vec(self):
        """Adding facts should work normally when vec is disabled."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Test fact about Debian")
            store.add_fact("Another fact about hardware")
            count = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            assert count == 2

    def test_search_works_fts5_only(self):
        """Search should return results via FTS5 when vec is unavailable."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Debian Trixie is the current stable release")
            store.add_fact("The Ryzen 9 8945 has an Radeon 780M iGPU")
            store.add_fact("Holographic memory uses HRR vectors for reasoning")

            results = store.search_facts("Debian Trixie", limit=5)
            assert len(results) >= 1
            contents = [r["content"] for r in results]
            assert any("Debian" in c for c in contents)

    def test_search_fallback_to_fts5_only(self):
        """The FactRetriever should fall back to FTS5-only when vec is disabled."""
        from store import MemoryStore
        from retrieval import FactRetriever
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Debian Trixie is the current stable release")
            store.add_fact("The Ryzen 9 8945 has a Radeon 780M iGPU")
            store.add_fact("Holographic memory uses HRR vectors for reasoning")
            store.add_fact("The evidence-gap tracker checks query coverage")

            retriever = FactRetriever(store)
            results = retriever.search("Debian stable release", limit=5)
            assert len(results) >= 1
            assert all(r.get("score", 0) > 0 for r in results)
            assert all("vec_rank" not in r for r in results)

    def test_search_rrf_degrades_to_two_signals(self):
        """RRF fusion should work with fewer signals when vec is unavailable."""
        from store import MemoryStore
        from retrieval import FactRetriever
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Debian Trixie is the current stable release of Debian Linux")
            store.add_fact("Ubuntu 24.04 is the latest LTS release")
            store.add_fact("Fedora 40 uses the latest kernel version")
            store.add_fact("Arch Linux is a rolling release distribution")
            store.add_fact("The Radeon 7900 XTX has 24GB of VRAM")
            store.add_fact("Linux kernel 6.14 introduced new scheduling improvements")

            retriever = FactRetriever(store)
            results = retriever.search("Linux distribution stable release", limit=5)
            assert len(results) >= 1
            assert all(r.get("score", 0) > 0 for r in results)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_compute_embedding_is_noop(self):
        """_compute_embedding should silently do nothing when vec is disabled."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Test fact")
            fact_id = store._conn.execute(
                "SELECT fact_id FROM facts LIMIT 1"
            ).fetchone()["fact_id"]
            store._compute_embedding(fact_id, "new content")

    def test_backfill_embeddings_returns_zero(self):
        """backfill_embeddings should return 0 when vec is unavailable."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Test fact one")
            store.add_fact("Test fact two")
            count = store.backfill_embeddings()
            assert count == 0

    def test_evidence_gap_works_without_vec(self):
        """Evidence-gap tracker should work independently of vec."""
        from store import MemoryStore
        from retrieval import FactRetriever
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Debian Trixie stable release 2024")
            store.add_fact("The Ryzen 9 8945 has Radeon 780M integrated GPU")

            retriever = FactRetriever(store)
            results = retriever.search("Debian Trixie stable", limit=5, evidence_gap=True)
            assert len(results) >= 1
            assert "evidence_gap" in results[0]
            gap = results[0]["evidence_gap"]
            assert "coverage_ratio" in gap
            assert "covered" in gap
            assert "uncovered" in gap
            assert "needs_followup" in gap

    def test_structural_linking_without_embedding(self):
        """Structural linking should still work with HRR + entity signals when vec is disabled."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            # Use facts with identical entity strings — avoid sentence-start capitalisation
            store.add_fact("The system runs Debian Trixie as the base operating system")
            store.add_fact("Python 3.11 on Debian Trixie provides the latest features")

            links = store.get_linked_facts(2)
            assert len(links) >= 1
            link_types = {l["link_type"] for l in links}
            assert link_types & {"entity_overlap", "hrr_similarity"}

    def test_update_fact_works_without_vec(self):
        """Updating facts should work when vec is disabled."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Original content")
            fact_id = store._conn.execute(
                "SELECT fact_id FROM facts LIMIT 1"
            ).fetchone()["fact_id"]
            store.update_fact(fact_id, content="Updated content")
            row = store._conn.execute(
                "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            assert row["content"] == "Updated content"

    def test_remove_fact_works_without_vec(self):
        """Removing facts should work when vec is disabled."""
        from store import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Fact to remove")
            store.add_fact("Fact to keep")
            fact_id = store._conn.execute(
                "SELECT fact_id FROM facts ORDER BY fact_id DESC LIMIT 1"
            ).fetchone()["fact_id"]
            store.remove_fact(fact_id)
            count = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            assert count == 1
