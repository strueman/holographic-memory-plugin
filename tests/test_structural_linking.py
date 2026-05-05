"""Tests for structural linking (fact-to-fact links via HRR/entity/embedding similarity)."""

import os
import tempfile
import pytest
import sqlite3
import struct

# Ensure we can import the plugin modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import MemoryStore, _HAS_SQLITE_VEC


@pytest.fixture
def store(tmp_path):
    """Create a temporary MemoryStore for testing."""
    db_path = str(tmp_path / "test.db")
    return MemoryStore(db_path=db_path)


@pytest.fixture
def populated_store(store):
    """Store with test facts that should create links."""
    store.add_fact("Simon uses Debian Trixie stable on his workstation")
    store.add_fact("Debian Trixie is installed on the Minisforum UM890 Pro")
    store.add_fact("The Minisforum UM890 Pro has a Ryzen 9 processor")
    return store


class TestFactLinksTable:
    """Tests for the fact_links table creation."""

    def test_fact_links_table_created(self, store):
        """fact_links table exists after initialization."""
        tables = [r["name"] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "fact_links" in tables

    def test_fact_links_indexes_created(self, store):
        """Indexes on fact_links are created."""
        indexes = [r["name"] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        assert "idx_fact_links_a" in indexes
        assert "idx_fact_links_b" in indexes


class TestLinkCreation:
    """Tests for automatic link creation on add_fact."""

    def test_links_created_for_similar_facts(self, store):
        """Adding semantically similar facts creates links."""
        id1 = store.add_fact("Simon uses Debian Trixie stable on his workstation")
        id2 = store.add_fact("Debian Trixie is installed on the Minisforum UM890 Pro")
        links = store.get_linked_facts(id1)
        linked_ids = [f["fact_id"] for f in links]
        assert id2 in linked_ids

    def test_no_links_for_unrelated_facts(self, store):
        """Facts with no semantic overlap don't get linked."""
        id1 = store.add_fact("Culinary recipe: bake sourdough bread at 250 degrees for forty minutes")
        id2 = store.add_fact("Quantum entanglement violates Bell inequality in photon polarization experiments")
        links = store.get_linked_facts(id1)
        linked_ids = [f["fact_id"] for f in links]
        assert id2 not in linked_ids

    def test_no_links_for_first_fact(self, store):
        """First fact in an empty store has no links."""
        id1 = store.add_fact("First fact in an empty store")
        links = store.get_linked_facts(id1)
        assert links == []

    def test_links_capped_per_fact(self, store):
        """Links per fact don't exceed _LINKS_PER_FACT_MAX."""
        from store import _LINKS_PER_FACT_MAX
        ids = []
        for i in range(_LINKS_PER_FACT_MAX + 5):
            fid = store.add_fact(f"Test fact number {i} about Debian configuration")
            ids.append(fid)
        link_count = store._conn.execute(
            "SELECT COUNT(*) FROM fact_links WHERE fact_id_a = ?", (ids[-1],)
        ).fetchone()[0]
        assert link_count <= _LINKS_PER_FACT_MAX

    def test_many_facts_dont_hang(self, store):
        """Adding many facts doesn't cause O(n) blowup."""
        for i in range(150):
            store.add_fact(f"Sequential test fact number {i} about system configuration")
        total_links = store._conn.execute(
            "SELECT COUNT(*) FROM fact_links"
        ).fetchone()[0]
        assert total_links > 0


class TestLinkTypes:
    """Tests for specific link types."""

    def test_entity_overlap_link(self, store):
        """Facts sharing entities get entity_overlap links."""
        id1 = store.add_fact("Simon Walker uses a Debian workstation for development")
        id2 = store.add_fact("Simon Walker prefers Linux over Windows for server deployment")
        links = store.get_linked_facts(id1, link_type="entity_overlap")
        linked_ids = [f["fact_id"] for f in links]
        assert id2 in linked_ids

    def test_hrr_similarity_link(self, store):
        """Facts with similar HRR vectors get hrr_similarity links."""
        id1 = store.add_fact("The holographic memory plugin uses HRR vectors for retrieval")
        id2 = store.add_fact("HRR vectors enable algebraic memory operations")
        links = store.get_linked_facts(id1, link_type="hrr_similarity")
        linked_ids = [f["fact_id"] for f in links]
        assert id2 in linked_ids

    def test_link_type_filter(self, store):
        """Can filter links by link_type."""
        store.add_fact("Simon uses Debian on his workstation")
        store.add_fact("Debian Trixie stable runs on the Minisforum")
        links_all = store.get_linked_facts(1)
        links_hrr = store.get_linked_facts(1, link_type="hrr_similarity")
        assert len(links_hrr) <= len(links_all)


class TestGetLinkedFacts:
    """Tests for the get_linked_facts query method."""

    def test_returns_link_details(self, populated_store):
        """Linked facts include link_type and strength fields."""
        links = populated_store.get_linked_facts(1)
        if links:
            link = links[0]
            assert "link_type" in link
            assert "strength" in link
            assert link["link_type"] in ["hrr_similarity", "entity_overlap", "embedding_similarity"]
            assert 0.0 <= link["strength"] <= 1.0

    def test_returns_fact_data(self, populated_store):
        """Linked facts include full fact data."""
        links = populated_store.get_linked_facts(1)
        if links:
            link = links[0]
            for field in ["fact_id", "content", "category", "trust_score", "created_at"]:
                assert field in link

    def test_nonexistent_fact_returns_empty(self, store):
        """Returns empty list for non-existent fact_id."""
        links = store.get_linked_facts(99999)
        assert links == []

    def test_min_strength_filter(self, store):
        """Can filter by minimum link strength."""
        store.add_fact("Simon uses Debian on his workstation")
        store.add_fact("Debian Trixie runs on the Minisforum")
        links_loose = store.get_linked_facts(1, min_strength=0.0)
        links_strict = store.get_linked_facts(1, min_strength=0.9)
        assert len(links_strict) <= len(links_loose)

    def test_bidirectional_query(self, store):
        """Links found regardless of direction (A->B or B->A)."""
        id1 = store.add_fact("Simon runs Debian Trixie on his workstation")
        id2 = store.add_fact("Debian Trixie is a stable Linux distribution")
        links_from_1 = store.get_linked_facts(id1)
        links_from_2 = store.get_linked_facts(id2)
        ids_from_1 = [f["fact_id"] for f in links_from_1]
        ids_from_2 = [f["fact_id"] for f in links_from_2]
        if id2 in ids_from_1:
            assert id1 in ids_from_2

    def test_link_strength_in_range(self, populated_store):
        """All link strengths are between 0 and 1."""
        links = populated_store.get_linked_facts(1)
        for link in links:
            assert 0.0 <= link["strength"] <= 1.0

    def test_results_sorted_by_strength(self, populated_store):
        """Results are ordered by strength descending."""
        links = populated_store.get_linked_facts(1)
        if len(links) > 1:
            strengths = [l["strength"] for l in links]
            assert strengths == sorted(strengths, reverse=True)


class TestLinkCleanup:
    """Tests for link cleanup when facts are removed."""

    def test_links_removed_with_fact(self, store):
        """Deleting a fact also removes its links."""
        id1 = store.add_fact("Simon uses Debian on his workstation")
        id2 = store.add_fact("Debian Trixie stable runs on the Minisforum")
        links_before = store.get_linked_facts(id1)
        linked_ids_before = [f["fact_id"] for f in links_before]
        if id2 in linked_ids_before:
            store.remove_fact(id2)
            links_after = store.get_linked_facts(id1)
            linked_ids_after = [f["fact_id"] for f in links_after]
            assert id2 not in linked_ids_after


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_get_fact_entities(self, store):
        """_get_fact_entities returns entity names."""
        id1 = store.add_fact("Simon uses Debian Trixie on his workstation")
        entities = store._get_fact_entities(id1)
        assert len(entities) > 0
        assert "debian trixie" in entities

    def test_cosine_similarity_identical(self):
        """Cosine similarity between identical vectors is 1.0."""
        vec = struct.pack("<3f", 1.0, 0.0, 0.0)
        sim = MemoryStore._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity between orthogonal vectors is 0.0."""
        a = struct.pack("<3f", 1.0, 0.0, 0.0)
        b = struct.pack("<3f", 0.0, 1.0, 0.0)
        sim = MemoryStore._cosine_similarity(a, b)
        assert abs(sim - 0.0) < 1e-5

    def test_cosine_similarity_zero_vector(self):
        """Cosine similarity with zero vector returns 0.0."""
        a = struct.pack("<3f", 0.0, 0.0, 0.0)
        b = struct.pack("<3f", 1.0, 0.0, 0.0)
        sim = MemoryStore._cosine_similarity(a, b)
        assert sim == 0.0
