"""Tests for cross-level compression (L0 → L1 observations)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from store import MemoryStore, _CONSOLIDATE_MIN_CLUSTER_SIZE, _CONSOLIDATE_MIN_TRUST


class TestConstants:
    """Validate consolidation constants."""

    def test_min_cluster_size(self):
        assert _CONSOLIDATE_MIN_CLUSTER_SIZE == 3

    def test_min_trust(self):
        assert _CONSOLIDATE_MIN_TRUST == 0.4

    def test_obs_trust(self):
        from store import _CONSOLIDATE_OBS_TRUST
        assert _CONSOLIDATE_OBS_TRUST == 0.7


class TestLevelField:
    """Test the level field in the facts table."""

    def test_default_level_is_zero(self):
        """New facts should have level=0."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Test fact")
            row = store._conn.execute(
                "SELECT level FROM facts WHERE content = ?", ("Test fact",)
            ).fetchone()
            assert row["level"] == 0

    def test_observation_has_level_one(self):
        """Observations created by consolidate should have level=1."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            # Add enough linked facts to form a cluster
            store.add_fact("The system runs Debian Trixie as the base operating system")
            store.add_fact("Python 3.11 on Debian Trixie provides the latest features")
            store.add_fact("Debian Trixie stable release includes many packages")
            store.add_fact("The Debian Trixie package manager uses APT for software management")
            store.add_fact("APT is the package management system for Debian Trixie")

            store.consolidate()

            obs = store._conn.execute(
                "SELECT level FROM facts WHERE content LIKE '[OBSERVATION]%'"
            ).fetchone()
            assert obs is not None
            assert obs["level"] == 1

    def test_existing_facts_migrated_to_level_zero(self):
        """Existing facts should be migrated to level=0."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            # Simulate old DB by inserting without level
            store._conn.execute(
                "INSERT INTO facts (content, trust_score) VALUES ('old fact', 0.5)"
            )
            store._conn.commit()
            # Re-init to trigger migration
            store._init_db()
            row = store._conn.execute(
                "SELECT level FROM facts WHERE content = ?", ("old fact",)
            ).fetchone()
            assert row["level"] == 0


class TestClusterFinding:
    """Test _find_clusters method."""

    def test_finds_connected_components(self):
        """Facts connected via links should form a cluster."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            # Add facts that will link together (shared entities)
            store.add_fact("The system runs Debian Trixie as the base operating system")
            store.add_fact("Python 3.11 on Debian Trixie provides the latest features")
            store.add_fact("Debian Trixie stable release includes many packages")
            store.add_fact("The Debian Trixie package manager uses APT for software management")

            clusters = store._find_clusters()
            assert len(clusters) >= 1
            # All facts should be in one cluster (they share Debian Trixie entity)
            total_in_clusters = sum(len(c) for c in clusters)
            assert total_in_clusters >= 3

    def test_filters_by_min_trust(self):
        """Facts below min trust should not be in clusters."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("High trust fact one")
            store.add_fact("High trust fact two")
            store.add_fact("High trust fact three")
            # Low trust fact
            low_id = store._conn.execute(
                "INSERT INTO facts (content, trust_score) VALUES ('Low trust fact', 0.3)"
            ).lastrowid
            store._conn.commit()

            # Create links between high trust facts
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (1, 2, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (2, 3, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (3, ?, 'entity_overlap', 0.5)".format(low_id),
                (low_id,),
            )
            store._conn.commit()

            clusters = store._find_clusters()
            # Low trust fact should be excluded
            all_cluster_ids = set()
            for c in clusters:
                all_cluster_ids.update(c)
            assert low_id not in all_cluster_ids

    def test_respects_min_cluster_size(self):
        """Clusters smaller than min size should be excluded."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            # Add only 2 linked facts (below min cluster size)
            store.add_fact("Two fact one")
            store.add_fact("Two fact two")
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (1, 2, 'entity_overlap', 0.5)"
            )
            store._conn.commit()

            clusters = store._find_clusters()
            assert len(clusters) == 0

    def test_filters_by_category(self):
        """Category filter should limit clusters to specified category."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Category A fact one", category="test_a")
            store.add_fact("Category A fact two", category="test_a")
            store.add_fact("Category A fact three", category="test_a")
            store.add_fact("Category B fact one", category="test_b")
            store.add_fact("Category B fact two", category="test_b")
            store.add_fact("Category B fact three", category="test_b")

            # Create links within categories
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (1, 2, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (2, 3, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (4, 5, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (5, 6, 'entity_overlap', 0.5)"
            )
            store._conn.commit()

            clusters_a = store._find_clusters(category="test_a")
            clusters_b = store._find_clusters(category="test_b")

            assert len(clusters_a) >= 1
            assert len(clusters_b) >= 1

            # Verify category filtering
            for c in clusters_a:
                cats = set()
                for fid in c:
                    row = store._conn.execute(
                        "SELECT category FROM facts WHERE fact_id = ?", (fid,)
                    ).fetchone()
                    cats.add(row["category"])
                assert cats == {"test_a"}


class TestSummaryGeneration:
    """Test _generate_cluster_summary method."""

    def test_shared_entities(self):
        """Shared entities should appear in summary."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Debian Trixie is the current stable Linux distribution")
            store.add_fact("The Debian Trixie stable release includes Python 3.11")
            store.add_fact("Python 3.11 on Debian Trixie provides the latest features")

            cluster = [1, 2, 3]
            summary = store._generate_cluster_summary(cluster)
            assert "debian trixie" in [e.lower() for e in summary["shared_entities"]]

    def test_dominant_category(self):
        """Dominant category should be the most common."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Fact one", category="tech")
            store.add_fact("Fact two", category="tech")
            store.add_fact("Fact three", category="other")

            summary = store._generate_cluster_summary([1, 2, 3])
            assert summary["dominant_category"] == "tech"

    def test_avg_trust(self):
        """Average trust should be computed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Fact one")  # trust 0.5
            store.add_fact("Fact two")  # trust 0.5
            store.add_fact("Fact three")  # trust 0.5

            summary = store._generate_cluster_summary([1, 2, 3])
            assert summary["avg_trust"] == 0.5

    def test_representative_facts(self):
        """Top 3 facts by trust should be included."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Low trust fact")
            store.add_fact("High trust fact")
            store.add_fact("Medium trust fact")

            # Adjust trust
            store._conn.execute(
                "UPDATE facts SET trust_score = 0.3 WHERE content = 'Low trust fact'"
            )
            store._conn.execute(
                "UPDATE facts SET trust_score = 0.9 WHERE content = 'High trust fact'"
            )
            store._conn.commit()

            summary = store._generate_cluster_summary([1, 2, 3])
            assert len(summary["representative"]) == 3
            # High trust should be first
            assert "High trust" in summary["representative"][0]


class TestObservationStorage:
    """Test _add_observation method."""

    def test_creates_l1_fact(self):
        """Observation should be stored as level=1."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            obs_id = store._add_observation(
                content="[OBSERVATION] Test observation",
                category="test",
                source_fact_ids=[1, 2, 3],
                cluster_entities={"test entity"},
            )
            assert obs_id is not None
            row = store._conn.execute(
                "SELECT level FROM facts WHERE fact_id = ?", (obs_id,)
            ).fetchone()
            assert row["level"] == 1

    def test_deduplication(self):
        """Duplicate entity sets should be skipped."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            entities = {"shared entity"}
            result1 = store._add_observation(
                content="[OBSERVATION] First",
                category="test",
                source_fact_ids=[1],
                cluster_entities=entities,
            )
            result2 = store._add_observation(
                content="[OBSERVATION] Second",
                category="test",
                source_fact_ids=[2],
                cluster_entities=entities,
            )
            assert result1 is not None
            assert result2 is None  # Should be deduplicated

    def test_linked_to_source_facts(self):
        """Observation should have consolidation links to source facts."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            obs_id = store._add_observation(
                content="[OBSERVATION] Test",
                category="test",
                source_fact_ids=[1, 2],
                cluster_entities={"entity"},
            )
            links = store._conn.execute(
                "SELECT fact_id_b FROM fact_links WHERE fact_id_a = ? AND link_type = 'consolidation'",
                (obs_id,),
            ).fetchall()
            fact_ids = {r["fact_id_b"] for r in links}
            assert 1 in fact_ids
            assert 2 in fact_ids

    def test_linked_to_entities(self):
        """Observation should be linked to shared entities."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            obs_id = store._add_observation(
                content="[OBSERVATION] Test",
                category="test",
                source_fact_ids=[1],
                cluster_entities={"test entity"},
            )
            entities = store._get_fact_entities(obs_id)
            assert "test entity" in entities


class TestPipelineIntegration:
    """Test full consolidate() pipeline."""

    def test_consolidate_creates_observations(self):
        """consolidate() should create observations from linked facts."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            # Add facts with shared entities that will link together
            store.add_fact("Debian Trixie is the current stable Linux distribution")
            store.add_fact("The Debian Trixie stable release includes Python 3.11")
            store.add_fact("Python 3.11 on Debian Trixie provides the latest features")
            store.add_fact("Debian Trixie package manager uses APT for software")

            result = store.consolidate()
            assert result["observations_created"] >= 1
            assert result["facts_consolidated"] >= 3

    def test_consolidate_empty_store(self):
        """Empty store should return zero observations."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            result = store.consolidate()
            assert result["observations_created"] == 0
            assert result["clusters_found"] == 0

    def test_consolidate_no_links(self):
        """Facts with no structural links should not form clusters.

        Note: structural linking creates links via HRR/embedding similarity,
        so truly unlinked facts need to be carefully chosen. This test verifies
        that consolidation respects the min_cluster_size threshold.
        """
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            # Add only 2 facts — below min cluster size
            store.add_fact("Completely unrelated fact one about computers")
            store.add_fact("Completely unrelated fact two about cooking")

            result = store.consolidate()
            # With only 2 facts, even if linked, cluster size < 3
            assert result["clusters_eligible"] == 0
            assert result["observations_created"] == 0

    def test_consolidate_idempotent(self):
        """Running consolidate twice should not create duplicate observations."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Debian Trixie is the current stable Linux distribution")
            store.add_fact("The Debian Trixie stable release includes Python 3.11")
            store.add_fact("Python 3.11 on Debian Trixie provides the latest features")

            result1 = store.consolidate()
            obs_count_1 = store._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE level = 1"
            ).fetchone()[0]

            result2 = store.consolidate()
            obs_count_2 = store._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE level = 1"
            ).fetchone()[0]

            assert obs_count_1 == obs_count_2
            assert result2["observations_created"] == 0

    def test_consolidate_with_category_filter(self):
        """Category filter should only consolidate facts in that category."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Tech fact one", category="tech")
            store.add_fact("Tech fact two", category="tech")
            store.add_fact("Tech fact three", category="tech")
            store.add_fact("Other fact one", category="other")
            store.add_fact("Other fact two", category="other")
            store.add_fact("Other fact three", category="other")

            # Create links within categories
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (1, 2, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (2, 3, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (4, 5, 'entity_overlap', 0.5)"
            )
            store._conn.execute(
                "INSERT OR IGNORE INTO fact_links (fact_id_a, fact_id_b, link_type, strength) VALUES (5, 6, 'entity_overlap', 0.5)"
            )
            store._conn.commit()

            result = store.consolidate(category="tech")
            # Should only consolidate tech facts
            tech_obs = store._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE level = 1 AND category = 'tech'"
            ).fetchone()[0]
            assert tech_obs >= 1

    def test_observation_text_format(self):
        """Observation text should follow the expected format."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = MemoryStore(db_path=f.name)
            store.add_fact("Debian Trixie is the current stable Linux distribution")
            store.add_fact("The Debian Trixie stable release includes Python 3.11")
            store.add_fact("Python 3.11 on Debian Trixie provides the latest features")

            store.consolidate()

            obs = store._conn.execute(
                "SELECT content FROM facts WHERE content LIKE '[OBSERVATION]%'"
            ).fetchone()
            assert obs is not None
            assert obs["content"].startswith("[OBSERVATION]")
            assert "Consolidated from" in obs["content"]
            assert "Key points:" in obs["content"]
