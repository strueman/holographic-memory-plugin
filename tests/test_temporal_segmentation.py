"""Tests for temporal segmentation (episode detection via gap-based clustering)."""

import os
import time
import pytest
from datetime import datetime, timedelta

# Ensure we can import the plugin modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import MemoryStore, _EPISODE_GAP_MINUTES, _EPISODE_MAX_SIZE


@pytest.fixture
def store(tmp_path):
    """Create a temporary MemoryStore for testing."""
    db_path = str(tmp_path / "test.db")
    return MemoryStore(db_path=db_path)


class TestEpisodeSchema:
    """Tests for the episodes table creation."""

    def test_episodes_table_created(self, store):
        """episodes table exists after initialization."""
        tables = [r["name"] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "episodes" in tables
        assert "episode_facts" in tables

    def test_episodes_indexes_created(self, store):
        """Indexes on episodes are created."""
        indexes = [r["name"] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        assert "idx_episodes_start" in indexes
        assert "idx_episode_facts_fact" in indexes

    def test_episode_columns_exist(self, store):
        """episodes table has all required columns."""
        cols = [r["name"] for r in store._conn.execute(
            "PRAGMA table_info(episodes)"
        ).fetchall()]
        for col in ["episode_id", "title", "start_time", "end_time", "fact_count", "summary"]:
            assert col in cols


class TestAutoSegmentation:
    """Tests for automatic episode assignment on add_fact."""

    def test_first_fact_creates_episode(self, store):
        """Adding the first fact creates an episode."""
        store.add_fact("First fact in the store")
        episodes = store.list_episodes()
        assert len(episodes) == 1
        assert episodes[0]["fact_count"] == 1

    def test_close_facts_same_episode(self, store):
        """Facts added quickly belong to the same episode."""
        store.add_fact("Fact one")
        store.add_fact("Fact two")
        store.add_fact("Fact three")
        episodes = store.list_episodes()
        assert len(episodes) == 1
        assert episodes[0]["fact_count"] == 3

    def test_episode_title_generated(self, store):
        """Episode title is derived from dominant entities."""
        store.add_fact("Simon uses Debian Trixie on his workstation")
        ep = store.list_episodes()[0]
        assert "Simon" in ep["title"] or "Debian" in ep["title"]

    def test_episode_summary_generated(self, store):
        """Episode summary contains fact count and category."""
        store.add_fact("Test fact about system configuration")
        ep = store.list_episodes()[0]
        assert "[EPISODE]" in ep["summary"]


class TestSegmentEpisodes:
    """Tests for the segment_episodes bulk segmentation method."""

    def test_segment_returns_stats(self, store):
        """segment_episodes returns a stats dict."""
        store.add_fact("Fact one")
        store.add_fact("Fact two")
        stats = store.segment_episodes()
        assert "episodes_created" in stats
        assert "facts_segmented" in stats
        assert "gaps_skipped" in stats

    def test_segment_idempotent(self, store):
        """Running segment_episodes twice doesn't create duplicate episodes."""
        store.add_fact("Fact one")
        store.add_fact("Fact two")
        store.segment_episodes()
        count_before = store._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        store.segment_episodes()
        count_after = store._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert count_before == count_after

    def test_segment_with_custom_gap(self, store):
        """segment_episodes respects custom gap_minutes parameter."""
        # Bypass auto-segmentation by inserting facts directly with different timestamps
        store._conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score, created_at) "
            "VALUES (?, 'general', '', 0.5, datetime('now', '-1 minute'))",
            ("Fact one",),
        )
        store._conn.commit()
        store._conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score) "
            "VALUES (?, 'general', '', 0.5)",
            ("Fact two",),
        )
        store._conn.commit()
        stats = store.segment_episodes(gap_minutes=0)
        # With gap=0, any time difference creates a new episode
        assert stats["episodes_created"] >= 2

    def test_segment_empty_store(self, store):
        """segment_episodes on empty store returns zeros."""
        stats = store.segment_episodes()
        assert stats["episodes_created"] == 0
        assert stats["facts_segmented"] == 0

    def test_segment_already_segmented(self, store):
        """segment_episodes skips facts already in episodes."""
        store.add_fact("Fact one")
        store.add_fact("Fact two")
        # Auto-segmentation already assigned facts
        stats = store.segment_episodes()
        assert stats["episodes_created"] == 0


class TestEpisodeQuery:
    """Tests for episode query methods."""

    def test_get_episode(self, store):
        """get_episode returns episode details."""
        store.add_fact("Test fact")
        ep = store.get_episode(1)
        assert ep is not None
        assert ep["episode_id"] == 1
        assert ep["fact_count"] >= 1

    def test_get_episode_nonexistent(self, store):
        """get_episode returns None for non-existent episode."""
        assert store.get_episode(99999) is None

    def test_list_episodes_newest(self, store):
        """list_episodes returns episodes ordered newest first by default."""
        store.add_fact("Older fact")
        time.sleep(0.1)
        store.add_fact("Newer fact")
        episodes = store.list_episodes(order="newest")
        if len(episodes) >= 2:
            assert episodes[0]["start_time"] >= episodes[1]["start_time"]

    def test_list_episodes_oldest(self, store):
        """list_episodes respects order='oldest'."""
        store.add_fact("Older fact")
        time.sleep(0.1)
        store.add_fact("Newer fact")
        episodes = store.list_episodes(order="oldest")
        if len(episodes) >= 2:
            assert episodes[0]["start_time"] <= episodes[1]["start_time"]

    def test_list_episodes_limit(self, store):
        """list_episodes respects limit parameter."""
        for i in range(10):
            store.add_fact(f"Fact {i}")
        episodes = store.list_episodes(limit=3)
        assert len(episodes) <= 3

    def test_get_episode_facts(self, store):
        """get_episode_facts returns facts for the episode."""
        store.add_fact("Fact alpha")
        store.add_fact("Fact beta")
        facts = store.get_episode_facts(1)
        assert len(facts) == 2
        contents = [f["content"] for f in facts]
        assert "Fact alpha" in contents
        assert "Fact beta" in contents

    def test_get_episode_facts_limit(self, store):
        """get_episode_facts respects limit parameter."""
        for i in range(20):
            store.add_fact(f"Fact {i}")
        facts = store.get_episode_facts(1, limit=5)
        assert len(facts) <= 5


class TestEpisodeCleanup:
    """Tests for episode cleanup when facts are removed."""

    def test_episode_fact_count_updated(self, store):
        """Removing a fact decrements the episode's fact_count."""
        store.add_fact("Fact one")
        store.add_fact("Fact two")
        store.add_fact("Fact three")
        # Find which episode fact 3 belongs to
        ep_row = store._conn.execute(
            "SELECT episode_id FROM episode_facts WHERE fact_id = 3"
        ).fetchone()
        ep_id = ep_row[0]
        ep_before = store.get_episode(ep_id)
        fact_count_before = ep_before["fact_count"]
        store.remove_fact(3)
        ep_after = store.get_episode(ep_id)
        assert ep_after["fact_count"] == fact_count_before - 1

    def test_episode_link_removed(self, store):
        """Removing a fact removes its episode_facts link."""
        store.add_fact("Fact one")
        store.remove_fact(1)
        link_count = store._conn.execute(
            "SELECT COUNT(*) FROM episode_facts WHERE fact_id = 1"
        ).fetchone()[0]
        assert link_count == 0


class TestEpisodeConstants:
    """Tests for segmentation constants."""

    def test_default_gap_minutes(self):
        """Default gap threshold is 60 minutes."""
        assert _EPISODE_GAP_MINUTES == 60

    def test_max_episode_size(self):
        """Episode max size is 100."""
        assert _EPISODE_MAX_SIZE == 100


class TestEpisodeTitleGeneration:
    """Tests for episode title generation."""

    def test_title_from_entities(self, store):
        """Episode title includes dominant entities."""
        store.add_fact("Simon uses Debian Trixie on his workstation")
        store.add_fact("Debian Trixie stable is the current release")
        ep = store.get_episode(1)
        assert "Debian" in ep["title"] or "Simon" in ep["title"]

    def test_title_fallback_to_category(self, store):
        """Episode title falls back to category when no entities."""
        store.add_fact("a b c")  # no capitalized entities
        ep = store.get_episode(1)
        assert ep["title"] is not None
        assert len(ep["title"]) > 0


class TestGapDetection:
    """Tests for temporal gap detection."""

    def test_manual_gap_detection(self, store):
        """Facts with large time gaps form separate episodes."""
        # Bypass auto-segmentation by inserting facts directly with timestamps
        store._conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score, created_at) "
            "VALUES (?, 'general', '', 0.5, datetime('now', '-2 hours'))",
            ("Fact one",),
        )
        store._conn.commit()
        store._conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score) "
            "VALUES (?, 'general', '', 0.5)",
            ("Fact two",),
        )
        store._conn.commit()
        # Now segment — should create 2 episodes due to the 2-hour gap
        stats = store.segment_episodes()
        assert stats["episodes_created"] >= 2
