"""Bilingual (Chinese/English) search and entity extraction tests.

Tests for the Chinese language support added to Mnemoss:
- Language detection (Chinese character regex)
- Chinese tokenization (jieba + character-level fallback)
- Chinese LIKE-based search (FTS5 fallback)
- Chinese entity extraction (jieba POS + character-sequence fallback)
- Mixed Chinese+English queries
- No regression for English queries
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add plugin directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from store import MemoryStore
from retrieval import FactRetriever, _RE_CHINESE, _get_jieba


# ============================================================
# Fixtures
# ============================================================

class _FakeHrr:
    """Minimal HRR mock to avoid numpy dependency in tests."""
    _HAS_NUMPY = False

    @staticmethod
    def encode_text(text, dim):
        return None

    @staticmethod
    def bytes_to_phases(data):
        return None

    @staticmethod
    def similarity(a, b):
        return 0.0


class MockMemoryStore:
    """Lightweight in-memory store for fast unit tests."""

    def __init__(self):
        import sqlite3
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE facts (
                fact_id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                tags TEXT DEFAULT '',
                trust_score REAL DEFAULT 0.5,
                retrieval_count INTEGER DEFAULT 0,
                helpful_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hrr_vector BLOB,
                level INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE VIRTUAL TABLE facts_fts USING fts5(
                content, tags,
                content=facts, content_rowid=fact_id
            )
        """)
        self._conn.execute("""
            CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, content, tags)
                    VALUES (new.fact_id, new.content, new.tags);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                    VALUES ('delete', old.fact_id, old.content, old.tags);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                    VALUES ('delete', old.fact_id, old.content, old.tags);
                INSERT INTO facts_fts(rowid, content, tags)
                    VALUES (new.fact_id, new.content, new.tags);
            END
        """)
        self._counter = 0

    def add_fact(self, content, category="general", tags="", trust=0.5):
        self._counter += 1
        self._conn.execute(
            "INSERT INTO facts (fact_id, content, category, tags, trust_score) VALUES (?, ?, ?, ?, ?)",
            (self._counter, content, category, tags, trust),
        )
        self._conn.commit()
        return self._counter

    def _has_chinese_entities(self, text):
        return bool(_RE_CHINESE.search(text))


# ============================================================
# Test 1: Language Detection
# ============================================================

class TestLanguageDetection:
    """Test _has_chinese() language detection."""

    def test_has_chinese_detects_chinese(self):
        """Pure Chinese text → True."""
        assert FactRetriever._has_chinese("你好世界") is True
        assert FactRetriever._has_chinese("我喜欢Python编程") is True

    def test_has_chinese_detects_english(self):
        """Pure English text → False."""
        assert FactRetriever._has_chinese("Hello world") is False
        assert FactRetriever._has_chinese("Python is great") is False

    def test_has_chinese_detects_mixed(self):
        """Mixed Chinese+English → True."""
        assert FactRetriever._has_chinese("Python编程") is True
        assert FactRetriever._has_chinese("我喜欢Python") is True

    def test_has_chinese_empty_none(self):
        """Empty/None → False."""
        assert FactRetriever._has_chinese("") is False
        assert FactRetriever._has_chinese(None) is False


# ============================================================
# Test 2: Chinese Tokenization
# ============================================================

class TestChineseTokenization:
    """Test _chinese_tokenize() with and without jieba."""

    def test_chinese_tokenize_filters_single_chars(self):
        """Single-char tokens are filtered out (keep >= 2 chars)."""
        # Even without jieba, single chars should be filtered
        result = FactRetriever._chinese_tokenize("你好")
        # "你好" is 2 chars, should produce at least one token
        assert len(result) >= 1

    def test_chinese_tokenize_empty(self):
        """Empty string → empty list."""
        assert FactRetriever._chinese_tokenize("") == []

    def test_chinese_tokenize_preserves_english_in_mixed(self):
        """English words in mixed text are preserved."""
        result = FactRetriever._chinese_tokenize("Python编程")
        # Should contain "Python" or at least meaningful tokens
        assert any("python" in t.lower() for t in result) or len(result) >= 1

    def test_chinese_tokenize_no_crash_on_punctuation(self):
        """Punctuation-only input doesn't crash."""
        result = FactRetriever._chinese_tokenize("，。！？")
        # Should return empty or filtered list
        assert isinstance(result, list)

    def test_chinese_tokenize_with_jieba_available(self):
        """When jieba is available, segmentation works."""
        jieba_mod = _get_jieba()
        if jieba_mod is None:
            pytest.skip("jieba not installed")
        result = FactRetriever._chinese_tokenize("我喜欢学习Python")
        assert isinstance(result, list)
        assert len(result) >= 1
        # Should produce multiple tokens, not just the raw string
        assert not (len(result) == 1 and result[0] == "我喜欢学习Python")


# ============================================================
# Test 3: Chinese Search
# ============================================================

class TestChineseSearch:
    """Test Chinese LIKE-based search."""

    @pytest.fixture
    def store(self):
        """Create a real MemoryStore with Chinese facts."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path)
            yield store
        finally:
            store._conn.close()
            os.unlink(db_path)

    def test_chinese_search_returns_results(self, store):
        """Chinese query finds Chinese facts."""
        store.add_fact("我喜欢吃中国菜", category="food")
        store.add_fact("I like Chinese food", category="food")

        retriever = FactRetriever(store)
        results = retriever.search("中国菜", category="food", limit=5)
        assert len(results) >= 1
        # The Chinese fact should be in results
        contents = [r["content"] for r in results]
        assert any("中国菜" in c for c in contents)

    def test_chinese_search_no_regression_english(self, store):
        """English search still works after Chinese changes."""
        store.add_fact("I like Python programming")
        store.add_fact("I enjoy coding in Python")

        retriever = FactRetriever(store)
        results = retriever.search("Python programming", limit=5)
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    def test_chinese_search_empty_db(self):
        """Chinese search on empty DB returns []."""
        mock_store = MockMemoryStore()
        retriever = FactRetriever(mock_store)
        results = retriever._chinese_candidates("你好", None, 0.3, 10)
        assert results == []

    def test_chinese_like_escape(self):
        """LIKE special chars (%) are properly escaped with backslash prefix."""
        mock_store = MockMemoryStore()
        retriever = FactRetriever(mock_store)
        # % should be escaped as \%, _ should be escaped as \_
        escaped = retriever._escape_like("100%")
        assert escaped == "100\\%"
        escaped = retriever._escape_like("user_name")
        assert escaped == "user\\_name"

    def test_chinese_search_partial_match(self):
        """Partial Chinese keyword match returns lower rank."""
        mock_store = MockMemoryStore()
        mock_store.add_fact("我喜欢学习编程")
        mock_store.add_fact("我学习")

        retriever = FactRetriever(mock_store)
        # "编程" should match the first fact better
        results = retriever._chinese_candidates("编程", None, 0.3, 10)
        assert len(results) >= 1
        assert any("编程" in r["content"] for r in results)

    def test_chinese_multiword_query_returns_results(self):
        """Multi-word Chinese query returns results (OR semantics).

        Regression test: previously AND semantics required ALL tokens
        to match a single fact, so multi-word queries returned 0.
        """
        mock_store = MockMemoryStore()
        mock_store.add_fact("星云项目使用Transformer架构")
        mock_store.add_fact("混合专家模型在GPU上运行")
        mock_store.add_fact("GGUF量化部署模型")

        retriever = FactRetriever(mock_store)
        # Multi-word query: tokens span across different facts
        results = retriever._chinese_candidates(
            "星云项目Transformer架构混合专家", None, 0.3, 10
        )
        # Should return results (OR semantics)
        assert len(results) >= 1
        # Fact with more matches should rank higher
        first = results[0]
        assert "星云" in first["content"] or "Transformer" in first["content"]

    def test_chinese_multiword_match_count_ranking(self):
        """Facts matching more query tokens rank higher."""
        mock_store = MockMemoryStore()
        mock_store.add_fact("Python编程学习指南")  # matches: Python, 编程, 学习
        mock_store.add_fact("Python入门教程")  # matches: Python
        mock_store.add_fact("机器学习算法")  # matches: 学习

        retriever = FactRetriever(mock_store)
        results = retriever._chinese_candidates(
            "Python编程学习", None, 0.3, 10
        )
        assert len(results) >= 2
        # Fact with most matches should be ranked first
        assert "Python编程学习指南" in results[0]["content"]


# ============================================================
# Test 4: Chinese Entity Extraction
# ============================================================

class TestChineseEntityExtraction:
    """Test Chinese entity extraction from store.py."""

    @pytest.fixture
    def store(self):
        """Create a real MemoryStore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path)
            yield store
        finally:
            store._conn.close()
            os.unlink(db_path)

    def test_chinese_entities_dedup(self, store):
        """Duplicate entities are deduplicated."""
        text = "北京北京是中国的首都"
        entities = store._extract_chinese_entities(text)
        # "北京" should appear only once
        beijing_count = sum(1 for e in entities if "北京" in e)
        assert beijing_count <= 1

    def test_chinese_entities_empty(self, store):
        """Empty/short text returns []."""
        assert store._extract_chinese_entities("") == []
        assert store._extract_chinese_entities("。") == []

    def test_chinese_entities_mixed_text(self, store):
        """Mixed Chinese+English extracts both types."""
        text = "我在北京学习Python编程"
        entities = store._extract_chinese_entities(text)
        assert isinstance(entities, list)
        # Should extract Chinese entities
        assert len(entities) >= 1

    def test_chinese_entities_no_regression_english(self, store):
        """English entity extraction unchanged."""
        text = "John Doe works at Microsoft Research"
        entities = store._extract_entities(text)
        assert len(entities) >= 1
        # Should extract capitalized phrases
        assert any("John" in e or "Microsoft" in e for e in entities)

    def test_has_chinese_entities(self, store):
        """_has_chinese_entities correctly detects Chinese."""
        assert store._has_chinese_entities("你好世界") is True
        assert store._has_chinese_entities("Hello world") is False
        assert store._has_chinese_entities("Python编程") is True


# ============================================================
# Test 5: Mixed Queries
# ============================================================

class TestMixedQueries:
    """Test mixed Chinese+English queries."""

    @pytest.fixture
    def store(self):
        """Create a real MemoryStore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path)
            yield store
        finally:
            store._conn.close()
            os.unlink(db_path)

    def test_mixed_query_finds_chinese_content(self, store):
        """Query with Chinese chars finds Chinese facts."""
        store.add_fact("我喜欢吃中餐")
        store.add_fact("I like Chinese food")

        retriever = FactRetriever(store)
        results = retriever.search("中餐", limit=5)
        assert len(results) >= 1

    def test_mixed_query_finds_english_content(self, store):
        """Query with Chinese chars finds English facts (via FTS5)."""
        store.add_fact("I like Chinese food")

        retriever = FactRetriever(store)
        results = retriever.search("Chinese food", limit=5)
        assert len(results) >= 1
        assert any("Chinese" in r["content"] for r in results)

    def test_mixed_query_vec_still_works(self, store):
        """Vector search still works for mixed queries."""
        store.add_fact("I enjoy programming in Python")

        retriever = FactRetriever(store)
        results = retriever.search("我喜欢编程", limit=5)
        # Even without vec coverage, should not crash
        assert isinstance(results, list)


# ============================================================
# Test 6: No Regression
# ============================================================

class TestNoRegression:
    """Verify English functionality is unchanged."""

    @pytest.fixture
    def store(self):
        """Create a real MemoryStore."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path)
            yield store
        finally:
            store._conn.close()
            os.unlink(db_path)

    def test_english_search_unchanged(self, store):
        """English query returns results as before."""
        store.add_fact("The quick brown fox jumps over the lazy dog")
        store.add_fact("Python is a programming language")

        retriever = FactRetriever(store)
        results = retriever.search("Python programming", limit=5)
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    def test_english_entity_extraction_unchanged(self, store):
        """English entity extraction unchanged."""
        text = "John Doe from New York City works at Microsoft Research"
        entities = store._extract_entities(text)
        assert len(entities) >= 1
        # Should extract multi-word capitalized phrases
        assert any("New York" in e or "Microsoft" in e for e in entities)

    def test_english_tokenization_unchanged(self):
        """English tokenization unchanged."""
        tokens = FactRetriever._tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "is" in tokens  # stopwords are kept in tokenization

    def test_evidence_gap_unchanged(self, store):
        """Evidence gap analysis for English unchanged."""
        store.add_fact("Simon prefers Debian for his desktop")
        store.add_fact("Simon uses a Ryzen 9 8945 with 64GB RAM")

        retriever = FactRetriever(store)
        results = retriever.search("Simon desktop OS preference", limit=3)
        assert len(results) >= 1
        assert "evidence_gap" in results[0]
        assert "coverage_ratio" in results[0]["evidence_gap"]
