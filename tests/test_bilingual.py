"""
Comprehensive bilingual search test suite.
Run directly: python tests/test_bilingual.py
"""

import os
import sys
import tempfile

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from store import MemoryStore
from retrieval import FactRetriever


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✅ {name}")

    def fail(self, name, reason=""):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ❌ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print(f"{'='*60}")
        return self.failed == 0


def make_store():
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    store = MemoryStore(db_path=db_path)
    return store, db_fd, db_path


def cleanup(store, db_fd, db_path):
    store.close()
    os.close(db_fd)
    os.unlink(db_path)


def seed_bilingual_data(store):
    """Seed a diverse bilingual fact store."""
    facts = [
        # Pure English
        ("Python is a popular programming language for data science", "tech", "python,programming"),
        ("Redis is an in-memory data structure store used as cache", "tech", "redis,cache"),
        ("Docker containers provide isolated runtime environments", "devops", "docker,containers"),
        ("Kubernetes orchestrates containerized applications at scale", "devops", "kubernetes,orchestration"),
        ("PostgreSQL is a powerful open-source relational database", "tech", "postgresql,database"),
        ("Rust is a systems programming language focused on safety", "tech", "rust,programming"),
        ("GraphQL is a query language for APIs", "tech", "graphql,api"),
        ("Machine learning models require training data", "ml", "ml,training"),
        # Pure Chinese
        ("Python是一门流行的编程语言，广泛用于数据科学", "tech", "python,编程"),
        ("Redis是一个内存数据结构存储，常用作缓存", "tech", "redis,缓存"),
        ("Docker容器提供隔离的运行时环境", "devops", "docker,容器"),
        ("Kubernetes用于大规模编排容器化应用", "devops", "kubernetes,编排"),
        ("PostgreSQL是一个强大的开源关系型数据库", "tech", "postgresql,数据库"),
        ("Rust是一门注重安全的系统编程语言", "tech", "rust,编程"),
        ("GraphQL是一种API查询语言", "tech", "graphql,api"),
        ("机器学习模型需要训练数据才能工作", "ml", "ml,训练"),
        # Mixed Chinese-English
        ("Redis缓存策略包括LRU和LFU两种算法", "tech", "redis,缓存,lru,lfu"),
        ("Docker容器使用namespace和cgroup实现隔离", "devops", "docker,namespace,cgroup"),
        ("Python的GIL限制了多线程并发性能", "tech", "python,gil,多线程"),
        ("Kubernetes的Pod是最小的调度单元", "devops", "kubernetes,pod,调度"),
        ("Rust的所有权系统避免了内存安全问题", "tech", "rust,所有权,内存安全"),
        ("PostgreSQL支持JSONB数据类型", "tech", "postgresql,jsonb"),
        ("TensorFlow和PyTorch是主流的深度学习框架", "ml", "tensorflow,pytorch,深度学习"),
        ("Git的rebase和merge有不同的使用场景", "tech", "git,rebase,merge"),
        # User facts (Chinese)
        ("龙哥喜欢使用Python和Rust进行开发", "user", "龙哥,python,rust"),
        ("龙哥的项目部署在Docker和Kubernetes上", "user", "龙哥,docker,kubernetes"),
        ("龙哥偏好直接和专注的回应风格", "user", "龙哥,偏好,沟通"),
        ("龙哥的服务器运行在WSL Ubuntu环境下", "user", "龙哥,wsl,ubuntu"),
        # User facts (English)
        ("Longge prefers Python and Rust for development", "user", "longge,python,rust"),
        ("Longge's projects run on Docker and Kubernetes", "user", "longge,docker,kubernetes"),
    ]
    for content, cat, tags in facts:
        store.add_fact(content, category=cat, tags=tags)


# ======== Test functions ========

def test_english_query_python(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("python programming", limit=5)
        assert len(results) > 0, "No results"
        contents = [res["content"].lower() for res in results]
        assert any("python" in c for c in contents), f"No 'python' in results: {contents}"
        r.ok("English query: python programming")
    except Exception as e:
        r.fail("English query: python programming", str(e))
    finally:
        cleanup(store, fd, path)


def test_english_query_docker(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("docker containers", limit=5)
        assert len(results) > 0, "No results"
        r.ok("English query: docker containers")
    except Exception as e:
        r.fail("English query: docker containers", str(e))
    finally:
        cleanup(store, fd, path)


def test_english_query_kubernetes(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("kubernetes orchestration", limit=5)
        assert len(results) > 0, "No results"
        r.ok("English query: kubernetes orchestration")
    except Exception as e:
        r.fail("English query: kubernetes orchestration", str(e))
    finally:
        cleanup(store, fd, path)


def test_english_query_graphql(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("graphql api", limit=5)
        assert len(results) > 0, "No results"
        r.ok("English query: graphql api")
    except Exception as e:
        r.fail("English query: graphql api", str(e))
    finally:
        cleanup(store, fd, path)


def test_english_query_ml(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("machine learning training", limit=5)
        assert len(results) > 0, "No results"
        r.ok("English query: machine learning training")
    except Exception as e:
        r.fail("English query: machine learning training", str(e))
    finally:
        cleanup(store, fd, path)


def test_chinese_query_python(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Python编程", limit=5)
        assert len(results) > 0, f"No results for 'Python编程'"
        r.ok("Chinese query: Python编程")
    except Exception as e:
        r.fail("Chinese query: Python编程", str(e))
    finally:
        cleanup(store, fd, path)


def test_chinese_query_docker(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Docker容器", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Chinese query: Docker容器")
    except Exception as e:
        r.fail("Chinese query: Docker容器", str(e))
    finally:
        cleanup(store, fd, path)


def test_chinese_query_redis_cache(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Redis缓存", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Chinese query: Redis缓存")
    except Exception as e:
        r.fail("Chinese query: Redis缓存", str(e))
    finally:
        cleanup(store, fd, path)


def test_chinese_query_龙哥(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("龙哥", limit=5)
        assert len(results) > 0, "No results"
        contents = [res["content"] for res in results]
        assert any("龙哥" in c for c in contents), f"No '龙哥' in results"
        r.ok("Chinese query: 龙哥")
    except Exception as e:
        r.fail("Chinese query: 龙哥", str(e))
    finally:
        cleanup(store, fd, path)


def test_chinese_query_机器学习(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("机器学习", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Chinese query: 机器学习")
    except Exception as e:
        r.fail("Chinese query: 机器学习", str(e))
    finally:
        cleanup(store, fd, path)


def test_chinese_query_数据库(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("数据库", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Chinese query: 数据库")
    except Exception as e:
        r.fail("Chinese query: 数据库", str(e))
    finally:
        cleanup(store, fd, path)


def test_chinese_query_编程语言(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("编程语言", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Chinese query: 编程语言")
    except Exception as e:
        r.fail("Chinese query: 编程语言", str(e))
    finally:
        cleanup(store, fd, path)


def test_mixed_query_redis_lru(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Redis LRU", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Mixed query: Redis LRU")
    except Exception as e:
        r.fail("Mixed query: Redis LRU", str(e))
    finally:
        cleanup(store, fd, path)


def test_mixed_query_docker_namespace(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Docker namespace", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Mixed query: Docker namespace")
    except Exception as e:
        r.fail("Mixed query: Docker namespace", str(e))
    finally:
        cleanup(store, fd, path)


def test_mixed_query_python_gil(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Python GIL", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Mixed query: Python GIL")
    except Exception as e:
        r.fail("Mixed query: Python GIL", str(e))
    finally:
        cleanup(store, fd, path)


def test_mixed_query_kubernetes_pod(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Kubernetes Pod", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Mixed query: Kubernetes Pod")
    except Exception as e:
        r.fail("Mixed query: Kubernetes Pod", str(e))
    finally:
        cleanup(store, fd, path)


def test_mixed_query_postgresql_jsonb(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("PostgreSQL JSONB", limit=5)
        assert len(results) > 0, "No results"
        r.ok("Mixed query: PostgreSQL JSONB")
    except Exception as e:
        r.fail("Mixed query: PostgreSQL JSONB", str(e))
    finally:
        cleanup(store, fd, path)


def test_empty_query(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("", limit=5)
        assert len(results) == 0, f"Expected 0 results, got {len(results)}"
        r.ok("Empty query")
    except Exception as e:
        r.fail("Empty query", str(e))
    finally:
        cleanup(store, fd, path)


def test_nonexistent_query(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("xyzabc123不存在", limit=5)
        assert isinstance(results, list), "Should return list"
        r.ok("Non-existent query")
    except Exception as e:
        r.fail("Non-existent query", str(e))
    finally:
        cleanup(store, fd, path)


def test_category_filter_english(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("programming", category="tech", limit=5)
        assert len(results) > 0, "No results"
        for res in results:
            assert res["category"] == "tech", f"Wrong category: {res['category']}"
        r.ok("Category filter (English): tech")
    except Exception as e:
        r.fail("Category filter (English): tech", str(e))
    finally:
        cleanup(store, fd, path)


def test_category_filter_chinese(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("编程", category="tech", limit=5)
        assert len(results) > 0, "No results"
        for res in results:
            assert res["category"] == "tech", f"Wrong category: {res['category']}"
        r.ok("Category filter (Chinese): tech")
    except Exception as e:
        r.fail("Category filter (Chinese): tech", str(e))
    finally:
        cleanup(store, fd, path)


def test_category_filter_mixed(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.search("Docker namespace", category="devops", limit=5)
        assert len(results) > 0, "No results"
        for res in results:
            assert res["category"] == "devops", f"Wrong category: {res['category']}"
        r.ok("Category filter (Mixed): devops")
    except Exception as e:
        r.fail("Category filter (Mixed): devops", str(e))
    finally:
        cleanup(store, fd, path)


def test_tokenize_english(r):
    try:
        tokens = FactRetriever._tokenize("hello world foo bar")
        assert "hello" in tokens, f"'hello' not in {tokens}"
        assert "world" in tokens, f"'world' not in {tokens}"
        r.ok("Tokenize English")
    except Exception as e:
        r.fail("Tokenize English", str(e))


def test_tokenize_chinese(r):
    try:
        tokens = FactRetriever._tokenize("Python是一门流行的编程语言")
        assert len(tokens) > 0, "No tokens from Chinese text"
        r.ok("Tokenize Chinese")
    except Exception as e:
        r.fail("Tokenize Chinese", str(e))


def test_tokenize_mixed(r):
    try:
        tokens = FactRetriever._tokenize("Redis缓存策略")
        assert len(tokens) > 0, "No tokens from mixed text"
        token_str = " ".join(tokens)
        assert "Redis" in token_str, f"'Redis' not in tokens: {tokens}"
        r.ok("Tokenize Mixed Chinese-English")
    except Exception as e:
        r.fail("Tokenize Mixed Chinese-English", str(e))


def test_tokenize_empty(r):
    try:
        tokens = FactRetriever._tokenize("")
        assert len(tokens) == 0, f"Expected 0 tokens, got {len(tokens)}"
        r.ok("Tokenize Empty")
    except Exception as e:
        r.fail("Tokenize Empty", str(e))


def test_has_chinese(r):
    try:
        assert FactRetriever._has_chinese("你好世界") == True
        assert FactRetriever._has_chinese("Redis缓存") == True
        assert FactRetriever._has_chinese("hello world") == False
        assert FactRetriever._has_chinese("") == False
        r.ok("Has Chinese detection")
    except Exception as e:
        r.fail("Has Chinese detection", str(e))


def test_extract_english_tokens(r):
    try:
        tokens = FactRetriever._extract_english_tokens("Redis缓存策略 Python编程")
        assert "redis" in tokens, f"'redis' not in {tokens}"
        assert "python" in tokens, f"'python' not in {tokens}"
        r.ok("Extract English tokens from mixed text")
    except Exception as e:
        r.fail("Extract English tokens from mixed text", str(e))


def test_probe_entity(r):
    store, fd, path = make_store()
    try:
        seed_bilingual_data(store)
        retriever = FactRetriever(store)
        results = retriever.probe("龙哥", limit=5)
        assert len(results) > 0, "No probe results for '龙哥'"
        r.ok("Probe entity: 龙哥")
    except Exception as e:
        r.fail("Probe entity: 龙哥", str(e))
    finally:
        cleanup(store, fd, path)


def test_smart_routing_english_fts5(r):
    """Verify English query uses FTS5 path."""
    store, fd, path = make_store()
    try:
        store.add_fact("Python is great for data science", "tech")
        store.add_fact("Python是一门流行的编程语言", "tech")
        retriever = FactRetriever(store)
        results = retriever.search("Python data", limit=5)
        assert len(results) > 0, "No results"
        contents = [res["content"] for res in results]
        assert any("Python is great" in c for c in contents), f"Expected English fact, got: {contents}"
        r.ok("Smart routing: English → FTS5")
    except Exception as e:
        r.fail("Smart routing: English → FTS5", str(e))
    finally:
        cleanup(store, fd, path)


def test_smart_routing_chinese_like(r):
    """Verify Chinese query uses LIKE path."""
    store, fd, path = make_store()
    try:
        store.add_fact("Python is great for data science", "tech")
        store.add_fact("Python是一门流行的编程语言", "tech")
        retriever = FactRetriever(store)
        results = retriever.search("编程语言", limit=5)
        assert len(results) > 0, "No results"
        contents = [res["content"] for res in results]
        assert any("编程语言" in c for c in contents), f"Expected Chinese fact, got: {contents}"
        r.ok("Smart routing: Chinese → LIKE")
    except Exception as e:
        r.fail("Smart routing: Chinese → LIKE", str(e))
    finally:
        cleanup(store, fd, path)


def test_smart_routing_mixed_merge(r):
    """Verify mixed query merges FTS5 + LIKE results."""
    store, fd, path = make_store()
    try:
        store.add_fact("Python is great for data science", "tech")
        store.add_fact("Python是一门流行的编程语言", "tech")
        store.add_fact("Redis缓存策略包括LRU", "tech")
        retriever = FactRetriever(store)
        results = retriever.search("Redis缓存", limit=5)
        assert len(results) > 0, "No results"
        contents = [res["content"] for res in results]
        assert any("Redis" in c or "缓存" in c for c in contents), f"Expected Redis/缓存 fact, got: {contents}"
        r.ok("Smart routing: Mixed → Merge FTS5 + LIKE")
    except Exception as e:
        r.fail("Smart routing: Mixed → Merge FTS5 + LIKE", str(e))
    finally:
        cleanup(store, fd, path)


# ======== Run all tests ========

ALL_TESTS = [
    # English queries (FTS5 path)
    test_english_query_python,
    test_english_query_docker,
    test_english_query_kubernetes,
    test_english_query_graphql,
    test_english_query_ml,
    # Chinese queries (LIKE path)
    test_chinese_query_python,
    test_chinese_query_docker,
    test_chinese_query_redis_cache,
    test_chinese_query_龙哥,
    test_chinese_query_机器学习,
    test_chinese_query_数据库,
    test_chinese_query_编程语言,
    # Mixed queries (merge path)
    test_mixed_query_redis_lru,
    test_mixed_query_docker_namespace,
    test_mixed_query_python_gil,
    test_mixed_query_kubernetes_pod,
    test_mixed_query_postgresql_jsonb,
    # Edge cases
    test_empty_query,
    test_nonexistent_query,
    # Category filter
    test_category_filter_english,
    test_category_filter_chinese,
    test_category_filter_mixed,
    # Tokenization
    test_tokenize_english,
    test_tokenize_chinese,
    test_tokenize_mixed,
    test_tokenize_empty,
    test_has_chinese,
    test_extract_english_tokens,
    # Probe
    test_probe_entity,
    # Smart routing
    test_smart_routing_english_fts5,
    test_smart_routing_chinese_like,
    test_smart_routing_mixed_merge,
]


if __name__ == "__main__":
    r = TestResult()
    print("🧪 Bilingual Search Optimization Test Suite")
    print("=" * 60)

    categories = [
        ("📝 English Queries (FTS5 Path)", [
            test_english_query_python, test_english_query_docker,
            test_english_query_kubernetes, test_english_query_graphql,
            test_english_query_ml,
        ]),
        ("📝 Chinese Queries (LIKE Path)", [
            test_chinese_query_python, test_chinese_query_docker,
            test_chinese_query_redis_cache, test_chinese_query_龙哥,
            test_chinese_query_机器学习, test_chinese_query_数据库,
            test_chinese_query_编程语言,
        ]),
        ("📝 Mixed Queries (Merge Path)", [
            test_mixed_query_redis_lru, test_mixed_query_docker_namespace,
            test_mixed_query_python_gil, test_mixed_query_kubernetes_pod,
            test_mixed_query_postgresql_jsonb,
        ]),
        ("📝 Edge Cases", [
            test_empty_query, test_nonexistent_query,
        ]),
        ("📝 Category Filter", [
            test_category_filter_english, test_category_filter_chinese,
            test_category_filter_mixed,
        ]),
        ("📝 Tokenization", [
            test_tokenize_english, test_tokenize_chinese,
            test_tokenize_mixed, test_tokenize_empty,
            test_has_chinese, test_extract_english_tokens,
        ]),
        ("📝 Probe", [
            test_probe_entity,
        ]),
        ("📝 Smart Routing", [
            test_smart_routing_english_fts5,
            test_smart_routing_chinese_like,
            test_smart_routing_mixed_merge,
        ]),
    ]

    for cat_name, tests in categories:
        print(f"\n{cat_name}")
        for t in tests:
            t(r)

    success = r.summary()
    sys.exit(0 if success else 1)
