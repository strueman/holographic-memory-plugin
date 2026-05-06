"""Pytest configuration for holographic-memory-plugin.

The root __init__.py imports from hermes-agent internals (agent.memory_provider)
which aren't available when running tests from the standalone plugin repo.
This conftest tells pytest to skip it during collection.
"""

import pathlib


def pytest_ignore_collect(collection_path: pathlib.Path) -> bool:
    """Skip the root __init__.py — it's a plugin entry point, not a test."""
    # Skip any __init__.py that is directly in the project root (same dir as conftest.py)
    conftest_dir = pathlib.Path(__file__).parent
    if collection_path.name == "__init__.py" and collection_path.parent == conftest_dir:
        return True
