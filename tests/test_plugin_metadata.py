"""Tests for plugin.yaml and cli.py metadata.

These tests verify that the plugin conforms to the Hermes Agent
Memory Provider plugin spec. They are for development/debugging only
and are not included in the standalone pip install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Resolve the plugin root (works both in-repo and when installed)
_PLUGIN_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def plugin_root():
    return _PLUGIN_ROOT


@pytest.fixture
def plugin_yaml_path(plugin_root):
    return plugin_root / "plugin.yaml"


@pytest.fixture
def cli_py_path(plugin_root):
    return plugin_root / "cli.py"


@pytest.fixture
def plugin_yaml(plugin_yaml_path):
    """Load and parse plugin.yaml."""
    import yaml
    with open(plugin_yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# plugin.yaml tests
# ---------------------------------------------------------------------------

class TestPluginYaml:
    """Verify plugin.yaml exists and has required fields."""

    def test_plugin_yaml_exists(self, plugin_yaml_path):
        assert plugin_yaml_path.exists(), "plugin.yaml must exist"

    def test_plugin_yaml_parses(self, plugin_yaml):
        assert isinstance(plugin_yaml, dict), "plugin.yaml must be a YAML mapping"

    def test_name_present(self, plugin_yaml):
        assert "name" in plugin_yaml, "plugin.yaml must have 'name' field"
        assert plugin_yaml["name"] == "mnemoss"

    def test_version_present(self, plugin_yaml):
        assert "version" in plugin_yaml, "plugin.yaml must have 'version' field"
        assert isinstance(plugin_yaml["version"], str)

    def test_description_present(self, plugin_yaml):
        assert "description" in plugin_yaml, "plugin.yaml must have 'description' field"
        assert len(plugin_yaml["description"].strip()) > 10

    def test_hooks_present(self, plugin_yaml):
        assert "hooks" in plugin_yaml, "plugin.yaml must have 'hooks' field"
        assert isinstance(plugin_yaml["hooks"], list)
        assert len(plugin_yaml["hooks"]) > 0

    def test_hooks_are_valid(self, plugin_yaml):
        """All listed hooks must be actual MemoryProvider hook names."""
        valid_hooks = {
            "prefetch",
            "queue_prefetch",
            "sync_turn",
            "on_session_end",
            "on_pre_compress",
            "on_memory_write",
            "on_turn_start",
            "on_session_switch",
            "on_delegation",
            "system_prompt_block",
            "shutdown",
        }
        for hook in plugin_yaml["hooks"]:
            assert hook in valid_hooks, f"Unknown hook: {hook}"

    def test_hooks_match_implementation(self, plugin_root):
        """Hooks listed in plugin.yaml should be implemented in __init__.py."""
        init_path = plugin_root / "__init__.py"
        source = init_path.read_text(encoding="utf-8")

        with open(plugin_root / "plugin.yaml", encoding="utf-8") as f:
            import yaml
            hooks = yaml.safe_load(f)["hooks"]

        for hook in hooks:
            assert (
                f"def {hook}" in source
            ), f"Hook '{hook}' is listed in plugin.yaml but not implemented in __init__.py"


# ---------------------------------------------------------------------------
# cli.py tests
# ---------------------------------------------------------------------------

class TestCliPy:
    """Verify cli.py exists and has correct structure."""

    def test_cli_py_exists(self, cli_py_path):
        assert cli_py_path.exists(), "cli.py must exist"

    def test_register_cli_function_exists(self, cli_py_path):
        """cli.py must define register_cli(subparser)."""
        source = cli_py_path.read_text(encoding="utf-8")
        assert "def register_cli" in source, "cli.py must define register_cli function"
        assert "subparser" in source, "register_cli must accept subparser argument"

    def test_register_cli_callable(self, plugin_root):
        """register_cli must be callable."""
        # Add plugin root to path so relative imports work
        sys.path.insert(0, str(plugin_root))
        try:
            import cli as mnemoss_cli
        except ImportError:
            # cli.py imports hermes_constants which isn't available in
            # standalone test env — that's fine, we just check source.
            pytest.skip("hermes_constants not available (development env)")
        finally:
            sys.path.remove(str(plugin_root))

        assert hasattr(mnemoss_cli, "register_cli")
        assert callable(mnemoss_cli.register_cli)

    def test_setup_command_handler_exists(self, cli_py_path):
        source = cli_py_path.read_text(encoding="utf-8")
        assert "def cmd_setup" in source, "cli.py must define cmd_setup"

    def test_status_command_handler_exists(self, cli_py_path):
        source = cli_py_path.read_text(encoding="utf-8")
        assert "def cmd_status" in source, "cli.py must define cmd_status"

    def test_backfill_command_handler_exists(self, cli_py_path):
        source = cli_py_path.read_text(encoding="utf-8")
        assert "def cmd_backfill" in source, "cli.py must define cmd_backfill"

    def test_migrate_command_handlers_exist(self, cli_py_path):
        source = cli_py_path.read_text(encoding="utf-8")
        assert "def cmd_migrate_list" in source, "cli.py must define cmd_migrate_list"
        assert "def cmd_migrate_run" in source, "cli.py must define cmd_migrate_run"


# ---------------------------------------------------------------------------
# __init__.py consistency tests
# ---------------------------------------------------------------------------

class TestInitConsistency:
    """Verify __init__.py has the right structure."""

    def test_version_matches_plugin_yaml(self, plugin_root, plugin_yaml):
        init_path = plugin_root / "__init__.py"
        source = init_path.read_text(encoding="utf-8")

        # Extract __version__ from __init__.py
        import re
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', source)
        assert match, "__init__.py must define __version__"
        init_version = match.group(1)

        assert init_version == plugin_yaml["version"], (
            f"Version mismatch: __init__.py={init_version}, "
            f"plugin.yaml={plugin_yaml['version']}"
        )

    def test_register_function_exists(self, plugin_root):
        init_path = plugin_root / "__init__.py"
        source = init_path.read_text(encoding="utf-8")
        has_register = "def register(ctx" in source
        assert has_register, "__init__.py must define register(ctx) function"

    def test_memory_provider_class_exists(self, plugin_root):
        init_path = plugin_root / "__init__.py"
        source = init_path.read_text(encoding="utf-8")
        has_provider = "MemoryProvider" in source
        assert has_provider, "__init__.py must reference MemoryProvider"
