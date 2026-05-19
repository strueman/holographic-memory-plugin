"""CLI commands for Mnemoss memory plugin management.

Handles: hermes mnemoss setup | status | backfill | migrate
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.config import cfg_get


def register_cli(subparser) -> None:
    """Register mnemoss CLI subcommands."""
    parser = subparser.add_parser(
        "mnemoss",
        help="Manage Mnemoss structured memory plugin",
        description="Setup, status, backfill, and migration for Mnemoss memory.",
    )
    subparsers = parser.add_subparsers(dest="mnemoss_command")

    # -- setup --
    setup_parser = subparsers.add_parser(
        "setup",
        help="Interactive setup wizard for Mnemoss",
    )
    setup_parser.set_defaults(handler=cmd_setup)

    # -- status --
    status_parser = subparsers.add_parser(
        "status",
        help="Show Mnemoss status (db path, fact count, features)",
    )
    status_parser.set_defaults(handler=cmd_status)

    # -- backfill --
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Backfill embeddings for all facts",
    )
    backfill_parser.set_defaults(handler=cmd_backfill)

    # -- migrate --
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Schema migration helpers",
    )
    migrate_sub = migrate_parser.add_subparsers(dest="migrate_command")

    migrate_list_parser = migrate_sub.add_parser(
        "list",
        help="List available migrations",
    )
    migrate_list_parser.set_defaults(handler=cmd_migrate_list)

    migrate_run_parser = migrate_sub.add_parser(
        "run",
        help="Run a specific migration",
    )
    migrate_run_parser.add_argument(
        "migration",
        help="Migration name to run (e.g. 'backfill-embeddings')",
    )
    migrate_run_parser.set_defaults(handler=cmd_migrate_run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    return get_hermes_home() / "config.yaml"


def _read_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8-sig") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _write_config(cfg: dict) -> None:
    import yaml
    path = _config_path()
    path.write_text(
        yaml.dump(cfg, default_flow_style=False, encoding="utf-8", width=120).decode("utf-8"),
        encoding="utf-8",
    )


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    if secret:
        if sys.stdin.isatty():
            import getpass
            val = getpass.getpass(prompt="")
        else:
            val = sys.stdin.readline().strip()
    else:
        val = sys.stdin.readline().strip()
    return val or (default or "")


def _get_mnemoss_config() -> dict:
    """Read mnemoss config from config.yaml."""
    cfg = _read_config()
    return cfg_get(cfg, "plugins", "mnemoss", default={}) or {}


def _default_db_path() -> str:
    return str(get_hermes_home() / "memory_store.db")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup(args) -> None:
    """Interactive Mnemoss setup wizard."""
    cfg = _read_config()
    mnemoss_cfg = cfg.setdefault("plugins", {}).setdefault("mnemoss", {})

    print("\nMnemoss memory setup")
    print("=" * 40)
    print("  Mnemoss gives Hermes structured fact storage with")
    print("  entity resolution, trust scoring, and HRR retrieval.")
    print(f"  Config: {_config_path()}")
    print()

    # --- 1. Database path ---
    current_db = mnemoss_cfg.get("db_path", _default_db_path())
    new_db = _prompt("Database path", default=current_db)
    if new_db:
        mnemoss_cfg["db_path"] = new_db

    # --- 2. Auto-extract ---
    current_auto = str(mnemoss_cfg.get("auto_extract", False)).lower()
    print("\n  Auto-extract modes:")
    print("    true  -- extract facts from conversations at session end")
    print("    false -- manual fact storage only (default)")
    new_auto = _prompt("Auto-extract", default=current_auto)
    if new_auto:
        mnemoss_cfg["auto_extract"] = new_auto.lower() in {"true", "yes", "1"}

    # --- 3. Default trust ---
    current_trust = str(mnemoss_cfg.get("default_trust", 0.5))
    new_trust = _prompt("Default trust score", default=current_trust)
    if new_trust:
        try:
            val = float(new_trust)
            if 0.0 <= val <= 1.0:
                mnemoss_cfg["default_trust"] = val
            else:
                print("  Value out of range (0.0-1.0), keeping current.")
        except ValueError:
            print("  Invalid number, keeping current.")

    # --- 4. HRR dimension ---
    current_hrr = str(mnemoss_cfg.get("hrr_dim", 1024))
    new_hrr = _prompt("HRR vector dimensions", default=current_hrr)
    if new_hrr:
        try:
            val = int(new_hrr)
            if val >= 64:
                mnemoss_cfg["hrr_dim"] = val
            else:
                print("  Value too low (min 64), keeping current.")
        except ValueError:
            print("  Invalid number, keeping current.")

    # --- 5. Min trust threshold ---
    current_min = str(mnemoss_cfg.get("min_trust_threshold", 0.3))
    new_min = _prompt("Min trust threshold", default=current_min)
    if new_min:
        try:
            val = float(new_min)
            if 0.0 <= val <= 1.0:
                mnemoss_cfg["min_trust_threshold"] = val
            else:
                print("  Value out of range (0.0-1.0), keeping current.")
        except ValueError:
            print("  Invalid number, keeping current.")

    _write_config(cfg)
    print(f"\n  Saved to {_config_path()}")
    print(f"  Mnemoss is configured. Restart Hermes Agent to apply.\n")


def cmd_status(args) -> None:
    """Show Mnemoss status."""
    mnemoss_cfg = _get_mnemoss_config()
    db_path = mnemoss_cfg.get("db_path", _default_db_path())

    # Expand env vars in db path
    db_path = db_path.replace("$HERMES_HOME", str(get_hermes_home()))
    db_path = db_path.replace("${HERMES_HOME}", str(get_hermes_home()))

    print("\nMnemoss status")
    print("=" * 40)
    print(f"  Database: {db_path}")
    print(f"  Auto-extract: {mnemoss_cfg.get('auto_extract', False)}")
    print(f"  Default trust: {mnemoss_cfg.get('default_trust', 0.5)}")
    print(f"  Min trust: {mnemoss_cfg.get('min_trust_threshold', 0.3)}")
    print(f"  HRR dim: {mnemoss_cfg.get('hrr_dim', 1024)}")

    # Check database
    if not os.path.exists(db_path):
        print("\n  Database file does not exist yet.")
        return

    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fact count
        try:
            count = cursor.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            print(f"  Facts: {count}")
        except sqlite3.OperationalError:
            print("  Facts table: not created yet")

        # Feature availability
        try:
            vec_count = cursor.execute("SELECT COUNT(*) FROM facts_vec").fetchone()[0]
            print(f"  Vector embeddings: {vec_count} facts")
        except sqlite3.OperationalError:
            print("  Vector embeddings: table not created")

        try:
            hrr_count = cursor.execute(
                "SELECT COUNT(*) FROM facts WHERE hrr_vector IS NOT NULL"
            ).fetchone()[0]
            print(f"  HRR vectors: {hrr_count} facts")
        except sqlite3.OperationalError:
            print("  HRR vectors: not available")

        try:
            ep_count = cursor.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            print(f"  Episodes: {ep_count}")
        except sqlite3.OperationalError:
            print("  Episodes: table not created")

        conn.close()

        # Check optional deps
        try:
            import numpy
            print("  numpy: available")
        except ImportError:
            print("  numpy: NOT installed (HRR features disabled)")

        try:
            import sqlite_vec
            print("  sqlite-vec: available")
        except ImportError:
            print("  sqlite-vec: NOT installed (vector search disabled)")

        try:
            import onnxruntime
            print("  onnxruntime: available")
        except ImportError:
            print("  onnxruntime: NOT installed (embedding generation disabled)")

    except Exception as e:
        print(f"\n  Error reading database: {e}")


def cmd_backfill(args) -> None:
    """Backfill embeddings for all facts."""
    mnemoss_cfg = _get_mnemoss_config()
    db_path = mnemoss_cfg.get("db_path", _default_db_path())
    db_path = db_path.replace("$HERMES_HOME", str(get_hermes_home()))
    db_path = db_path.replace("${HERMES_HOME}", str(get_hermes_home()))

    if not os.path.exists(db_path):
        print(f"\n  Database not found at {db_path}. Run 'hermes mnemoss setup' first.")
        return

    try:
        from mnemoss.store import MemoryStore
    except ImportError:
        print("\n  Mnemoss not installed. Install with:")
        print("    pip install -e .  # from mnemoss repo")
        return

    try:
        store = MemoryStore(db_path=db_path)
        count = store.backfill_embeddings()
        print(f"\n  Backfilled {count} embeddings.")
    except Exception as e:
        print(f"\n  Backfill failed: {e}")


def cmd_migrate_list(args) -> None:
    """List available migrations."""
    print("\nAvailable Mnemoss migrations:")
    print("=" * 40)
    print("  backfill-embeddings  -- Generate embeddings for all facts")
    print("  migrate-embedding-dim -- Change embedding dimension (drops existing vec data)")
    print()


def cmd_migrate_run(args) -> None:
    """Run a specific migration."""
    migration = args.migration
    mnemoss_cfg = _get_mnemoss_config()
    db_path = mnemoss_cfg.get("db_path", _default_db_path())
    db_path = db_path.replace("$HERMES_HOME", str(get_hermes_home()))
    db_path = db_path.replace("${HERMES_HOME}", str(get_hermes_home()))

    if not os.path.exists(db_path):
        print(f"\n  Database not found at {db_path}.")
        return

    try:
        from mnemoss.store import MemoryStore
    except ImportError:
        print("\n  Mnemoss not installed. Install with:")
        print("    pip install -e .  # from mnemoss repo")
        return

    store = MemoryStore(db_path=db_path)

    if migration == "backfill-embeddings":
        count = store.backfill_embeddings()
        print(f"\n  Backfilled {count} embeddings.")

    elif migration == "migrate-embedding-dim":
        try:
            from mnemoss.embedding import EMBEDDING_DIM
            old_dim = EMBEDDING_DIM
            new_dim_str = _prompt("New embedding dimension", default=str(old_dim))
            new_dim = int(new_dim_str)
            store.migrate_embedding_dimension(new_dim)
            print(f"\n  Migrated from {old_dim}d to {new_dim}d.")
            print("  Run 'hermes mnemoss backfill' to regenerate embeddings.")
        except Exception as e:
            print(f"\n  Migration failed: {e}")

    else:
        print(f"\n  Unknown migration: {migration}")
        print("  Run 'hermes mnemoss migrate list' for available migrations.")
