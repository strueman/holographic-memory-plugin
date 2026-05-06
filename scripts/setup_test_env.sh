#!/usr/bin/env bash
# setup_test_env.sh — Bootstrap a standalone test environment for the holographic memory plugin.
#
# Usage:
#   ./scripts/setup_test_env.sh [--venv .venv] [--system]
#
# --venv DIR    Create a virtual environment in DIR (default: .venv)
# --system      Use system Python instead of creating a venv
#
# After running, activate the venv:
#   source .venv/bin/activate
#   python3 -m pytest tests/ -v

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

VENV_DIR=""
USE_SYSTEM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --system)
            USE_SYSTEM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$VENV_DIR" && "$USE_SYSTEM" == false ]]; then
    VENV_DIR="$REPO_DIR/.venv"
fi

PYTHON="python3"

if [[ "$USE_SYSTEM" == false ]]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    PYTHON="$VENV_DIR/bin/python3"
    PIP="$VENV_DIR/bin/pip"
    echo "Activate with: source $VENV_DIR/bin/activate"
else
    PIP="pip3"
fi

echo ""
echo "Installing dependencies with $PIP ..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r "$REPO_DIR/requirements.txt" --quiet

echo ""
echo "Verifying dependencies ..."
"$PYTHON" -c "
import sqlite3
print('  sqlite3 (stdlib): OK')

try:
    import sqlite_vec
    print('  sqlite-vec: OK')
except ImportError:
    print('  sqlite-vec: MISSING (vec search disabled)')

try:
    import onnxruntime
    print('  onnxruntime: OK')
except ImportError:
    print('  onnxruntime: MISSING (vec search disabled)')

try:
    import tokenizers
    print('  tokenizers: OK')
except ImportError:
    print('  tokenizers: MISSING (vec search disabled)')

try:
    import numpy
    print('  numpy: OK')
except ImportError:
    print('  numpy: MISSING (HRR operations disabled)')

try:
    import pytest
    print('  pytest: OK')
except ImportError:
    print('  pytest: MISSING')
"

echo ""
echo "Downloading embedding model (if not cached) ..."
"$PYTHON" -c "
import sys, os
sys.path.insert(0, '$REPO_DIR')
from embedding import is_available
if is_available():
    print('  Embedding model: available')
else:
    print('  Embedding model: will auto-download on first use')
"

echo ""
echo "Running tests ..."
cd "$REPO_DIR"
"$PYTHON" -m pytest tests/ -v --tb=short

echo ""
echo "Done."
