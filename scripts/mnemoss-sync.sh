#!/bin/bash
# mnemoss-sync.sh — Sync Mnemoss plugin between repos
# Usage: mnemoss-sync.sh [check|pull|push|sync-to-agent|full]
#
# Workflow:
# 1. Work in ~/.hermes/mnemoss/ (source of truth)
# 2. When done: mnemoss-sync.sh full (pull → push → sync to agent)
# 3. Before starting work: mnemoss-sync.sh check (verify sync status)

set -euo pipefail

PLUGIN_DIR="$HOME/.hermes/mnemoss"
AGENT_DIR="$HOME/.hermes/hermes-agent/plugins/memory/mnemoss"
REPO_URL="https://github.com/strueman/mnemoss.git"

cd "$PLUGIN_DIR"

case "${1:-check}" in
  check)
    echo "=== Mnemoss Plugin Sync Status ==="
    echo "Plugin repo: $PLUGIN_DIR"
    echo "Agent dir:   $AGENT_DIR"
    echo ""
    # Local status
    local_head=$(git log --oneline -1 --format="%h %s")
    echo "Local HEAD:  $local_head"
    # Remote status
    git fetch --quiet 2>/dev/null
    remote_head=$(git log --oneline -1 --format="%h %s" origin/main 2>/dev/null || echo "(fetch failed)")
    echo "Remote HEAD: $remote_head"
    # Compare
    local_hash=$(git rev-parse HEAD)
    remote_hash=$(git rev-parse origin/main 2>/dev/null || echo "unknown")
    if [ "$local_hash" = "$remote_hash" ]; then
        echo ""
        echo "✓ Local and remote are in sync"
    else
        echo ""
        echo "⚠ Local and remote are OUT OF SYNC"
        echo "  Run 'mnemoss-sync.sh push' to push local changes"
        echo "  Run 'mnemoss-sync.sh pull' to pull remote changes"
    fi
    # Agent sync status
    echo ""
    if [ -f "$AGENT_DIR/store.py" ]; then
        agent_mod=$(stat -c %Y "$AGENT_DIR/store.py" 2>/dev/null || echo "0")
        plugin_mod=$(stat -c %Y "$PLUGIN_DIR/store.py" 2>/dev/null || echo "0")
        if [ "$agent_mod" = "$plugin_mod" ]; then
            echo "✓ Agent checkout is in sync with plugin repo"
        else
            echo "⚠ Agent checkout may be out of sync (plugin newer)"
            echo "  Run 'mnemoss-sync.sh sync-to-agent' to update"
        fi
    else
        echo "⚠ Agent checkout missing — run 'mnemoss-sync.sh sync-to-agent'"
    fi
    ;;

  pull)
    echo "=== Pulling from GitHub ==="
    git fetch origin
    git pull origin main --ff-only
    echo "Pulled successfully"
    ;;

  push)
    echo "=== Pushing to GitHub ==="
    # Check for uncommitted changes
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        echo "⚠ Uncommitted changes detected. Commit first."
        git status --short
        exit 1
    fi
    git fetch origin
    git push origin main
    echo "Pushed successfully"
    ;;

  sync-to-agent)
    echo "=== Syncing plugin → hermes-agent checkout ==="
    # Copy plugin source files to agent checkout
    # rsync pattern order matters: excludes before includes
    rsync -av \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='benchmarks' \
        --exclude='AGENTS.md' \
        --exclude='README.md' \
        --exclude='.gitignore' \
        --exclude='skills' \
        --include='*/' \
        --include='*.py' \
        --include='*.onnx' \
        --include='*.json' \
        --exclude='*' \
        "$PLUGIN_DIR/" "$AGENT_DIR/"
    echo "Synced to agent checkout"
    ;;

  full)
    echo "=== Full sync cycle ==="
    echo ""
    echo "Step 1: Pull from GitHub"
    git fetch origin
    git pull origin main --ff-only || true
    echo ""
    echo "Step 2: Push local changes"
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        echo "⚠ Uncommitted changes — skipping push"
    else
        git push origin main || true
    fi
    echo ""
    echo "Step 3: Sync to hermes-agent checkout"
    # rsync pattern order matters: excludes before includes
    rsync -av \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='benchmarks' \
        --exclude='AGENTS.md' \
        --exclude='README.md' \
        --exclude='.gitignore' \
        --exclude='skills' \
        --include='*/' \
        --include='*.py' \
        --include='*.onnx' \
        --include='*.json' \
        --exclude='*' \
        "$PLUGIN_DIR/" "$AGENT_DIR/"
    echo ""
    echo "=== Full sync complete ==="
    ;;

  *)
    echo "Usage: $0 [check|pull|push|sync-to-agent|full]"
    exit 1
    ;;
esac
