#!/usr/bin/env bash
# install.sh — Install Mnemoss memory plugin for Hermes Agent
# Usage: curl -fsSL https://raw.githubusercontent.com/strueman/mnemoss/main/install.sh | bash
#
# This script downloads the Mnemoss plugin files and installs them into
# your Hermes Agent profile(s).

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors and formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[info]${NC} $*"; }
ok()    { echo -e "${GREEN}[ ok ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[err ]${NC} $*"; }
header() { echo -e "\n${BOLD}${CYAN}$*${NC}"; }

# ---------------------------------------------------------------------------
# GitHub repo config
# ---------------------------------------------------------------------------
REPO="strueman/mnemoss"
BRANCH="main"
RAW_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}"

# Files to install (core plugin files only)
PLUGIN_FILES=(
    __init__.py
    store.py
    retrieval.py
    hrr.py
    embedding.py
    fact_splitter.py
    plugin.yaml
    cli.py
)

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
header "Mnemoss Memory Plugin Installer"
echo "  Structured fact storage with entity resolution, trust scoring,"
echo "  and HRR compositional retrieval for Hermes Agent."
echo ""

# Check for curl or wget
if command -v curl &>/dev/null; then
    DOWNLOADER="curl -fsSL"
elif command -v wget &>/dev/null; then
    DOWNLOADER="wget -qO-"
else
    error "Neither curl nor wget found. Please install one and try again."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Detect base path
# ---------------------------------------------------------------------------
header "Step 1/4 — Base Path"

DEFAULT_BASE="$HOME/.hermes"
read -rp "  Hermes Agent base path [$DEFAULT_BASE]: " INPUT_BASE
BASE="${INPUT_BASE:-$DEFAULT_BASE}"

if [[ ! -d "$BASE" ]]; then
    warn "Path $BASE does not exist. The installer will create it."
    read -rp "  Continue? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy] ]]; then
        info "Aborted."
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Detect profiles
# ---------------------------------------------------------------------------
header "Step 2/4 — Profile Detection"

PROFILES_DIR="$BASE/profiles"
DEFAULT_PROFILE="default"

if [[ -d "$PROFILES_DIR" ]]; then
    # List profile directories (non-hidden, non-empty)
    mapfile -t DETECTED_PROFILES < <(
        find "$PROFILES_DIR" -mindepth 1 -maxdepth 1 -type d -not -name '.*' 2>/dev/null | \
        while read -r dir; do
            basename "$dir"
        done | sort
    )
else
    DETECTED_PROFILES=()
fi

# Determine which profiles to install to
declare -a TARGET_PROFILES=()

if [[ ${#DETECTED_PROFILES[@]} -eq 0 ]]; then
    # No profiles detected — assume default profile only
    info "No profiles found in $PROFILES_DIR (or directory missing)."
    info "Installing to default profile: $DEFAULT_BASE"
    TARGET_PROFILES=("$DEFAULT_PROFILE")
else
    echo ""
    echo "  Detected profiles:"
    for i in "${!DETECTED_PROFILES[@]}"; do
        echo "    $((i + 1)). ${DETECTED_PROFILES[$i]}"
    done
    echo ""
    echo "  0. All profiles (${#DETECTED_PROFILES[@]})"
    echo "  ${#DETECTED_PROFILES[@]}+1. Specific profile..."
    echo ""

    read -rp "  Install to [0=all, ${#DETECTED_PROFILES[@]}+1=specific, default=default]: " PROFILE_CHOICE
    PROFILE_CHOICE="${PROFILE_CHOICE:-0}"

    if [[ "$PROFILE_CHOICE" == "0" ]]; then
        TARGET_PROFILES=("${DETECTED_PROFILES[@]}")
    elif [[ "$PROFILE_CHOICE" =~ ^[0-9]+$ ]] && [[ "$PROFILE_CHOICE" -gt 0 ]] && [[ "$PROFILE_CHOICE" -le ${#DETECTED_PROFILES[@]} ]]; then
        TARGET_PROFILES=("${DETECTED_PROFILES[$((PROFILE_CHOICE - 1))]}")
    elif [[ "$PROFILE_CHOICE" =~ ^[0-9]+$ ]] && [[ "$PROFILE_CHOICE" -gt ${#DETECTED_PROFILES[@]} ]]; then
        # Specific profile by name
        read -rp "  Enter profile name: " PROFILE_NAME
        if [[ -d "$PROFILES_DIR/$PROFILE_NAME" ]]; then
            TARGET_PROFILES=("$PROFILE_NAME")
        else
            error "Profile '$PROFILE_NAME' not found."
            exit 1
        fi
    else
        # Default — install to default profile
        TARGET_PROFILES=("$DEFAULT_PROFILE")
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Download plugin files
# ---------------------------------------------------------------------------
header "Step 3/4 — Downloading Plugin Files"

TMPDIR=$(mktemp -d)
trap "rm -rf '$TMPDIR'" EXIT

download_ok=true
for file in "${PLUGIN_FILES[@]}"; do
    url="${RAW_URL}/${file}"
    info "  Downloading $file ..."
    if ! $DOWNLOADER "$url" > "$TMPDIR/$file" 2>/dev/null; then
        error "  Failed to download $file from $url"
        download_ok=false
    fi
done

if [[ "$download_ok" != "true" ]]; then
    error "One or more files failed to download. Check your internet connection."
    exit 1
fi
ok "Downloaded ${#PLUGIN_FILES[@]} plugin files."

# ---------------------------------------------------------------------------
# Step 4: Install to target profiles
# ---------------------------------------------------------------------------
header "Step 4/4 — Installing to ${#TARGET_PROFILES[@]} profile(s)"

PLUGIN_DIR="memory/mnemoss"
installed_count=0

for profile in "${TARGET_PROFILES[@]}"; do
    if [[ "$profile" == "$DEFAULT_PROFILE" ]]; then
        install_dir="$BASE/$PLUGIN_DIR"
        label="default profile"
    else
        install_dir="$BASE/profiles/$profile/$PLUGIN_DIR"
        label="profile '$profile'"
    fi

    # Create directory
    mkdir -p "$install_dir"

    # Check if already installed
    if [[ -f "$install_dir/__init__.py" ]]; then
        warn "  $label: Mnemoss already installed at $install_dir"
        read -rp "  Overwrite? [y/N]: " overwrite
        if [[ ! "$overwrite" =~ ^[Yy] ]]; then
            info "  Skipping $label."
            continue
        fi
    fi

    # Copy files
    for file in "${PLUGIN_FILES[@]}"; do
        cp "$TMPDIR/$file" "$install_dir/$file"
    done

    ok "  Installed to $label ($install_dir)"
    installed_count=$((installed_count + 1))
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Installation Complete"

if [[ $installed_count -eq 0 ]]; then
    warn "No profiles were updated."
    exit 0
fi

echo ""
echo -e "  ${GREEN}Mnemoss is installed and ready to configure.${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Restart Hermes Agent (or run the setup command below)"
echo ""
echo "  2. Run the interactive setup wizard:"
echo ""
if [[ ${#TARGET_PROFILES[@]} -eq 1 ]] && [[ "${TARGET_PROFILES[0]}" == "$DEFAULT_PROFILE" ]]; then
    echo -e "     ${BOLD}hermes mnemoss setup${NC}"
else
    echo -e "     ${BOLD}hermes mnemoss setup${NC}  (for each profile)"
fi
echo ""
echo "  3. Select 'mnemoss' as your memory provider in config.yaml:"
echo ""
echo -e "     ${BOLD}memory:${NC}"
echo -e "     ${BOLD}  provider: mnemoss${NC}"
echo ""
echo "  4. Optional: backfill embeddings for existing facts:"
echo ""
echo -e "     ${BOLD}hermes mnemoss backfill${NC}"
echo ""
echo "  Run ${BOLD}hermes mnemoss status${NC} to verify the installation."
echo ""
