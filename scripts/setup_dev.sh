#!/usr/bin/env bash
# =============================================================================
# AI Gym Trainer FYP — Idempotent dev-environment bootstrap (POSIX / bash)
#
# Usage:
#   chmod +x scripts/setup_dev.sh
#   ./scripts/setup_dev.sh
#
# Tested on: Ubuntu 22.04, macOS 13+, WSL2 (Ubuntu 22.04)
# =============================================================================
set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'

step()  { echo -e "\n${CYAN}==> $*${RESET}"; }
ok()    { echo -e "    ${GREEN}[OK]${RESET} $*"; }
warn()  { echo -e "    ${YELLOW}[WARN]${RESET} $*"; }
fail()  { echo -e "\n${RED}[FAIL]${RESET} $*" >&2; exit 1; }

# ── Resolve repo root (script lives in scripts/) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
echo -e "${BOLD}Repo root: $REPO_ROOT${RESET}"

# ── Step 1: Verify Python 3.11 ────────────────────────────────────────────────
step "Checking Python version"

PYTHON_CMD=""
for candidate in python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1)
        if echo "$ver" | grep -qE 'Python 3\.11\.[0-9]+'; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    fail "Python 3.11 was not found on PATH.

Install it via your package manager:
  Ubuntu/Debian : sudo apt install python3.11 python3.11-venv python3.11-dev
  macOS (brew)  : brew install python@3.11
  Windows/WSL   : https://www.python.org/downloads/release/python-3110/

Then re-run this script."
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version)
ok "Found: $PYTHON_VERSION  (command: $PYTHON_CMD)"

# ── Step 2: Create / activate .venv ──────────────────────────────────────────
step "Preparing virtual environment"

VENV_DIR="$REPO_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "    Creating .venv ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    ok ".venv created at $VENV_DIR"
else
    ok ".venv already exists — skipping creation"
fi

# Activate for the remainder of this script
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

PIP_CMD="$VENV_DIR/bin/pip"
[[ -x "$PIP_CMD" ]] || fail "pip not found at $PIP_CMD — delete .venv and re-run."

# ── Step 3: Upgrade pip + install requirements ────────────────────────────────
step "Installing Python dependencies"

echo "    Upgrading pip ..."
"$PIP_CMD" install --quiet --upgrade pip

REQ_FILE="$REPO_ROOT/requirements.txt"
[[ -f "$REQ_FILE" ]] || fail "requirements.txt not found at $REQ_FILE"

echo "    Installing requirements.txt ..."
"$PIP_CMD" install --quiet -r "$REQ_FILE"

ok "All Python dependencies installed"

# ── Step 4: Claude Code native binary ─────────────────────────────────────────
step "Checking Claude Code installation"

if command -v claude &>/dev/null; then
    CLAUDE_VER=$(claude --version 2>&1 || true)
    ok "Claude Code already installed: $CLAUDE_VER"
else
    warn "Claude Code not found on PATH — installing via official installer ..."
    if command -v curl &>/dev/null; then
        curl -fsSL https://claude.ai/install.sh | bash
        # Re-source shell profile so 'claude' is available in current session
        for profile in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile"; do
            # shellcheck source=/dev/null
            [[ -f "$profile" ]] && source "$profile" 2>/dev/null || true
        done
        if command -v claude &>/dev/null; then
            CLAUDE_VER=$(claude --version 2>&1 || true)
            ok "Claude Code installed: $CLAUDE_VER"
        else
            warn "Claude Code installer ran but 'claude' is not yet on PATH."
            warn "Open a new shell or run: source ~/.bashrc"
        fi
    else
        warn "'curl' is not available — cannot auto-install Claude Code."
        warn "Install manually: https://claude.ai/download"
    fi
fi

# ── Step 5: Success banner ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║          AI Gym Trainer FYP — Dev Environment Ready         ║${RESET}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "${BOLD}  Activate venv (if not already active):${RESET}"
echo -e "    ${YELLOW}source .venv/bin/activate${RESET}"
echo ""
echo -e "${BOLD}  Start the CV loop:${RESET}"
echo -e "    ${YELLOW}python main.py${RESET}"
echo ""
echo -e "${BOLD}  Start the REST API:${RESET}"
echo -e "    ${YELLOW}uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000${RESET}"
echo ""
echo -e "${BOLD}  Run tests:${RESET}"
echo -e "    ${YELLOW}pytest tests/ -v${RESET}"
echo ""
echo -e "${BOLD}  Train all 9 ML models:${RESET}"
echo -e "    ${YELLOW}python training/train_universal.py${RESET}"
echo ""
