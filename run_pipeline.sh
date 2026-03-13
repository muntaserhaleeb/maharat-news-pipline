#!/usr/bin/env bash
# Run the full Maharat news pipeline: extract → normalize → export
#
# Usage:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --base-url https://maharat.com
#   ./run_pipeline.sh --base-url https://maharat.com --no-body
#   ./run_pipeline.sh --input "input/MyFile.docx" --split-level 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse args ─────────────────────────────────────────────────────────────
BASE_URL=""
NO_BODY=""
INPUT_ARG=""
SPLIT_LEVEL=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --base-url)   BASE_URL="$2";    shift 2 ;;
    --no-body)    NO_BODY="--no-body"; shift ;;
    --input)      INPUT_ARG="--input $2"; shift 2 ;;
    --split-level) SPLIT_LEVEL="--split-level $2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

step() { echo -e "\n${CYAN}▶ $1${RESET}"; }
done_msg() { echo -e "${GREEN}✓ $1${RESET}"; }

# ── Run ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}Maharat News Pipeline${RESET}"
echo "────────────────────────────────────"

step "Stage 1 — extract_posts.py"
python3 scripts/extract_posts.py $INPUT_ARG $SPLIT_LEVEL
done_msg "Extraction complete"

step "Stage 2 — normalize_posts.py"
python3 scripts/normalize_posts.py
done_msg "Normalisation complete"

step "Stage 3 — export_feed.py"
BASE_URL_ARG=""
[[ -n "$BASE_URL" ]] && BASE_URL_ARG="--base-url $BASE_URL"
python3 scripts/export_feed.py $BASE_URL_ARG $NO_BODY
done_msg "Export complete"

echo -e "\n${BOLD}Pipeline finished.${RESET}"
