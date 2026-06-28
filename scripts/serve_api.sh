#!/usr/bin/env bash
#
# Start the local FastAPI server.
#
# This script is intended for local development and demo usage:
# - Runs `uvicorn src.api.main:app` from the repository root
# - Optionally uses `AI_CACHE_ROOT` as an artifact directory (defaults to `./.ai_cache`)
#
# Usage:
#   scripts/serve_api.sh
#   HOST=127.0.0.1 PORT=8000 scripts/serve_api.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Optional "warehouse" directory for artifacts/logs. The minimal API does not require it,
# but keeping the directory is useful for consistency across scripts.
AI_CACHE_ROOT="${AI_CACHE_ROOT:-"${REPO_ROOT}/.ai_cache"}"
mkdir -p "${AI_CACHE_ROOT}"/{models,outputs,train,logs,metrics}

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "❌ uvicorn is not installed. Try: python -m pip install -r requirements.txt" >&2
  exit 1
fi

echo "Starting GAN-AE-VISION-SUITE API"
echo "  Repo root:     ${REPO_ROOT}"
echo "  AI_CACHE_ROOT: ${AI_CACHE_ROOT}"
echo "  Listening on:  http://${HOST}:${PORT}"

cd "${REPO_ROOT}"
exec uvicorn src.api.main:app --host "${HOST}" --port "${PORT}" --workers 1
