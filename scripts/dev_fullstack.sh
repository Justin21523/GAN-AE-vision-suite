#!/usr/bin/env bash
#
# One-command local dev launcher (FastAPI + React UI).
#
# What it does:
# - Starts the FastAPI server in the background (via `scripts/serve_api.sh`)
# - Starts the Vite dev server in the foreground (`gan-ui`)
# - Cleans up the API process when you stop the UI (Ctrl+C)
#
# Prerequisites:
# - Python deps installed: `python -m pip install -r requirements.txt`
# - Node deps installed:   `cd gan-ui && npm ci`
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

if ! command -v npm >/dev/null 2>&1; then
  echo "❌ npm not found. Please install Node.js first." >&2
  exit 1
fi

if [ ! -d "${REPO_ROOT}/gan-ui/node_modules" ]; then
  echo "❌ gan-ui/node_modules not found." >&2
  echo "Run: (cd gan-ui && npm ci)" >&2
  exit 1
fi

echo "Starting API in background..."
(
  cd "${REPO_ROOT}"
  HOST="${API_HOST}" PORT="${API_PORT}" "${REPO_ROOT}/scripts/serve_api.sh"
) &
API_PID="$!"

cleanup() {
  echo ""
  echo "Stopping API (pid=${API_PID})..."
  kill "${API_PID}" >/dev/null 2>&1 || true
  wait "${API_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "API: http://${API_HOST}:${API_PORT}"
echo "UI:  http://127.0.0.1:5173"
echo ""

cd "${REPO_ROOT}/gan-ui"
npm run dev
