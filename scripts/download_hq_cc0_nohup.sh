#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-umzi/AnimeSet}"
OUT_ROOT="${2:-/mnt/data/datasets/gan-ae-vision-suite/data/anime_hq_cc0}"
MAX_IMAGES="${3:-200000}"
RESIZE="${4:-512}"
FORMAT="${5:-jpg}"

TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="/mnt/data/datasets/gan-ae-vision-suite/logs/nohup_download_${TS}"
mkdir -p "${RUN_DIR}"

CONDA_ENV="${CONDA_ENV:-ai_env}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY=("${PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1; then
  PY=(conda run -n "${CONDA_ENV}" python)
elif [[ -x "/home/justin/miniconda3/envs/${CONDA_ENV}/bin/python" ]]; then
  PY=("/home/justin/miniconda3/envs/${CONDA_ENV}/bin/python")
else
  PY=(python)
fi

CMD=(
  "${PY[@]}" -u -m src.scripts.download_anime_hq_cc0
  --repo "${REPO}"
  --out-root "${OUT_ROOT}"
  --max-images "${MAX_IMAGES}"
  --center-crop
  --resize "${RESIZE}"
  --format "${FORMAT}"
  --print-every 50
  --min-free-gb 20
)

echo "#!/usr/bin/env bash" > "${RUN_DIR}/cmd.sh"
printf "%q " "${CMD[@]}" >> "${RUN_DIR}/cmd.sh"
echo >> "${RUN_DIR}/cmd.sh"
chmod +x "${RUN_DIR}/cmd.sh"

echo "Starting download with nohup..."
echo "Run dir:  ${RUN_DIR}"
echo "Repo:     ${REPO}"
echo "Out root: ${OUT_ROOT}"
echo "Max:      ${MAX_IMAGES}"
echo "Resize:   ${RESIZE}"
echo "Format:   ${FORMAT}"
echo "Cmd:      ${CMD[*]}"

nohup "${CMD[@]}" > "${RUN_DIR}/stdout.log" 2>&1 &
PID=$!
echo "${PID}" > "${RUN_DIR}/pid.txt"

echo "PID: ${PID}"
echo "Watch logs: tail -f ${RUN_DIR}/stdout.log"
echo "Progress:   ls -la ${OUT_ROOT}/images | tail"
echo "Disk:       df -h /mnt/data | tail -n 1"
