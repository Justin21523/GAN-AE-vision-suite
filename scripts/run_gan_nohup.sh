#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-}"
if [[ -z "${CONFIG}" ]]; then
  echo "Usage: $0 <config.yaml> [run_dir]"
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${2:-/mnt/data/datasets/gan-ae-vision-suite/logs/nohup_gan_${TS}}"

mkdir -p "${RUN_DIR}"

CONDA_ENV="${CONDA_ENV:-ai_env}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  CMD=("${PYTHON_BIN}" -u -m src.scripts.train_gan --config "${CONFIG}" --run-dir "${RUN_DIR}")
elif command -v conda >/dev/null 2>&1; then
  CMD=(conda run -n "${CONDA_ENV}" python -u -m src.scripts.train_gan --config "${CONFIG}" --run-dir "${RUN_DIR}")
elif [[ -x "/home/justin/miniconda3/envs/${CONDA_ENV}/bin/python" ]]; then
  CMD=("/home/justin/miniconda3/envs/${CONDA_ENV}/bin/python" -u -m src.scripts.train_gan --config "${CONFIG}" --run-dir "${RUN_DIR}")
else
  CMD=(python -u -m src.scripts.train_gan --config "${CONFIG}" --run-dir "${RUN_DIR}")
fi

echo "#!/usr/bin/env bash" > "${RUN_DIR}/cmd.sh"
printf "%q " "${CMD[@]}" >> "${RUN_DIR}/cmd.sh"
echo >> "${RUN_DIR}/cmd.sh"
chmod +x "${RUN_DIR}/cmd.sh"

echo "Starting GAN training with nohup..."
echo "Run dir: ${RUN_DIR}"
echo "Cmd: ${CMD[*]}"

nohup "${CMD[@]}" > "${RUN_DIR}/stdout.log" 2>&1 &
PID=$!
echo "${PID}" > "${RUN_DIR}/pid.txt"

echo "PID: ${PID}"
echo "Watch stdout: tail -f ${RUN_DIR}/stdout.log"
echo "Run dir:      ls -la ${RUN_DIR}/gan_*"
echo "Watch logs:   tail -f ${RUN_DIR}/gan_*/*.log"
echo "Samples:      ls -la ${RUN_DIR}/gan_*/samples_*.png  (also ${RUN_DIR}/gan_*/samples_latest.png)"
echo "TensorBoard:  tensorboard --logdir ${RUN_DIR}/gan_*/tb --port 6006 --bind_all"
