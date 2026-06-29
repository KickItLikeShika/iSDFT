#!/bin/bash
# Usage: bash scripts/run_multipl_e.sh <model_path|hf_repo_id> [out_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${1:?"usage: $0 <model_path|hf_repo_id> [out_dir]"}
OUT_DIR=${2:-"./benchmark/multipl_e"}
TP=${TP:-$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)}
LANGS=${MULTIPL_E_LANGS:-"cpp java php ts cs sh js"}
EXP_NAME="${EXP_NAME:-${MODEL//\//_}}"

command -v docker >/dev/null || { echo "error: docker required"; exit 1; }
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"
${DOCKER} info >/dev/null 2>&1 || { echo "error: cannot reach docker daemon"; exit 1; }

mkdir -p "${OUT_DIR}/gens" "${OUT_DIR}/results" "${OUT_DIR}/logs"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

for LANG in ${LANGS}; do
    GEN="${OUT_DIR}/gens/${EXP_NAME}_${LANG}"
    RES="${OUT_DIR}/results/${EXP_NAME}_${LANG}"
    [[ -d "${RES}" && -n "$(ls -A "${RES}" 2>/dev/null)" ]] && continue

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "${SCRIPT_DIR}/gen_multipl_e.py" \
        --model "${MODEL}" --lang "${LANG}" --out_dir "${GEN}" --tp "${TP}" --dtype "${DTYPE:-float32}" \
        2>&1 | tee "${OUT_DIR}/logs/${EXP_NAME}_${LANG}.log"

    ${DOCKER} run --rm \
        -v "${OUT_DIR}/gens:/g" -v "${OUT_DIR}/results:/r" \
        ghcr.io/nuprl/multipl-e-evaluation:latest \
        --dir "/g/${EXP_NAME}_${LANG}" --output-dir "/r/${EXP_NAME}_${LANG}" --max-workers 8
done
