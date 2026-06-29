#!/bin/bash
# Usage: bash scripts/run_livecodebench.sh <model_path|hf_repo_id> [out_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL=${1:?"usage: $0 <model_path|hf_repo_id> [out_dir]"}
OUT_DIR=${2:-"${REPO_ROOT}/benchmark/livecodebench"}
LCB_DIR=${LIVECODEBENCH_DIR:-"${REPO_ROOT}/third_party/LiveCodeBench"}

if [[ ! -d "${LCB_DIR}/lcb_runner" ]]; then
    [[ "${AUTO_SETUP_LCB:-1}" == "1" ]] || { echo "error: LiveCodeBench not found at ${LCB_DIR}"; exit 1; }
    mkdir -p "$(dirname "${LCB_DIR}")"
    git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git "${LCB_DIR}"
    python3 -m pip install -q -e "${LCB_DIR}"
fi
python3 "${SCRIPT_DIR}/patch_livecodebench.py" "${LCB_DIR}" >/dev/null 2>&1 || true

LCB_MODEL="${LCB_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/lcb_$(echo "${MODEL}" | tr '/ ' '__').log"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "${LCB_DIR}"
python -m lcb_runner.runner.main \
    --model "${LCB_MODEL}" \
    --local_model_path "${MODEL}" \
    --trust_remote_code \
    --scenario codegeneration \
    --evaluate \
    --release_version "${LCB_RELEASE:-release_latest}" \
    --n "${LCB_N:-10}" \
    --temperature "${LCB_TEMP:-0.2}" \
    --tensor_parallel_size "${TP:-$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)}" \
    --dtype "${DTYPE:-bfloat16}" \
    2>&1 | tee "${LOG}"
