#!/bin/bash
# EvalPlus: HumanEval + HumanEval+ and MBPP + MBPP+ (pass@1, greedy).
# Usage: bash scripts/run_coding_evalplus.sh <model_path|hf_repo_id> [out_dir]
set -euo pipefail

MODEL=${1:?"usage: $0 <model_path|hf_repo_id> [out_dir]"}
OUT_DIR=${2:-"./benchmark/coding_evalplus"}
TP=${TP:-$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)}
DTYPE=${DTYPE:-float32}

if ! command -v evalplus.evaluate >/dev/null 2>&1; then
    echo "error: evalplus not found. Install with: pip install evalplus"
    exit 1
fi

SAFE_NAME="${MODEL//\//_}"
EXP_NAME="${EXP_NAME:-${SAFE_NAME}}"
mkdir -p "${OUT_DIR}"

for DS in humaneval mbpp; do
    LOG="${OUT_DIR}/${EXP_NAME}_${DS}.log"
    echo "=== EvalPlus ${DS} (base + plus) ==="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} evalplus.evaluate \
        --model "${MODEL}" \
        --dataset "${DS}" \
        --backend vllm \
        --tp "${TP}" \
        --dtype "${DTYPE}" \
        --greedy \
        --trust_remote_code \
        --root "${OUT_DIR}/${EXP_NAME}_${DS}" \
        2>&1 | tee "${LOG}"

    if [[ -d "evalplus_results/${DS}" ]]; then
        mkdir -p "${OUT_DIR}/${EXP_NAME}_${DS}/artifacts"
        cp -r "evalplus_results/${DS}/"* "${OUT_DIR}/${EXP_NAME}_${DS}/artifacts/" 2>/dev/null || true
    fi
done

echo "done. results in ${OUT_DIR}"
