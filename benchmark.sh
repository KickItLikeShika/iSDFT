set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_TASKS_DIR="${SCRIPT_DIR}/tasks"
DEFAULT_MODEL="KickItLikeShika/qwen2.5-7b-science-rholinear-Mu2e-3-200steps"

MODEL_ARG=${1:-${DEFAULT_MODEL}}
OUT_ARG=${2:-}

DTYPE=${DTYPE:-float32}
BATCH_SIZE=${BATCH_SIZE:-auto}
TASKS=${TASKS:-hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval}
GEN_KWARGS=${GEN_KWARGS:-max_gen_toks=4096}

if [[ -d "${MODEL_ARG}" ]]; then
    CHECKPOINT="$(cd "${MODEL_ARG}" && pwd)"
    OUT_DIR=${OUT_ARG:-"${CHECKPOINT}/benchmark"}
    OUT_DIR="$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"
else
    CHECKPOINT="${MODEL_ARG}"
    SAFE_NAME="${CHECKPOINT//\//_}"
    OUT_DIR=${OUT_ARG:-"${SCRIPT_DIR}/benchmark_results/${SAFE_NAME}"}
    OUT_DIR="$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"
    echo "loading from Hugging Face hub: ${CHECKPOINT}"
fi

RUN_INFO="${OUT_DIR}/run_info.txt"

{
    echo "benchmark run"
    echo "started: $(date -Is)"
    echo "model: ${CHECKPOINT}"
    echo "output: ${OUT_DIR}"
    echo "dtype: ${DTYPE}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "tasks: ${TASKS}"
    echo "custom_tasks_dir: ${CUSTOM_TASKS_DIR}"
    echo "gen_kwargs: ${GEN_KWARGS}"
} | tee "${RUN_INFO}"


export HF_ALLOW_CODE_EVAL=1

lm_eval \
    --model hf \
    --model_args "pretrained=${CHECKPOINT},dtype=${DTYPE},trust_remote_code=True" \
    --apply_chat_template \
    --tasks "${TASKS}" \
    --include_path "${CUSTOM_TASKS_DIR}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUT_DIR}" \
    --confirm_run_unsafe_code \
    2>&1 | tee -a "${OUT_DIR}/lm_eval.log"
