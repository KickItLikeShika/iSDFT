set -euo pipefail

MODEL_ARG=${1:?"usage: $0 <checkpoint_dir|hf_repo_id> [output_dir]"}

DTYPE=${DTYPE:-float32}
BATCH_SIZE=${BATCH_SIZE:-auto}
TASKS=${TASKS:-hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval}

if [[ -d "${MODEL_ARG}" ]]; then
    CHECKPOINT="$(cd "${MODEL_ARG}" && pwd)"
    OUT_DIR=${2:-"${CHECKPOINT}/benchmark"}
    OUT_DIR="$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"
    if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
        echo "error: no config.json in ${CHECKPOINT} — is this a valid transformers checkpoint?"
        exit 1
    fi
else
    if [[ "${MODEL_ARG}" == ./* ]] || [[ "${MODEL_ARG}" == /* ]] || [[ "${MODEL_ARG}" == ~* ]]; then
        echo "error: path not found: ${MODEL_ARG}"
        exit 1
    fi
    if [[ "${MODEL_ARG}" != */* ]]; then
        echo "error: expected Hugging Face repo id like org/model-name, got: ${MODEL_ARG}"
        exit 1
    fi
    CHECKPOINT="${MODEL_ARG}"
    SAFE_NAME="${CHECKPOINT//\//_}"
    OUT_DIR=${2:-"./benchmark_results/${SAFE_NAME}"}
    OUT_DIR="$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"
    echo "note: loading from Hugging Face hub (will download if not cached): ${CHECKPOINT}"
fi

if ! command -v lm_eval >/dev/null 2>&1; then
    echo "error: lm_eval not found. Install with: pip install lm-eval"
    exit 1
fi

RUN_INFO="${OUT_DIR}/run_info.txt"

{
    echo "=== benchmark run ==="
    echo "started: $(date -Is)"
    echo "model: ${CHECKPOINT}"
    echo "output: ${OUT_DIR}"
    echo "dtype: ${DTYPE}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "tasks: ${TASKS}"
} | tee "${RUN_INFO}"

echo ">>> ${RUN_INFO}"
echo

export HF_ALLOW_CODE_EVAL=1

lm_eval \
    --model hf \
    --model_args "pretrained=${CHECKPOINT},dtype=${DTYPE},trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUT_DIR}" \
    --confirm_run_unsafe_code \
    2>&1 | tee -a "${OUT_DIR}/lm_eval.log"

{
    echo "finished: $(date -Is)"
} | tee -a "${RUN_INFO}"

echo "done. ${OUT_DIR}"
