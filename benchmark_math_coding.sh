set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL="KickItLikeShika/qwen2.5-7b-science-rholinear-Mu2e-3-200steps"
MODEL_ARG=${1:-${DEFAULT_MODEL}}

SUITE=${BENCHMARK_SUITE:-all}
DTYPE=${DTYPE:-bfloat16}
BATCH_SIZE=${BATCH_SIZE:-auto}
GEN_KWARGS=${GEN_KWARGS:-max_gen_toks=4096}
MAX_K=${MAX_K:-16}
LCB_MAX_TOKENS=${LCB_MAX_TOKENS:-4096}
MULTIPL_E_MAX_TOKENS=${MULTIPL_E_MAX_TOKENS:-4096}
EVALPLUS_MAX_NEW_TOKENS=${EVALPLUS_MAX_NEW_TOKENS:-4096}
MATH_PASSK_DATASETS=${MATH_PASSK_DATASETS:-"aime aime25 amc beyondaime"}
CUSTOM_TASKS_DIR="${SCRIPT_DIR}/tasks"
SKIP_LM_EVAL=${SKIP_LM_EVAL:-1}
SKIP_PASSK_MATH=${SKIP_PASSK_MATH:-0}
SKIP_EVALPLUS=${SKIP_EVALPLUS:-0}
SKIP_EXTERNAL=${SKIP_EXTERNAL:-0}
SKIP_MULTIPL_E=${SKIP_MULTIPL_E:-0}
SKIP_LIVECODEBENCH=${SKIP_LIVECODEBENCH:-0}
AUTO_SETUP_LCB=${AUTO_SETUP_LCB:-1}
LCB_MODEL_NAME=${LCB_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}

# Model / Output directory logic
if [[ -d "${MODEL_ARG}" ]]; then
    CHECKPOINT="$(cd "${MODEL_ARG}" && pwd)"
    OUT_DIR=${2:-"${CHECKPOINT}/benchmark_math_coding"}
    [[ -f "${CHECKPOINT}/config.json" ]] || { echo "error: no config.json in ${CHECKPOINT}"; exit 1; }
else
    [[ "${MODEL_ARG}" == ./* || "${MODEL_ARG}" == /* || "${MODEL_ARG}" == ~* ]] && { echo "error: path not found: ${MODEL_ARG}"; exit 1; }
    [[ "${MODEL_ARG}" == */* ]] || { echo "error: expected Hugging Face repo id like org/model-name, got: ${MODEL_ARG}"; exit 1; }
    CHECKPOINT="${MODEL_ARG}"
    SAFE_NAME="${CHECKPOINT//\//_}"
    OUT_DIR=${2:-"${SCRIPT_DIR}/benchmark_results/${SAFE_NAME}_math_coding"}
fi
OUT_DIR="$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"

EXP_NAME="${EXP_NAME:-${CHECKPOINT##*/}}"
EXP_NAME="${EXP_NAME:-${SAFE_NAME:-model}}"

RUN_INFO="${OUT_DIR}/run_info.txt"
{
    echo "math+coding benchmark run"
    echo "started: $(date -Is)"
    echo "model: ${CHECKPOINT}"
    echo "output: ${OUT_DIR}"
    echo "suite: ${SUITE}"
    echo "dtype: ${DTYPE}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "gen_kwargs: ${GEN_KWARGS}"
    echo "max_k: ${MAX_K}"
    echo "lcb_max_tokens: ${LCB_MAX_TOKENS}"
    echo "multipl_e_max_tokens: ${MULTIPL_E_MAX_TOKENS}"
} | tee "${RUN_INFO}"

export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="${SCRIPT_DIR}/scripts:${PYTHONPATH:-}"
if [[ -z "${TOKENIZER_PATH:-}" ]]; then
    TOKENIZER_PATH="$(python3 -c "from resolve_tokenizer import resolve_tokenizer; print(resolve_tokenizer('${CHECKPOINT}'))")"
    export TOKENIZER_PATH
fi
if [[ "${TOKENIZER_PATH}" != "${CHECKPOINT}" ]]; then
    echo "tokenizer: ${TOKENIZER_PATH} (fallback; model repo tokenizer is broken/missing)" | tee -a "${RUN_INFO}"
fi

run_lm_eval() {
    local tasks=$1
    local outdir="${OUT_DIR}/$2"
    mkdir -p "${outdir}"
    LM_EVAL_BACKEND="${LM_EVAL_BACKEND:-vllm}" \
    TP="${TP:-}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}" \
    DTYPE="${DTYPE}" BATCH_SIZE="${BATCH_SIZE}" GEN_KWARGS="${GEN_KWARGS}" TASKS="${tasks}" \
    bash "${SCRIPT_DIR}/scripts/run_fast_lm_eval_math.sh" "${CHECKPOINT}" "${outdir}"
}

run_math_passk() {
    [[ "${SKIP_PASSK_MATH}" == "1" ]] && { echo "skip pass@k math (SKIP_PASSK_MATH=1)"; return; }
    local math_out="${OUT_DIR}/math_passk"
    mkdir -p "${math_out}"
    export PYTHONPATH="${SCRIPT_DIR}/scripts:${PYTHONPATH:-}"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
    for ds in ${MATH_PASSK_DATASETS}; do
        echo ">>> math_eval.py dataset=${ds} max_k=${MAX_K}"
        python3 "${SCRIPT_DIR}/scripts/math_eval.py" \
            --model_path "${CHECKPOINT}" \
            --exp_name "${EXP_NAME}" \
            --dataset "${ds}" \
            --base_folder "${math_out}" \
            --gen_kwargs "${GEN_KWARGS}" \
            --max_k "${MAX_K}" \
            --dtype "${DTYPE}" \
            ${TP:+--tp "${TP}"} \
            2>&1 | tee -a "${math_out}/${ds}.log"
    done
}

wait_for_gpu() {
    local min_free_mb="${1:-45000}"
    local waited=0
    while true; do
        local free_mb
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ -n "${free_mb}" && "${free_mb}" -ge "${min_free_mb}" ]]; then
            echo "GPU free memory OK: ${free_mb} MiB (need >= ${min_free_mb})"
            return 0
        fi
        echo "waiting for GPU memory (${free_mb:-?} MiB free, need ${min_free_mb})..."
        sleep 30
        waited=$((waited + 30))
        [[ "${waited}" -ge 1800 ]] && { echo "warning: GPU still busy after 30m, continuing anyway"; return 0; }
    done
}

run_coding() {
    wait_for_gpu "${MIN_GPU_FREE_MB:-40000}"
    export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
    if [[ "${SKIP_EVALPLUS}" != "1" ]]; then
        EXP_NAME="${EXP_NAME}" DTYPE="${DTYPE}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
        VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
        EVALPLUS_MAX_NEW_TOKENS="${EVALPLUS_MAX_NEW_TOKENS}" \
        bash "${SCRIPT_DIR}/scripts/run_coding_evalplus.sh" \
            "${CHECKPOINT}" "${OUT_DIR}/coding_evalplus" \
            2>&1 | tee -a "${OUT_DIR}/coding_evalplus.log" || echo "warning: evalplus failed"
    else
        echo "skip EvalPlus (SKIP_EVALPLUS=1)"
    fi

    [[ "${SKIP_EXTERNAL}" == "1" && "${SKIP_LIVECODEBENCH}" == "1" && "${SKIP_MULTIPL_E}" == "1" ]] && {
        echo "skip LiveCodeBench + MultiPL-E (all external coding skipped)"
        return
    }

    if [[ "${SKIP_EXTERNAL}" != "1" && "${SKIP_LIVECODEBENCH}" != "1" ]]; then
        wait_for_gpu "${MIN_GPU_FREE_MB:-40000}"
        AUTO_SETUP_LCB="${AUTO_SETUP_LCB}" LCB_MODEL_NAME="${LCB_MODEL_NAME:-}" DTYPE="${DTYPE}" TP="${TP:-}" \
        LCB_MAX_TOKENS="${LCB_MAX_TOKENS}" VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
        bash "${SCRIPT_DIR}/scripts/run_livecodebench.sh" \
            "${CHECKPOINT}" "${OUT_DIR}/livecodebench" \
            2>&1 | tee -a "${OUT_DIR}/livecodebench.log" || echo "warning: LiveCodeBench failed"
    else
        echo "skip LiveCodeBench (SKIP_EXTERNAL=1 or SKIP_LIVECODEBENCH=1)"
    fi

    if [[ "${SKIP_EXTERNAL}" != "1" && "${SKIP_MULTIPL_E}" != "1" && -f "${SCRIPT_DIR}/scripts/run_multipl_e.sh" ]]; then
        wait_for_gpu "${MIN_GPU_FREE_MB:-40000}"
        EXP_NAME="${EXP_NAME}" DTYPE="${DTYPE}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
        VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
        MULTIPL_E_MAX_TOKENS="${MULTIPL_E_MAX_TOKENS}" \
        bash "${SCRIPT_DIR}/scripts/run_multipl_e.sh" \
            "${CHECKPOINT}" "${OUT_DIR}/multipl_e" \
            2>&1 | tee -a "${OUT_DIR}/multipl_e.log" || echo "warning: MultiPL-E failed"
    else
        echo "skip MultiPL-E (SKIP_EXTERNAL=1, SKIP_MULTIPL_E=1, or arm64 host)"
    fi
}

# Main suite logic (vLLM phases first; lm-eval uses HF backend and hoards GPU memory)
case "${SUITE}" in
    math)
        run_math_passk
        [[ "${SKIP_LM_EVAL}" != "1" ]] && run_lm_eval "${TASKS:-gsm8k_cot_zeroshot,minerva_math500}" "math_lm_eval" \
            || echo "skip lm-eval math (SKIP_LM_EVAL=1)"
        ;;
    coding)
        run_coding
        ;;
    all)
        run_math_passk
        run_coding
        [[ "${SKIP_LM_EVAL}" != "1" ]] && run_lm_eval "${TASKS:-gsm8k_cot_zeroshot,minerva_math500}" "math_lm_eval" \
            || echo "skip lm-eval math (SKIP_LM_EVAL=1)"
        ;;
    *)
        echo "error: unknown BENCHMARK_SUITE='${SUITE}' (use math|coding|all)"
        exit 1
        ;;
esac

echo "finished: $(date -Is)" | tee -a "${RUN_INFO}"
echo "done. ${OUT_DIR}"
