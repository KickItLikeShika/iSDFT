# Math + coding retention benchmarks.
# Usage: bash benchmark_math_coding.sh <checkpoint_dir|hf_repo_id> [output_dir]
# Env vars: BENCHMARK_SUITE={math,coding,all} DTYPE BATCH_SIZE GEN_KWARGS MAX_K TP TASKS
#           MATH_PASSK_DATASETS SKIP_LM_EVAL SKIP_PASSK_MATH SKIP_EVALPLUS SKIP_EXTERNAL LCB_MODEL_NAME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ARG=${1:?"usage: $0 <checkpoint_dir|hf_repo_id> [output_dir]"}

SUITE=${BENCHMARK_SUITE:-all}
DTYPE=${DTYPE:-float32}
BATCH_SIZE=${BATCH_SIZE:-auto}
GEN_KWARGS=${GEN_KWARGS:-max_gen_toks=4096}
MAX_K=${MAX_K:-16}
MATH_PASSK_DATASETS=${MATH_PASSK_DATASETS:-"aime aime25 amc beyondaime"}
CUSTOM_TASKS_DIR="${SCRIPT_DIR}/tasks"
SKIP_LM_EVAL=${SKIP_LM_EVAL:-0}
SKIP_PASSK_MATH=${SKIP_PASSK_MATH:-0}
SKIP_EVALPLUS=${SKIP_EVALPLUS:-0}
SKIP_EXTERNAL=${SKIP_EXTERNAL:-0}
AUTO_SETUP_LCB=${AUTO_SETUP_LCB:-1}

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
} | tee "${RUN_INFO}"

export HF_ALLOW_CODE_EVAL=1

run_lm_eval() {
    local tasks=$1
    local outdir="${OUT_DIR}/$2"
    mkdir -p "${outdir}"
    echo ">>> lm_eval: tasks=${tasks} -> ${outdir}"
    lm_eval \
        --model hf \
        --model_args "pretrained=${CHECKPOINT},dtype=${DTYPE},trust_remote_code=True" \
        --apply_chat_template \
        --tasks "${tasks}" \
        --include_path "${CUSTOM_TASKS_DIR}" \
        --gen_kwargs "${GEN_KWARGS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${outdir}" \
        --confirm_run_unsafe_code \
        2>&1 | tee -a "${outdir}/lm_eval.log"
}

run_math_passk() {
    [[ "${SKIP_PASSK_MATH}" == "1" ]] && { echo "skip pass@k math (SKIP_PASSK_MATH=1)"; return; }
    local math_out="${OUT_DIR}/math_passk"
    mkdir -p "${math_out}"
    export PYTHONPATH="${SCRIPT_DIR}/scripts:${PYTHONPATH:-}"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
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

run_coding() {
    if [[ "${SKIP_EVALPLUS}" != "1" ]]; then
        EXP_NAME="${EXP_NAME}" DTYPE="${DTYPE}" bash "${SCRIPT_DIR}/scripts/run_coding_evalplus.sh" \
            "${CHECKPOINT}" "${OUT_DIR}/coding_evalplus" \
            2>&1 | tee -a "${OUT_DIR}/coding_evalplus.log" || echo "warning: evalplus failed"
    else
        echo "skip EvalPlus (SKIP_EVALPLUS=1)"
    fi

    [[ "${SKIP_EXTERNAL}" == "1" ]] && { echo "skip LiveCodeBench + MultiPL-E (SKIP_EXTERNAL=1)"; return; }

    AUTO_SETUP_LCB="${AUTO_SETUP_LCB}" LCB_MODEL_NAME="${LCB_MODEL_NAME:-}" DTYPE="${DTYPE}" TP="${TP:-}" \
    bash "${SCRIPT_DIR}/scripts/run_livecodebench.sh" \
        "${CHECKPOINT}" "${OUT_DIR}/livecodebench" \
        2>&1 | tee -a "${OUT_DIR}/livecodebench.log" || echo "warning: LiveCodeBench failed"

    [[ -f "${SCRIPT_DIR}/scripts/run_multipl_e.sh" ]] && \
        EXP_NAME="${EXP_NAME}" DTYPE="${DTYPE}" bash "${SCRIPT_DIR}/scripts/run_multipl_e.sh" \
            "${CHECKPOINT}" "${OUT_DIR}/multipl_e" \
            2>&1 | tee -a "${OUT_DIR}/multipl_e.log" || echo "warning: MultiPL-E failed"
}

# Main suite logic
case "${SUITE}" in
    math)
        [[ "${SKIP_LM_EVAL}" != "1" ]] && run_lm_eval "${TASKS:-gsm8k_cot_zeroshot,minerva_math500}" "math_lm_eval" \
            || echo "skip lm-eval math (SKIP_LM_EVAL=1)"
        run_math_passk
        ;;
    coding)
        run_coding
        ;;
    all)
        [[ "${SKIP_LM_EVAL}" != "1" ]] && run_lm_eval "${TASKS:-gsm8k_cot_zeroshot,minerva_math500}" "math_lm_eval" \
            || echo "skip lm-eval math (SKIP_LM_EVAL=1)"
        run_math_passk
        run_coding
        ;;
    *)
        echo "error: unknown BENCHMARK_SUITE='${SUITE}' (use math|coding|all)"
        exit 1
        ;;
esac

echo "finished: $(date -Is)" | tee -a "${RUN_INFO}"
echo "done. ${OUT_DIR}"
