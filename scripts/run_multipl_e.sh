set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${1:?"usage: $0 <model_path|hf_repo_id> [out_dir]"}
OUT_DIR=${2:-"./benchmark/multipl_e"}
TP=${TP:-$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)}
LANGS=${MULTIPL_E_LANGS:-"cpp java php ts cs sh js"}
EXP_NAME="${EXP_NAME:-${MODEL//\//_}}"
ARCH="$(uname -m)"
DOCKER_PLATFORM=()

if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    if [[ -f /proc/sys/fs/binfmt_misc/qemu-x86_64 ]] || command -v qemu-x86_64-static >/dev/null 2>&1; then
        DOCKER_PLATFORM=(--platform linux/amd64)
        echo "note: MultiPL-E eval uses amd64 docker image via qemu on ${ARCH}"
    elif [[ "${FORCE_MULTIPL_E:-0}" != "1" ]]; then
        echo "skip MultiPL-E: eval image is linux/amd64 only and qemu-user-static is not available."
        echo "  install: sudo apt-get install -y qemu-user-static binfmt-support"
        echo "  or set FORCE_MULTIPL_E=1 to try anyway."
        exit 0
    else
        DOCKER_PLATFORM=(--platform linux/amd64)
        echo "warning: running amd64 MultiPL-E container on ${ARCH} (qemu not detected)"
    fi
fi

command -v docker >/dev/null || { echo "error: docker required"; exit 1; }
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"
${DOCKER} info >/dev/null 2>&1 || { echo "error: cannot reach docker daemon"; exit 1; }

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"

echo "pulling MultiPL-E eval image (if needed)..."
${DOCKER} pull --platform linux/amd64 ghcr.io/nuprl/multipl-e-evaluation:latest >/dev/null 2>&1 \
    || echo "warning: docker pull failed; eval may fail if image is missing"

mkdir -p "${OUT_DIR}/gens" "${OUT_DIR}/results" "${OUT_DIR}/logs"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

for LANG in ${LANGS}; do
    GEN="${OUT_DIR}/gens/${EXP_NAME}_${LANG}"
    RES="${OUT_DIR}/results/${EXP_NAME}_${LANG}"
    [[ -d "${RES}" && -n "$(ls -A "${RES}" 2>/dev/null)" ]] && { echo "skip ${LANG}: results exist"; continue; }

    if [[ "${SKIP_MULTIPL_E_GEN:-0}" != "1" ]] && { [[ ! -d "${GEN}" ]] || [[ -z "$(ls -A "${GEN}"/*.json 2>/dev/null)" ]]; }; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "${SCRIPT_DIR}/gen_multipl_e.py" \
            --model "${MODEL}" --lang "${LANG}" --out_dir "${GEN}" --tp "${TP}" \
            --dtype "${DTYPE:-bfloat16}" --max_tokens "${MULTIPL_E_MAX_TOKENS:-2048}" \
            2>&1 | tee "${OUT_DIR}/logs/${EXP_NAME}_${LANG}.log"
    else
        echo "skip ${LANG} generation: using existing gens in ${GEN}"
    fi

    ${DOCKER} run --rm "${DOCKER_PLATFORM[@]}" \
        -v "${OUT_DIR}/gens:/g" -v "${OUT_DIR}/results:/r" \
        ghcr.io/nuprl/multipl-e-evaluation:latest \
        --dir "/g/${EXP_NAME}_${LANG}" --output-dir "/r/${EXP_NAME}_${LANG}" --max-workers 8 \
        || { echo "warning: MultiPL-E docker eval failed for ${LANG}"; continue; }
done
