#!/bin/bash
#SBATCH --job-name="inferx_nsys"
#SBATCH --output=inferx_nsys_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G

# Nsight Systems profiling wrapper.
# Wraps bench_inference with nsys to capture full CPU/GPU timeline.
#
# Usage: sbatch profiling/nsight-systems/profile_nsys.sh \
#            --engine pytorch --model /data/inferx/models/llama-3.1-8b \
#            --input-tokens 512 --output-tokens 64 --batch-size 1 --requests 5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINER="${CONTAINER:-/data/inferx/containers/pytorch.sif}"
PROFILE_DIR="${REPO_ROOT}/profiling/nsight-systems"
ENGINE="${ENGINE:-pytorch}"

# Remaining args are passed to bench_inference
BENCH_ARGS=("$@")

PROFILE_NAME="${PROFILE_DIR}/profile_${ENGINE}_${SLURM_JOB_ID}"

module load h100Env 2>/dev/null && echo "h100Env loaded" || echo "h100Env not available"
echo "Nsight Systems profiling"
echo "Engine:   $ENGINE"
echo "Profile:  ${PROFILE_NAME}.nsys-rep"
echo "Node:     $(hostname)"
echo ""

# Check nsys is available (may be outside container on the host)
if command -v nsys &>/dev/null; then
    NSYS_CMD="nsys"
elif [[ -f /opt/nvidia/nsight-systems/*/bin/nsys ]]; then
    NSYS_CMD=$(ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1)
else
    echo "WARNING: nsys not found on host. Falling back to torch.profiler inside container."
    NSYS_CMD=""
fi

mkdir -p "$PROFILE_DIR"

if [[ -n "$NSYS_CMD" ]]; then
    # Profile from outside the container, targeting the containerized process
    srun "$NSYS_CMD" profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --output="${PROFILE_NAME}" \
        --force-overwrite=true \
        --capture-range=cudaProfilerApi \
        singularity exec --nv \
            -B /data:/data \
            -B "${REPO_ROOT}:${REPO_ROOT}" \
            "${CONTAINER}" \
            python3 "${REPO_ROOT}/benchmarks/bench_inference" \
            --engine "$ENGINE" \
            "${BENCH_ARGS[@]}"
else
    # Fallback: use torch.profiler via INFERX_TORCH_PROFILE env var
    srun singularity exec --nv \
        -B /data:/data \
        -B "${REPO_ROOT}:${REPO_ROOT}" \
        "${CONTAINER}" \
        bash -c "
            INFERX_TORCH_PROFILE=1 \
            INFERX_PROFILE_OUTPUT='${PROFILE_NAME}_torch' \
            python3 '${REPO_ROOT}/benchmarks/bench_inference' \
            --engine '$ENGINE' ${BENCH_ARGS[*]}
        "
fi

echo ""
echo "Profile written to: ${PROFILE_NAME}"
echo "View with: nsys-ui ${PROFILE_NAME}.nsys-rep"
