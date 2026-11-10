#!/bin/bash
#SBATCH --job-name="inferx_ncu"
#SBATCH --output=inferx_ncu_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

# Nsight Compute kernel profiling.
# Use AFTER Nsight Systems has identified the dominant kernel.
# Drill into that kernel to determine compute vs memory bound, roofline position, etc.
#
# Set TARGET_KERNEL to the kernel name shown by Nsight Systems (or leave blank to profile all).
# WARNING: ncu is very slow — use --requests 1 and short sequences.
#
# Usage:
#   TARGET_KERNEL="fmha_v2_flash_attention" sbatch profiling/nsight-compute/profile_ncu.sh \
#       --engine pytorch --model /data/inferx/models/llama-3.1-8b \
#       --input-tokens 512 --output-tokens 32 --batch-size 1 --requests 1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINER="${CONTAINER:-/data/inferx/containers/pytorch.sif}"
PROFILE_DIR="${REPO_ROOT}/profiling/nsight-compute"
ENGINE="${ENGINE:-pytorch}"
TARGET_KERNEL="${TARGET_KERNEL:-}"

BENCH_ARGS=("$@")
PROFILE_NAME="${PROFILE_DIR}/ncu_${ENGINE}_${SLURM_JOB_ID}"

module load h100Env 2>/dev/null && echo "h100Env loaded" || echo "h100Env not available"
echo "Nsight Compute kernel profiling"
echo "Engine:         $ENGINE"
echo "Target kernel:  ${TARGET_KERNEL:-<all>}"
echo "Profile output: ${PROFILE_NAME}.ncu-rep"
echo ""

mkdir -p "$PROFILE_DIR"

# Core metrics: SM throughput, memory bandwidth, roofline
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum,\
dram__bytes.sum,\
sm__cycles_elapsed.avg,\
gpu__time_duration.sum"

KERNEL_FLAG=""
if [[ -n "$TARGET_KERNEL" ]]; then
    KERNEL_FLAG="--kernel-name ${TARGET_KERNEL} --launch-count 1"
fi

if command -v ncu &>/dev/null; then
    srun ncu \
        --metrics "${METRICS}" \
        ${KERNEL_FLAG} \
        --export "${PROFILE_NAME}" \
        --force-overwrite \
        singularity exec --nv \
            -B /data:/data \
            -B "${REPO_ROOT}:${REPO_ROOT}" \
            "${CONTAINER}" \
            python3 "${REPO_ROOT}/benchmarks/bench_inference" \
            --engine "$ENGINE" \
            "${BENCH_ARGS[@]}"
else
    echo "ERROR: ncu not found on host."
    echo "Nsight Compute must be installed on the compute node."
    exit 1
fi

echo ""
echo "Profile written to: ${PROFILE_NAME}.ncu-rep"
echo "View with: ncu-ui ${PROFILE_NAME}.ncu-rep"
