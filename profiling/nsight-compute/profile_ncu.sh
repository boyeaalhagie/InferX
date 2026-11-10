#!/bin/bash
#SBATCH --job-name="inferx_ncu"
#SBATCH --output=/home/ad.msoe.edu/boyea/InferX/profiling/nsight-compute/ncu_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

# Nsight Compute kernel-level profiling.
# ncu lives at /usr/local/bin/ncu inside the pytorch container.
# Use AFTER nsys has identified the dominant kernel.
# ncu replays each kernel many times — use 1 request and short sequences.
#
# Env vars:
#   ENGINE          pytorch (default)
#   INPUT_TOKENS    512 (default) — use short sequences to keep ncu fast
#   OUTPUT_TOKENS   32  (default)
#   TARGET_KERNEL   substring filter (e.g. "fmha" or "gemm"); empty = all kernels
#   LAUNCH_SKIP     skip first N launches of target kernel (default: 20, skip warmup)
#   LAUNCH_COUNT    how many launches to profile per kernel (default: 3)

set -euo pipefail

REPO_ROOT="/home/ad.msoe.edu/boyea/InferX"
CONTAINER="/data/containers/msoe-pytorch-25.05-py3.sif"
PROFILE_DIR="${REPO_ROOT}/profiling/nsight-compute"

ENGINE="${ENGINE:-pytorch}"
MODEL="${MODEL:-${REPO_ROOT}/models/qwen2.5-7b}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_KERNEL="${TARGET_KERNEL:-}"
LAUNCH_SKIP="${LAUNCH_SKIP:-20}"
LAUNCH_COUNT="${LAUNCH_COUNT:-3}"

PROFILE_OUT="${PROFILE_DIR}/ncu_${ENGINE}_in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_${SLURM_JOB_ID}"

module load h100Env 2>/dev/null && echo "h100Env loaded" || echo "h100Env not available"
mkdir -p "$PROFILE_DIR"

echo "========================================"
echo "Nsight Compute Kernel Profile"
echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Engine:        $ENGINE"
echo "Input tok:     $INPUT_TOKENS"
echo "Output tok:    $OUTPUT_TOKENS"
echo "Target kernel: ${TARGET_KERNEL:-<all>}"
echo "Output:        ${PROFILE_OUT}.ncu-rep"
echo ""

# Key metrics for roofline / bottleneck analysis:
#   sm__throughput           — SM utilization (compute bound if near 100%)
#   dram__bytes.sum          — DRAM bytes (memory bound if bandwidth-limited)
#   l1tex__t_bytes.sum       — L1 cache bytes
#   sm__cycles_elapsed.avg   — cycle count for latency
#   gpu__time_duration.sum   — wall time per kernel
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
l1tex__t_bytes.sum,\
sm__cycles_elapsed.avg,\
gpu__time_duration.sum"

KERNEL_FLAGS=""
if [[ -n "$TARGET_KERNEL" ]]; then
    KERNEL_FLAGS="-k regex:${TARGET_KERNEL} --launch-skip ${LAUNCH_SKIP} --launch-count ${LAUNCH_COUNT}"
fi

# --clock-control none: skip GPU clock locking (avoids root requirement on HPC).
# --target-processes all: profile child processes launched by Python.
# ncu is at /usr/local/bin/ncu inside the container.
srun singularity exec --nv \
    -B /data:/data \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${CONTAINER}" \
    /usr/local/bin/ncu \
        --metrics "${METRICS}" \
        --clock-control none \
        --target-processes all \
        --export "${PROFILE_OUT}" \
        --force-overwrite \
        ${KERNEL_FLAGS} \
        python3 "${REPO_ROOT}/benchmarks/bench_inference" \
            --engine "${ENGINE}" \
            --model "${MODEL}" \
            --dtype bf16 \
            --input-tokens "${INPUT_TOKENS}" \
            --output-tokens "${OUTPUT_TOKENS}" \
            --batch-size "${BATCH_SIZE:-1}" \
            --requests 1 \
            --warmup 0 \
        2>&1

echo ""
echo "Profile written to: ${PROFILE_OUT}.ncu-rep"
echo "Transfer to local machine to view in Nsight Compute UI."
