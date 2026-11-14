#!/bin/bash
#SBATCH --job-name="inferx_nsys"
#SBATCH --output=profiling/nsight-systems/nsys_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_username@your_institution.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

# Nsight Systems CPU/GPU timeline profiling.
# nsys lives inside the pytorch container at /usr/local/cuda/bin/nsys.
# Run from INSIDE singularity exec — the host has no nsys binary.
#
# Env vars (set before sbatch):
#   ENGINE         pytorch | vllm | tensorrt_llm  (default: pytorch)
#   INPUT_TOKENS   prefill length                  (default: 512)
#   OUTPUT_TOKENS  decode length                   (default: 128)
#   BATCH_SIZE                                     (default: 1)
#   REQUESTS       how many requests to run        (default: 5)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/InferX}"
CONTAINER="/data/containers/msoe-pytorch-25.05-py3.sif"
PROFILE_DIR="${REPO_ROOT}/profiling/nsight-systems"

ENGINE="${ENGINE:-pytorch}"
MODEL="${MODEL:-${REPO_ROOT}/models/qwen2.5-7b}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
REQUESTS="${REQUESTS:-5}"

PROFILE_OUT="${PROFILE_DIR}/profile_${ENGINE}_in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_bs${BATCH_SIZE}_${SLURM_JOB_ID}"

module load h100Env 2>/dev/null && echo "h100Env loaded" || echo "h100Env not available"
mkdir -p "$PROFILE_DIR"

echo "========================================"
echo "Nsight Systems Profile"
echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Engine:     $ENGINE"
echo "Input tok:  $INPUT_TOKENS"
echo "Output tok: $OUTPUT_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo "Requests:   $REQUESTS"
echo "Output:     ${PROFILE_OUT}.nsys-rep"
echo ""

# --sample=none  avoids perf_event_paranoid permission errors on HPC nodes.
# --stats=true   prints kernel/API summary to stdout (useful without UI access).
# nsys is at /usr/local/cuda/bin/nsys inside the container.
srun singularity exec --nv \
    -B /data:/data \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${CONTAINER}" \
    /usr/local/cuda/bin/nsys profile \
        --trace=cuda,nvtx,cudnn,cublas \
        --sample=none \
        --output="${PROFILE_OUT}" \
        --force-overwrite=true \
        --stats=true \
        python3 "${REPO_ROOT}/benchmarks/bench_inference" \
            --engine "${ENGINE}" \
            --model "${MODEL}" \
            --dtype bf16 \
            --input-tokens "${INPUT_TOKENS}" \
            --output-tokens "${OUTPUT_TOKENS}" \
            --batch-size "${BATCH_SIZE}" \
            --requests "${REQUESTS}" \
            --warmup 1 \
        2>&1

echo ""
echo "Profile written to: ${PROFILE_OUT}.nsys-rep"
echo "Transfer to local machine to view in Nsight Systems UI."
