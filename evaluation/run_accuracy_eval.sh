#!/bin/bash
#SBATCH --job-name="inferx"
#SBATCH --output=/home/ad.msoe.edu/boyea/InferX/evaluation/accuracy_eval_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/home/ad.msoe.edu/boyea/InferX"
CONTAINER="/data/containers/msoe-pytorch-25.05-py3.sif"

module load h100Env 2>/dev/null && echo "h100Env loaded" || echo "h100Env not available"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo ""

srun singularity exec \
    --nv \
    -B /data:/data \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${CONTAINER}" \
    python3 "${REPO_ROOT}/evaluation/accuracy_eval.py" \
        --model "${REPO_ROOT}/models/qwen2.5-7b" \
        --output-file "${REPO_ROOT}/evaluation/accuracy_results.json" \
        --dtypes bf16 fp16
