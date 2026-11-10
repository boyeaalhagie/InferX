#!/bin/bash
#SBATCH --job-name="inferx_env_check"
#SBATCH --output=inferx_env_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G

echo "========================================"
echo "InferX Environment Check"
echo "========================================"
echo ""

echo "--- Node ---"
hostname
date
echo ""

echo "--- GPU ---"
nvidia-smi
echo ""

echo "--- CUDA ---"
nvcc --version 2>/dev/null || echo "nvcc not in PATH"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo ""

echo "--- Python ---"
python3 --version
which python3
echo ""

echo "--- Singularity ---"
singularity --version
which singularity
echo ""

echo "--- Profilers ---"
which nsys 2>/dev/null && nsys --version || echo "nsys not found"
which ncu  2>/dev/null && ncu  --version || echo "ncu not found"
echo ""

echo "--- /data/ containers ---"
ls -lh /data/containers/ 2>/dev/null || echo "/data/containers/ not found or empty"
echo ""

echo "--- /data/ storage ---"
df -h /data 2>/dev/null || echo "Cannot check /data quota"
echo ""

echo "--- /data/inferx/ ---"
ls -la /data/inferx/ 2>/dev/null || echo "/data/inferx/ does not exist yet"
echo ""

echo "--- Module system ---"
module avail 2>&1 | head -60 || echo "module command not available"
echo ""

echo "--- pip packages (system python) ---"
python3 -m pip list 2>/dev/null | grep -iE "torch|vllm|tensorrt|transformers|huggingface" || echo "no relevant packages found"
echo ""

echo "--- Network access (login nodes only — expect failure on compute) ---"
curl -s --max-time 5 https://huggingface.co > /dev/null && echo "HuggingFace reachable" || echo "HuggingFace NOT reachable (expected on compute nodes)"
echo ""

echo "========================================"
echo "Environment check complete"
echo "========================================"
