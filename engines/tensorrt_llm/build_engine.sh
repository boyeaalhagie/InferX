#!/bin/bash
#SBATCH --job-name="inferx"
#SBATCH --output=/home/ad.msoe.edu/boyea/InferX/results/slurm_%j_trtllm_build.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

# TensorRT-LLM 0.21.0 engine build for Qwen2.5-7B.
# Uses trtllm-build from ~/.local/bin (installed via pip install tensorrt-llm).
#
# Environment variables (set before submitting):
#   MODEL_PATH   — HuggingFace model weights (default: ~/InferX/models/qwen2.5-7b)
#   OUTPUT_DIR   — compiled engine output (default: ~/InferX/results/trt_engines/qwen2.5-7b_bf16)
#   DTYPE        — bf16 | fp16 (default: bf16)
#   TP_SIZE      — tensor parallelism (default: 1)

set -euo pipefail

REPO_ROOT="/home/ad.msoe.edu/boyea/InferX"
CONTAINER="/data/containers/msoe-pytorch-25.05-py3.sif"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/qwen2.5-7b}"
DTYPE="${DTYPE:-bf16}"
TP_SIZE="${TP_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/trt_engines/$(basename ${MODEL_PATH})_${DTYPE}_tp${TP_SIZE}}"

echo "========================================"
echo "TRT-LLM 0.21.0 Engine Build"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Model:     $MODEL_PATH"
echo "Output:    $OUTPUT_DIR"
echo "dtype:     $DTYPE"
echo "TP size:   $TP_SIZE"
echo ""

module load h100Env 2>/dev/null && echo "h100Env loaded" || echo "h100Env not available"
mkdir -p "$OUTPUT_DIR"

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint"
mkdir -p "$CHECKPOINT_DIR"

# TRT-LLM bindings need libcuda.so.1 from the CUDA compat path.
# SINGULARITYENV_LD_LIBRARY_PATH is applied before --nv prepends driver paths.
export SINGULARITYENV_LD_LIBRARY_PATH="/usr/local/cuda/compat/lib"

echo "--- Step 1: Convert Qwen2.5-7B HF weights to TRT-LLM checkpoint ---"
srun singularity exec --nv \
    -B /data:/data \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${CONTAINER}" \
    python3 -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.12/site-packages'))
from tensorrt_llm.models.qwen.convert import convert_hf_qwen
from tensorrt_llm.models.qwen.config import QWenConfig
from tensorrt_llm import str_dtype_to_torch, logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dtype_map = {'bf16': 'bfloat16', 'fp16': 'float16'}
torch_dtype = getattr(torch, dtype_map.get('${DTYPE}', 'bfloat16'))

print('Loading HF model for weight conversion...')
hf_model = AutoModelForCausalLM.from_pretrained(
    '${MODEL_PATH}',
    torch_dtype=torch_dtype,
    device_map='cpu',
)
tokenizer = AutoTokenizer.from_pretrained('${MODEL_PATH}')

config = QWenConfig.from_hugging_face(
    '${MODEL_PATH}',
    dtype='${DTYPE}',
    mapping=None,
)
print('Config loaded. Converting weights...')
convert_hf_qwen(
    hf_model=hf_model,
    hf_config=hf_model.config,
    mapping=config.mapping,
    dtype='${DTYPE}',
    output_dir='${CHECKPOINT_DIR}',
)
print('Conversion complete.')
" 2>&1

echo ""
echo "--- Step 2: Build TRT-LLM engine ---"
srun singularity exec --nv \
    -B /data:/data \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${CONTAINER}" \
    python3 "${HOME}/.local/bin/trtllm-build" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --gemm_plugin "${DTYPE}" \
        --gpt_attention_plugin "${DTYPE}" \
        --max_batch_size 32 \
        --max_input_len 4096 \
        --max_seq_len 4352 \
        --use_paged_context_fmha enable \
        --workers "${TP_SIZE}" \
        2>&1

echo ""
echo "========================================"
echo "Build complete."
ls -lh "${OUTPUT_DIR}"

# Write build config sidecar
cat > "${OUTPUT_DIR}/build_config.json" <<JSONEOF
{
  "model_path": "${MODEL_PATH}",
  "dtype": "${DTYPE}",
  "tp_size": ${TP_SIZE},
  "max_batch_size": 32,
  "max_seq_len": 4352,
  "trtllm_version": "0.21.0",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)",
  "build_date": "$(date -Iseconds)"
}
JSONEOF

echo "Engine ready at: ${OUTPUT_DIR}"
