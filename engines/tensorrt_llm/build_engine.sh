#!/bin/bash
#SBATCH --job-name="inferx_trtllm_build"
#SBATCH --output=inferx_trtllm_build_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8

# TensorRT-LLM engine build script.
# Run as: sbatch engines/tensorrt_llm/build_engine.sh
#
# Environment variables (set before submitting or edit defaults below):
#   MODEL_PATH   — path to HuggingFace model weights
#   OUTPUT_DIR   — where to write the compiled engine
#   DTYPE        — bf16 | fp16 | int8  (default: bf16)
#   TP_SIZE      — tensor parallelism degree (default: 1)
#   MAX_BATCH    — max batch size to compile for (default: 16)
#   MAX_SEQ_LEN  — max sequence length (default: 8192)
#   CONTAINER    — Singularity .sif path (default: /data/inferx/containers/tensorrt_llm.sif)

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/inferx/models/llama-3.1-8b}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/inferx/engines/$(basename $MODEL_PATH)_${DTYPE:-bf16}_tp${TP_SIZE:-1}}"
DTYPE="${DTYPE:-bf16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_BATCH="${MAX_BATCH:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
CONTAINER="${CONTAINER:-/data/inferx/containers/tensorrt_llm.sif}"

echo "========================================"
echo "TRT-LLM Engine Build"
echo "========================================"
echo "Model:      $MODEL_PATH"
echo "Output:     $OUTPUT_DIR"
echo "dtype:      $DTYPE"
echo "TP size:    $TP_SIZE"
echo "Max batch:  $MAX_BATCH"
echo "Max seqlen: $MAX_SEQ_LEN"
echo "Container:  $CONTAINER"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo ""

mkdir -p "$OUTPUT_DIR"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint"
mkdir -p "$CHECKPOINT_DIR"

srun singularity exec --nv -B /data:/data "${CONTAINER}" bash -c "

set -e

echo '--- Step 1: Convert HuggingFace weights to TRT-LLM checkpoint ---'
python3 /usr/local/lib/python3*/dist-packages/tensorrt_llm/examples/*/convert_checkpoint.py \
    --model_dir '${MODEL_PATH}' \
    --output_dir '${CHECKPOINT_DIR}' \
    --dtype '${DTYPE}' \
    --tp_size '${TP_SIZE}' \
    2>&1 | tail -20

echo ''
echo '--- Step 2: Build TRT engine ---'
trtllm-build \
    --checkpoint_dir '${CHECKPOINT_DIR}' \
    --output_dir '${OUTPUT_DIR}' \
    --gpt_attention_plugin '${DTYPE}' \
    --gemm_plugin '${DTYPE}' \
    --max_batch_size '${MAX_BATCH}' \
    --max_input_len 4096 \
    --max_seq_len '${MAX_SEQ_LEN}' \
    --use_paged_context_fmha enable \
    2>&1 | tail -40

echo ''
echo 'Build complete.'
ls -lh '${OUTPUT_DIR}'
"

# Write a build config sidecar for reproducibility
cat > "${OUTPUT_DIR}/build_config.json" <<EOF
{
  "model_path": "${MODEL_PATH}",
  "dtype": "${DTYPE}",
  "tp_size": ${TP_SIZE},
  "max_batch_size": ${MAX_BATCH},
  "max_seq_len": ${MAX_SEQ_LEN},
  "slurm_job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)",
  "build_date": "$(date -Iseconds)"
}
EOF

echo ""
echo "Build config written to: ${OUTPUT_DIR}/build_config.json"
echo "Engine ready at: ${OUTPUT_DIR}"
