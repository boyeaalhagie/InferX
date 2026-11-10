#!/bin/bash
# Run Experiment F — CUDA Graphs vs Normal Launch
# Submits all 8 runs: pytorch baseline vs pytorch_compiled, BS 1/4, out 128/512

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${REPO_ROOT}/models/qwen2.5-7b"
OUTPUT_DIR="${REPO_ROOT}/results/baselines"
mkdir -p "$OUTPUT_DIR"

submit() {
    local ENGINE="$1"
    local BS="$2"
    local OUT_TOKENS="$3"
    local REQUESTS="$4"

    ./scripts/submit_job.sh --engine "${ENGINE}" --gpus 1 -- \
        --model "$MODEL" \
        --engine "${ENGINE}" \
        --dtype bf16 \
        --input-tokens 512 \
        --output-tokens "${OUT_TOKENS}" \
        --batch-size "${BS}" \
        --requests "${REQUESTS}" \
        --warmup 2 \
        --output-file "${OUTPUT_DIR}/exp_F_cuda_graphs.jsonl"
}

echo "Submitting Experiment F — CUDA Graphs"
echo ""

# pytorch baseline
submit pytorch 1 128 20
submit pytorch 1 512 10
submit pytorch 4 128 20
submit pytorch 4 512 10

# pytorch_compiled (CUDA graphs via torch.compile)
# Note: submit_job.sh uses pytorch container for pytorch_compiled too
submit pytorch_compiled 1 128 20
submit pytorch_compiled 1 512 10
submit pytorch_compiled 4 128 20
submit pytorch_compiled 4 512 10

echo ""
echo "All Experiment F jobs submitted."
echo "Results -> ${OUTPUT_DIR}/exp_F_cuda_graphs.jsonl"
