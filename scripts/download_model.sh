#!/bin/bash
# Download a HuggingFace model to /data/inferx/models/.
# Run this on the LOGIN NODE — compute nodes have no internet.
#
# Usage:
#   ./scripts/download_model.sh llama        # downloads Llama-3.1-8B
#   ./scripts/download_model.sh qwen         # downloads Qwen2.5-7B
#   ./scripts/download_model.sh mistral      # downloads Mistral-7B-v0.3
#   ./scripts/download_model.sh <hf-repo-id> # any HuggingFace repo

set -euo pipefail

DEST_BASE="/data/inferx/models"

# Friendly aliases
case "${1:-}" in
    llama)   MODEL_ID="meta-llama/Llama-3.1-8B" ;;
    qwen)    MODEL_ID="Qwen/Qwen2.5-7B" ;;
    mistral) MODEL_ID="mistralai/Mistral-7B-v0.3" ;;
    "")      echo "Usage: $0 <llama|qwen|mistral|hf-repo-id>"; exit 1 ;;
    *)       MODEL_ID="$1" ;;
esac

# Derive a clean local directory name from the repo id
MODEL_DIR="${DEST_BASE}/$(echo "$MODEL_ID" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"

echo "Model:       $MODEL_ID"
echo "Destination: $MODEL_DIR"
echo ""

# Check destination
mkdir -p "$DEST_BASE"
if [[ -d "$MODEL_DIR" ]]; then
    echo "Directory already exists. Checking for incomplete download..."
fi

# Check huggingface-hub is available
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install --quiet huggingface-hub
fi

echo "Starting download..."
python3 - <<PYEOF
from huggingface_hub import snapshot_download
import os

model_id = "${MODEL_ID}"
local_dir = "${MODEL_DIR}"

print(f"Downloading {model_id} -> {local_dir}")
path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    ignore_patterns=["*.pt", "original/*"],  # skip non-safetensor weights
)
print(f"Done: {path}")

# Report size
total = 0
for dirpath, _, filenames in os.walk(path):
    for f in filenames:
        total += os.path.getsize(os.path.join(dirpath, f))
print(f"Total size: {total / 1e9:.1f} GB")
PYEOF

echo ""
echo "Model ready at: $MODEL_DIR"
echo "Use --model $MODEL_DIR in bench_inference"
