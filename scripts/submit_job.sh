#!/bin/bash
# Generic SLURM job wrapper for InferX benchmark runs.
# Usage: ./scripts/submit_job.sh [--engine ENGINE] [--gpus N] [--dry-run] -- <bench_inference args>
#
# Examples:
#   ./scripts/submit_job.sh --engine pytorch -- --model /data/inferx/models/llama-3.1-8b \
#       --input-tokens 2048 --output-tokens 256 --batch-size 8 --requests 50
#
#   ./scripts/submit_job.sh --engine vllm --gpus 1 --dry-run -- --model ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults
ENGINE="pytorch"
GPUS=1
DRY_RUN=false
PARTITION="dgxh100"
MAIL_USER="${MAIL_USER:-your_username@your_institution.edu}"
CONTAINER=""
BENCH_ARGS=()

# Parse args up to --
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)   ENGINE="$2";  shift 2 ;;
        --gpus)     GPUS="$2";    shift 2 ;;
        --dry-run)  DRY_RUN=true; shift   ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --)         shift; BENCH_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Select container based on engine
case "$ENGINE" in
    pytorch|pytorch_compiled) CONTAINER="/data/containers/msoe-pytorch-25.05-py3.sif" ;;
    vllm)                     CONTAINER="/data/containers/msoe-vllm.sif" ;;
    # TRT-LLM 0.21.0 is pip-installed to ~/.local inside the pytorch container.
    tensorrt_llm)             CONTAINER="/data/containers/msoe-pytorch-25.05-py3.sif" ;;
    *)                        echo "Unknown engine: $ENGINE"; exit 1 ;;
esac

JOB_NAME="inferx"
RESULTS_DIR="${REPO_ROOT}/results"
BENCH_SCRIPT="${REPO_ROOT}/benchmarks/bench_inference"

# Build the sbatch script content
JOB_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --output=${RESULTS_DIR}/slurm_%j_${ENGINE}.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:${GPUS}
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G

echo "Job ID: \$SLURM_JOB_ID"
echo "Node:   \$(hostname)"
echo "Engine: ${ENGINE}"
echo "Date:   \$(date)"
echo ""

# Load h100Env for profiler access (nsys, ncu)
module load h100Env 2>/dev/null && echo "h100Env loaded" || echo "h100Env not available"


NOUSERSITE=""
[[ "${ENGINE}" == "vllm" ]] && NOUSERSITE="--env PYTHONNOUSERSITE=1"

# TRT-LLM bindings explicitly load libcuda.so.1 (driver API).
# SINGULARITYENV_LD_LIBRARY_PATH is applied before --nv prepends paths,
# making the compat library discoverable inside the container.
[[ "${ENGINE}" == "tensorrt_llm" ]] && export SINGULARITYENV_LD_LIBRARY_PATH="/usr/local/cuda/compat/lib"

srun singularity exec \\
    --nv \\
    \${NOUSERSITE} \\
    -B /data:/data \\
    -B ${REPO_ROOT}:${REPO_ROOT} \\
    ${CONTAINER} \\
    python3 ${BENCH_SCRIPT} \\
    --engine ${ENGINE} \\
    --slurm-job-id \$SLURM_JOB_ID \\
    ${BENCH_ARGS[*]}
EOF
)

if $DRY_RUN; then
    echo "=== DRY RUN — would submit: ==="
    echo "$JOB_SCRIPT"
    exit 0
fi

# Check container exists
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found: $CONTAINER"
    echo "Run scripts/build_containers.sh first."
    exit 1
fi

# Submit
echo "$JOB_SCRIPT" | sbatch
