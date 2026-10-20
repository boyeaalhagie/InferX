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
GPUS=8
DRY_RUN=false
PARTITION="dgxh100"
MAIL_USER="boyea@ad.msoe.edu"
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
    pytorch)      CONTAINER="/data/inferx/containers/pytorch.sif" ;;
    vllm)         CONTAINER="/data/inferx/containers/vllm.sif" ;;
    tensorrt_llm) CONTAINER="/data/inferx/containers/tensorrt_llm.sif" ;;
    *)            echo "Unknown engine: $ENGINE"; exit 1 ;;
esac

JOB_NAME="inferx_${ENGINE}"
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

echo "Job ID: \$SLURM_JOB_ID"
echo "Node:   \$(hostname)"
echo "Engine: ${ENGINE}"
echo "Date:   \$(date)"
echo ""

srun singularity exec \\
    --nv \\
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
