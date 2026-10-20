#!/bin/bash
# Run a full experiment sweep by iterating over the config grid
# and submitting one SLURM job per combination.
#
# Usage:
#   ./scripts/run_experiment.sh A              # Experiment A (baseline)
#   ./scripts/run_experiment.sh B              # Experiment B (KV-cache)
#   ./scripts/run_experiment.sh A --dry-run    # print jobs without submitting

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=false
EXP="${1:-}"

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXP" ]]; then
    echo "Usage: $0 <A|B|C|D|E> [--dry-run]"
    exit 1
fi

CONFIG="${REPO_ROOT}/configs/experiments/${EXP}_*.yaml"
CONFIG_FILE=$(ls $CONFIG 2>/dev/null | head -1)

if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "No config found for experiment $EXP (looked for $CONFIG)"
    exit 1
fi

MODEL=$(python3 -c "
import yaml, sys
with open('${REPO_ROOT}/configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('model', ''))
")

echo "Experiment:  $EXP"
echo "Config:      $CONFIG_FILE"
echo "Model:       $MODEL"
echo "Dry run:     $DRY_RUN"
echo ""

# Parse sweep grid from config and submit one job per combination
python3 - <<PYEOF
import yaml
import itertools
import subprocess
import sys

with open("${CONFIG_FILE}") as f:
    cfg = yaml.safe_load(f)

sweep = cfg.get("sweep", {})
engine_default = cfg.get("engine", "pytorch")
dtype = cfg.get("dtype", "bf16")
input_tokens_default = cfg.get("input_tokens", 512)
output_tokens = cfg.get("output_tokens", 256)
requests = cfg.get("requests", 50)
warmup = cfg.get("warmup", 3)

# Build all combinations from the sweep block
engines      = sweep.get("engine", [engine_default])
dtypes       = sweep.get("dtype", [dtype])
input_tokens = sweep.get("input_tokens", [input_tokens_default])
batch_sizes  = sweep.get("batch_size", [cfg.get("batch_size", 1)])

combos = list(itertools.product(engines, dtypes, input_tokens, batch_sizes))
print(f"Total jobs to submit: {len(combos)}")
print()

for (eng, dt, itok, bs) in combos:
    cmd = [
        "${REPO_ROOT}/scripts/submit_job.sh",
        "--engine", eng,
    ]
    if "${DRY_RUN}" == "true":
        cmd.append("--dry-run")
    cmd += [
        "--",
        "--model", "${MODEL}",
        "--engine", eng,
        "--dtype", dt,
        "--input-tokens", str(itok),
        "--output-tokens", str(output_tokens),
        "--batch-size", str(bs),
        "--requests", str(requests),
        "--warmup", str(warmup),
        "--output-file",
        f"${REPO_ROOT}/results/${EXP}_{eng}_{dt}_in{itok}_bs{bs}.jsonl",
    ]
    label = f"  {eng:15s} dtype={dt} in={itok:4d} bs={bs}"
    if "${DRY_RUN}" == "true":
        print(f"[DRY RUN] {label}")
        print("  " + " ".join(cmd))
        print()
    else:
        print(f"Submitting: {label}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  -> {result.stdout.strip()}")
        else:
            print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
PYEOF
