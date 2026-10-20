# InferX — Inference Performance Engineering Platform

> A performance study and optimization framework for production-scale Transformer inference on NVIDIA GPUs.

**Core question:** How do we make Transformer inference faster, more memory-efficient, and more scalable?

This is not an LLM. It's the engineering infrastructure *around* inference — a benchmark harness, profiling pipeline, and experiment framework running on MSOE's Rosie HPC cluster (H100 GPUs).

---

## Quick start

### 1. Check the environment (first time only)
```bash
sbatch scripts/check_environment.sh
cat inferx_env_<JOBID>.out
```

### 2. Download the model (on login node)
```bash
./scripts/download_model.sh llama    # or: qwen, mistral
```

### 3. Dry-run the harness
```bash
python3 benchmarks/bench_inference --dry-run \
    --engine pytorch \
    --model /data/inferx/models/meta-llama_llama-3.1-8b \
    --input-tokens 512 --output-tokens 256 --batch-size 4 --requests 50
```

### 4. Submit a real benchmark job
```bash
./scripts/submit_job.sh --engine pytorch -- \
    --model /data/inferx/models/meta-llama_llama-3.1-8b \
    --input-tokens 512 --output-tokens 256 --batch-size 4 --requests 50
```

### 5. Run a full experiment sweep
```bash
./scripts/run_experiment.sh A --dry-run   # preview
./scripts/run_experiment.sh A             # submit all jobs
```

### 6. Generate a report
```bash
python3 reports/report_generator.py --results-dir results/
python3 reports/report_generator.py --table --compare results/A_*.jsonl
```

---

## Architecture

```
bench_inference (CLI)
    │
    ├─ WorkloadGenerator  — token-calibrated synthetic prompts
    │
    ├─ InferenceEngine (abstract)
    │       ├─ PyTorchEngine      — naive baseline, manual decode loop
    │       ├─ VLLMEngine         — continuous batching, paged KV cache
    │       └─ TRTLLMEngine       — NVIDIA optimized, in-flight batching
    │
    ├─ GPUPoller          — background nvidia-smi memory/utilization sampling
    │
    └─ BenchResult → JSONL → report_generator.py
```

Profiling: Nsight Systems (system timeline) → Nsight Compute (kernel drill-down).

---

## Experiments

| ID | Name | Research question |
|----|------|-------------------|
| A | Baseline | Establish PyTorch reference numbers |
| B | KV-Cache | How does memory scale with sequence length × concurrency? |
| C | Batching | Where does batching stop helping? |
| D | Prefill vs Decode | How does the bottleneck shift between phases? |
| E | Precision | What are the perf + memory + accuracy tradeoffs of INT8? |

---

## Repository layout

```
InferX/
├── benchmarks/bench_inference    ← main CLI entrypoint
├── engines/
│   ├── base.py                   ← InferenceEngine abstract interface
│   ├── pytorch/engine.py
│   ├── vllm/engine.py
│   └── tensorrt_llm/
│       ├── engine.py
│       └── build_engine.sh       ← compile HF weights → TRT engine
├── configs/
│   ├── default.yaml
│   └── experiments/              ← A_baseline, B_kvcache, C_batching, ...
├── scripts/
│   ├── check_environment.sh      ← run this first on dgxh100
│   ├── download_model.sh         ← run on login node
│   ├── submit_job.sh             ← SLURM wrapper
│   └── run_experiment.sh         ← sweep submitter
├── profiling/
│   ├── nsight-systems/profile_nsys.sh
│   └── nsight-compute/profile_ncu.sh
├── reports/report_generator.py
└── results/baselines/            ← committed reference results (Phase 2)
```

---

## Hardware

**Cluster:** MSOE Rosie HPC — `dgxh100` partition
**GPUs:** 8× NVIDIA H100 (80GB) per node
**Scheduler:** SLURM (`sbatch` for all H100 jobs)
**Containers:** Singularity 3.3 (not Docker)
**Data:** `/data/inferx/` on shared cluster storage
