# Benchmarking Guide

## Quick Start

```bash
# Submit a benchmark job for the pytorch engine
./scripts/submit_job.sh --engine pytorch -- \
    --model models/qwen2.5-7b \
    --input-tokens 512 --output-tokens 256 --batch-size 4 --requests 50

# vLLM
./scripts/submit_job.sh --engine vllm -- \
    --model models/qwen2.5-7b \
    --input-tokens 512 --output-tokens 256 --batch-size 4 --requests 50

# TensorRT-LLM (needs pre-built engine)
./scripts/submit_job.sh --engine tensorrt_llm -- \
    --model models/qwen2.5-7b \
    --trtllm-engine-dir results/trt_engines/qwen2.5-7b_bf16_tp1 \
    --input-tokens 512 --output-tokens 256 --batch-size 4 --requests 50

# Dry run (print sbatch script without submitting)
./scripts/submit_job.sh --engine pytorch --dry-run -- --model ...
```

## bench_inference CLI Reference

```
usage: bench_inference [--engine ENGINE] [--model MODEL] [--dtype DTYPE]
                       [--input-tokens N] [--output-tokens N]
                       [--batch-size N] [--requests N]
                       [--warmup N] [--output-file FILE]
                       [--slurm-job-id ID]
                       [--trtllm-engine-dir DIR]

Required:
  --engine      pytorch | vllm | tensorrt_llm
  --model       Path to HuggingFace model directory

Sizing:
  --input-tokens    Tokens per prompt (synthetic prompts used if no dataset)
  --output-tokens   Max new tokens to generate
  --batch-size      Number of requests per batch
  --requests        Total number of inference requests (warmup excluded)

Options:
  --dtype       bf16 | fp16 | int8 (default: bf16)
  --warmup      Warmup batches before timing (default: 3)
  --output-file  Explicit output JSONL path (default: auto-generated in results/)
```

## Output Format (JSONL)

Each run writes one JSON record to `results/`:

```json
{
  "run_id": "c2213bb5",
  "model": "/path/to/model",
  "engine": "pytorch",
  "gpu": "NVIDIA H100 80GB HBM3",
  "dtype": "bf16",
  "batch_size": 4,
  "input_tokens": 512,
  "output_tokens": 256,
  "requests": 50,
  "ttft_ms": 55.9,
  "itl_ms": 11.0,
  "latency_ms": 2850.3,
  "output_tokens_per_sec": 357.8,
  "total_tokens_per_sec": 537.4,
  "requests_per_sec": 1.40,
  "gpu_memory_peak_gb": 15.0,
  "gpu_utilization_pct": 82.4,
  "slurm_job_id": "296980",
  "timestamp": "2026-11-09T18:15:48",
  "warmup_requests": 3
}
```

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| `ttft_ms` | Time-to-first-token: wall time from request start to first output token |
| `itl_ms` | Inter-token latency: mean time between consecutive output tokens (decode phase) |
| `latency_ms` | End-to-end wall time for the full batch |
| `output_tokens_per_sec` | Generated (output) tokens per second: `batch × output_tokens / latency_s` |
| `total_tokens_per_sec` | (input + output) tokens per second |
| `gpu_memory_peak_gb` | Max `memory.used` reported by `nvidia-smi` during the run. **Caveat:** the poller reads every GPU on the node, so on shared nodes this can include other jobs |
| `gpu_utilization_pct` | Mean `utilization.gpu` from `nvidia-smi`, averaged over all GPUs on the node (same caveat) |

**Note on vLLM ITL**: vLLM's offline inference API (`LLM.generate`) does not expose
per-request metrics in this setup, so `itl_ms` is 0 and `ttft_ms` falls back to the total batch
latency. Compare vLLM on throughput only.

**Note on TRT-LLM TTFT**: `ModelRunner` does not expose per-token timestamps.
`ttft_ms` is a placeholder (30% of total wall time), not a measurement. Compare TRT-LLM on throughput only.
Use the streaming executor API for exact timing.

## Experiment Naming Convention

Result files follow the pattern `{experiment}_{engine}_{dtype}_{params}.jsonl`:

| Prefix | Experiment |
|--------|-----------|
| `A_`   | Baseline throughput/latency sweep (input length × batch size) |
| `B_`   | vLLM PagedAttention KV cache sweep |
| `C_`   | Batching efficiency (batch size 1–64, fixed input/output) |
| `D_`   | Input/output length sensitivity |
| `E_`   | Precision comparison (bf16 / fp16 / int8) |
| `exp_F_` | CUDA graphs: `pytorch` vs `pytorch_compiled` |
| `trtllm_` | TensorRT-LLM throughput runs |

## TensorRT-LLM Engine Build

Before running TRT-LLM benchmarks, build the engine:

```bash
# Build with defaults (Qwen2.5-7B, BF16, TP=1)
sbatch engines/tensorrt_llm/build_engine.sh

# Custom model/dtype
MODEL_PATH=/path/to/model DTYPE=fp16 sbatch engines/tensorrt_llm/build_engine.sh
```

Build takes ~20–30 minutes on a single H100. Output in
`results/trt_engines/<model>_<dtype>_tp<N>/`.

## Profiling

See `docs/profiling.md` for Nsight Systems usage.
Nsight Compute is not available on Rosie (ERR_NVGPUCTRPERM).
