# InferX Architecture

InferX is an inference performance engineering platform for large language models on NVIDIA H100 GPUs.
It provides a unified benchmark harness, multiple inference engine backends, profiling integration,
and analysis tooling for understanding latency, throughput, memory, and kernel-level behavior.

## Repo Layout

```
InferX/
├── benchmarks/
│   └── bench_inference        # Main benchmark script (Python, executable)
├── configs/
│   └── default.yaml           # Default sweep parameters
├── engines/
│   ├── base.py                # InferenceEngine ABC, GenerationResult, MemoryStats
│   ├── pytorch/               # HuggingFace Transformers baseline engine
│   ├── vllm/                  # vLLM PagedAttention engine
│   └── tensorrt_llm/          # TensorRT-LLM 0.21.0 engine + build scripts
├── profiling/
│   ├── nsight-systems/        # nsys SLURM script + .nsys-rep profiles
│   └── nsight-compute/        # ncu SLURM script (ERR_NVGPUCTRPERM on Rosie)
├── results/                   # JSONL benchmark outputs, SLURM logs, TRT engines
├── scripts/
│   └── submit_job.sh          # Generic SLURM wrapper for all engines
├── workloads/                 # Prompt datasets (synthetic + LMSYS)
├── evaluation/                # Accuracy eval scripts (perplexity, MMLU)
├── reports/
│   └── full_report.txt        # Consolidated findings from all experiments
└── docs/                      # This directory
```

## Benchmark Harness (`benchmarks/bench_inference`)

The benchmark script is engine-agnostic. It:
1. Instantiates the requested engine (`--engine pytorch|vllm|tensorrt_llm`)
2. Calls `engine.load()` to load model weights
3. Runs warmup batches, then timed inference batches
4. Records TTFT, ITL, E2E latency, tokens/sec, GPU memory via `engine.generate()`
5. Writes one JSONL record per run to `results/`

Each engine implements `InferenceEngine` from `engines/base.py`:
- `load()` — model load + warmup
- `generate(prompts, max_new_tokens) -> GenerationResult`
- `teardown()` — VRAM release
- `get_memory_stats() -> MemoryStats`

## Engine Backends

| Engine        | Container                        | Key Feature                          |
|---------------|----------------------------------|--------------------------------------|
| pytorch       | msoe-pytorch-25.05-py3.sif       | HF Transformers, autoregressive loop |
| vllm          | msoe-vllm.sif                    | PagedAttention, continuous batching  |
| tensorrt_llm  | msoe-pytorch-25.05-py3.sif       | Fused TRT kernels, paged KV cache    |

TensorRT-LLM uses the pytorch container because no dedicated TRT-LLM Singularity image
exists on Rosie; TRT-LLM 0.21.0 is pip-installed to `~/.local` within that container.

## SLURM Integration

All benchmark runs go through `scripts/submit_job.sh`, which:
- Selects the correct container based on `--engine`
- Requests `gpu:h100:N` on the `dgxh100` partition
- Mounts `/data` and the repo root into the container
- Passes all benchmark arguments through to `bench_inference`

Engine build jobs (TRT-LLM) submit directly via `sbatch engines/tensorrt_llm/build_engine.sh`.

## Import Conflict Workarounds

TRT-LLM 0.21.0 pip-install creates two Python import conflicts that affect all engines
when running inside the pytorch container:

1. **torch._dynamo circular import**: `transformers` flex_attention uses
   `@torch.compiler.disable`, which calls `torch._dynamo.config` at import time before
   `_dynamo` is fully initialized. Fix: `import torch._dynamo; import torch._dynamo.config`
   before any transformers import.

2. **tensorflow/protobuf conflict**: `tensorrt_llm.quantization` → `AutoProcessor` →
   `image_transforms` → `import tensorflow`. TRT-LLM downgraded protobuf to 4.24.4;
   TF 2.21.0 needs 6.31.1+. Fix: stub tensorflow in `sys.modules` before TRT-LLM imports.
   The stub must be added AFTER `_dynamo` is initialized (otherwise `_dynamo.trace_rules`
   sees `__spec__=None` on the stub and raises `ValueError`).

Both fixes are applied at the top of `benchmarks/bench_inference` so all engines benefit.
