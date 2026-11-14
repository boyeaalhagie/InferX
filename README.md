# InferX

**LLM inference performance engineering on an NVIDIA H100.** A benchmark harness that runs the
same workload on three inference engines (PyTorch, vLLM, and TensorRT-LLM), profiles them with
Nsight Systems, and explains *why* each one performs the way it does.

- **Model:** Qwen2.5-7B (BF16; FP16 and INT8 for the precision study)
- **Hardware:** 1× NVIDIA H100 80GB HBM3 (DGX H100), via SLURM + Singularity
- **Scope:** 6 experiments, 70 benchmark runs, Nsight Systems kernel profiling

## The core idea

LLM inference has two phases with opposite bottlenecks:

- **Prefill** reads the whole prompt in one pass. Big matrix multiplies, so it's **compute-bound**.
- **Decode** writes one token per step, and every step streams all ~15 GB of weights from
  memory. That's ~1 FLOP per byte, so it's **memory-bandwidth-bound**.

Almost every inference optimization is a way to do more useful work per byte read.

## Key results

| Finding | Result |
|---|---|
| Batching (PyTorch, BS 1 → 64) | **81 → 3,426 tok/s (42×)** |
| Peak throughput (vLLM, BS=64) | **8,186 tok/s**, 2.4× PyTorch at the same batch size |
| Single request (BS=1) | vLLM **2×** PyTorch from lower per-step overhead alone; TRT-LLM 1.6–1.8× |
| Decode time per token | ~11–12 ms, **flat from BS=1 to BS=32**: the signature of a memory-bound workload |
| Where GPU time goes (decode) | 68% matrix multiplies, 4% attention |
| CUDA graphs via `torch.compile` | **2–5× slower**: the graph fast path never ran (fell back to eager) |
| INT8 (bitsandbytes `LLM.int8()`) | **No speedup** (351 vs 357 tok/s). It saves memory, not time |

Two "obvious" optimizations, CUDA graphs and INT8, made things worse or did nothing. Both are
measured and explained in `reports/full_report.txt` and `docs/`.

## Experiments

| | Question | Result files |
|---|---|---|
| **A** Baseline | How do TTFT and ITL scale with prompt length and batch size? | `A_*.jsonl` |
| **B** KV cache | How does vLLM throughput change with prompt length? | `B_*.jsonl` |
| **C** Batching | Where does batching stop helping? | `C_*.jsonl` |
| **D** Prefill vs decode | How do input/output lengths shift the bottleneck? | `D_*.jsonl` |
| **E** Precision | BF16 vs FP16 vs INT8: speed and accuracy | `E_*.jsonl`, `evaluation/` |
| **F** CUDA graphs | Does graph capture remove launch overhead? | `exp_F_cuda_graphs.jsonl` |

All results are in `results/baselines/`, one JSON record per run, tagged with GPU, node, SLURM
job ID, and configuration.

## Measurement caveats

Some recorded fields are not reliable. Each is documented rather than hidden:

- **GPU memory / utilization:** the `nvidia-smi` poller reads every GPU on the 8-GPU node, not
  just the assigned one, so these values can include other jobs.
- **vLLM TTFT:** the offline `LLM.generate` API returned no per-request metrics, so `ttft_ms`
  holds the total batch latency.
- **TensorRT-LLM TTFT:** `ModelRunner` exposes no per-token timing, so `ttft_ms` is a placeholder.
- **Nsight Compute** was blocked on the cluster (`ERR_NVGPUCTRPERM`), so bottlenecks are
  reasoned from timing plus FLOP and byte counts rather than hardware counters.

Throughput and PyTorch TTFT/ITL are measured correctly, so engine comparisons use throughput.

## Repository layout

```
benchmarks/bench_inference   Engine-agnostic benchmark CLI → JSONL records
engines/                     pytorch, pytorch_compiled (CUDA graphs), vllm, tensorrt_llm
configs/experiments/         One YAML file per experiment (A–F)
scripts/submit_job.sh        SLURM wrapper: picks the container per engine and submits
profiling/                   Nsight Systems / Nsight Compute job scripts
evaluation/                  BF16 vs FP16 output-similarity check (ROUGE-L)
reports/full_report.txt      Generated report for every run + analysis
docs/                        Inference model, batching, KV cache, quantization, profiling
```

## Running it

```bash
# Download weights (or point --model at an existing copy)
huggingface-cli download Qwen/Qwen2.5-7B --local-dir models/qwen2.5-7b

# Submit a benchmark on a SLURM cluster
./scripts/submit_job.sh --engine pytorch -- \
    --model models/qwen2.5-7b \
    --input-tokens 512 --output-tokens 256 --batch-size 4 --requests 50

# Swap --engine for vllm or tensorrt_llm; add --dry-run to print the sbatch script
```

Set `MAIL_USER` to your email for SLURM notifications, and update the container paths in
`scripts/submit_job.sh` for your cluster. TensorRT-LLM needs a built engine first:
`sbatch engines/tensorrt_llm/build_engine.sh`. See `docs/benchmarking.md` for the full CLI.

## Next steps

- Static, pre-allocated KV cache so CUDA graphs can actually be replayed in PyTorch
- Real low-precision kernels (FP8 on H100, TensorRT-LLM INT8/AWQ)
- Streaming APIs for true TTFT on vLLM and TensorRT-LLM; per-GPU memory polling
- Staggered request arrivals to test continuous batching
