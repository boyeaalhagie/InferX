# Inference Performance Model

## The Two Phases of Autoregressive Inference

Every request passes through two phases with very different hardware behavior.

### Prefill (prompt processing)
- Processes all N prompt tokens in one forward pass → large GEMMs with shape `[N × d_model]`
- Builds the KV cache entries for all N positions
- Each weight is reused across N tokens → high arithmetic intensity
- **Compute-bound** once the prompt is long enough to fill the GPU

Measured TTFT (PyTorch, BS=1, BF16):

| Prompt tokens | TTFT (ms) |
|---------------|-----------|
| 128           | 12.6      |
| 512           | 17.8      |
| 2048          | 56.7      |
| 4096          | 117.6 \*  |

Going from 128 to 512 tokens (4× more) costs only 1.4× more time: a short prompt doesn't
fill the GPU, so extra tokens ride along almost for free. Past ~2K tokens, TTFT roughly
doubles when the prompt doubles, which is the compute-bound regime.

\* The N=4096 point comes from the Nsight Systems profiling run (in=4096, out=32), not from a
`results/` record.

### Decode (token generation)
- Generates one token per step per sequence
- **Every step streams all ~15.2 GB of BF16 weights from HBM**, plus the KV cache
- Very little math per byte → **memory-bandwidth-bound**
- Measured ITL: ~11–12 ms/token (PyTorch, BS=1 through BS=32)

## Roofline Numbers (H100 SXM5, dense)

| Quantity                       | Value            |
|--------------------------------|------------------|
| BF16 Tensor Core peak (dense)  | ~989 TFLOP/s     |
| HBM3 bandwidth                 | ~3.35 TB/s       |
| Ridge point (BF16)             | ~295 FLOP/byte   |

(The frequently quoted ~3,958 TFLOP/s is FP8 *with* 2:4 sparsity and doesn't apply to these BF16 runs.)

**Decode arithmetic intensity.** At batch size B, each 2-byte BF16 weight is read once per
step and used in B multiply-adds (2B FLOPs):

```
AI ≈ 2B FLOPs / 2 bytes ≈ B FLOP/byte
BS=1  → ~1 FLOP/byte
BS=64 → ~64 FLOP/byte
```

Both are far below the ~295 FLOP/byte ridge point, so decode stays memory-bound at every
batch size measured here. (KV-cache and activation traffic only lower the intensity further.)

**Lower bound on ITL.** Streaming 15.2 GB at 3.35 TB/s takes ~4.5 ms. Measured PyTorch ITL is
~11–12 ms, so about 40% of peak bandwidth is achieved. The rest goes to kernel-launch overhead,
synchronization, and non-GEMM kernels (see `profiling.md`).

## Measured Performance on Qwen2.5-7B, H100 80GB HBM3, BF16

### PyTorch (HF Transformers model, custom decode loop)

| Input | Output | BS | TTFT (ms) | ITL (ms) | Tok/s   |
|-------|--------|----|-----------|----------|---------|
| 128   | 256    |  1 |      12.6 |     12.1 |    82.8 |
| 512   | 256    |  1 |      17.8 |     11.2 |    88.9 |
| 2048  | 256    |  1 |      56.7 |     12.3 |    80.4 |
| 4096  | 32     |  1 |   117.6 \* |  16.6 \* |  50.1 \* |
| 128   | 256    |  8 |      29.2 |     11.3 |   704.8 |
| 512   | 256    | 64 |     876.2 |     15.2 | 3,425.9 |

\* From the Nsight Systems profiling run (profiler overhead included).

**Why ITL is flat across prompt lengths:** each decode step reads all ~15.2 GB of weights but
only a small KV cache: 56 KB/token, so ~115 MB for a 2,048-token sequence at BS=1. The KV
reads grow with sequence length, but at these lengths they are under 1% of the bytes read per
step, so ITL barely moves.

### Throughput Scaling with Batch Size

Experiment C, in=512, out=256:

| BS  | PyTorch tok/s | vLLM tok/s | vLLM / PyTorch |
|-----|--------------|------------|----------------|
|   1 |         81.1 |      167.3 |          2.06× |
|   4 |        354.2 |      641.0 |          1.81× |
|   8 |        698.0 |    1,264.6 |          1.81× |
|  16 |      1,246.0 |    2,476.8 |          1.99× |
|  32 |      2,203.8 |    4,878.8 |          2.21× |
|  64 |      3,425.9 |    8,185.9 |          2.39× |

vLLM is ~2× faster **even at BS=1**, where batching plays no role. So the gap comes from lower
per-step overhead (vLLM captures decode steps as CUDA graphs and uses optimized kernels), not
from continuous batching or paging.

These runs submit all requests at once with identical lengths, so continuous batching's main
advantage (admitting requests that arrive at different times) is not exercised here.

## GPU Kernel Time Breakdown (Nsight Systems)

Profiled on the pytorch engine, H100 80GB HBM3, BF16, BS=1.

### Decode-heavy: in=512, out=128
- Total GPU kernel time: 6,137 ms
- nvjet GEMM kernels: 4,179 ms (**68.1%**)
- Elementwise kernels: 260 ms (4.2%)
- Flash Attention: 254 ms (4.1%)

### Prefill-heavy: in=4096, out=32
- Total GPU kernel time: 1,727 ms
- nvjet GEMM kernels: 940 ms (**54.4%**)
- Elementwise kernels: 13.7%
- Flash Attention: 159 ms (9.2%)

Attention's share grows with sequence length, but linear-layer GEMMs (the weight reads)
dominate at these lengths.

## Python Dispatch Overhead

From the Nsight Systems CUDA API trace (decode-heavy profile):
- 855,648 `cudaLaunchKernel` calls × ~5 µs = **~4.3 s** of launch overhead in an ~11 s run
- 780 `cudaDeviceSynchronize` calls × 318 µs = **~248 ms** of GPU idle time
- GPU kernel time is only ~6.1 s of that ~11 s run

Note: the PyTorch engine calls `torch.cuda.synchronize()` after every decode step so it can
timestamp each token. That stops the CPU from queueing work ahead of the GPU, so part of this
idle time is caused by measurement itself.

Production runtimes reduce launch overhead with CUDA graphs captured over a **pre-allocated,
fixed-shape KV cache**, plus C++ schedulers. Simply wrapping this engine in
`torch.compile(backend="cudagraphs")` did not work; see Experiment F in
`reports/full_report.txt`.
