# Inference Performance Model

## The Two-Phase Structure of Autoregressive LLM Inference

Every autoregressive LLM inference request passes through two distinct computational phases:

### Prefill (prompt processing)
- Processes all N input tokens in a single forward pass
- Computes Q, K, V projections for every token simultaneously → large batched GEMMs
- Builds the KV cache entries for all N positions
- **Compute profile**: large GEMMs with shape [N × d_model], high arithmetic intensity
- **Bound by**: compute (tensor cores saturated for N > ~512 tokens on H100)
- **Measured TTFT**: 12.6ms (N=128) → 56.7ms (N=2048) → 117.6ms (N=4096) on H100 BF16

### Decode (token generation)
- Generates one token per step, attending over the full KV cache
- Each step: Q projection for one token [1 × d_model], then attention over [seq_len × d_head]
- **Compute profile**: tiny GEMMs (~GEMV), 42,672 kernel invocations per 128-token sequence
- **Bound by**: HBM memory bandwidth (weights loaded for minimal compute)
- **Measured ITL**: ~11–12ms/token (bs=1 on H100 BF16)

## Roofline Analysis

The H100 SXM5 specifications:
- Peak BF16 tensor TFLOPS: ~3,958 TFLOPS (with sparsity)
- Peak HBM3 bandwidth: ~3.35 TB/s
- Arithmetic intensity crossover: ~1,183 FLOP/byte

**Decode at bs=1** (from nsys kernel geometry analysis):
- nvjet GEMM tiles are `256×8 output, 64×6 tiles` with `bz` (batch=1) suffix
- Compute per tile: 256×8×64 = 131,072 FLOPs
- Bytes loaded (weights): 256×64×2 (BF16) = 32,768 bytes → ~4 FLOP/byte
- Far below the ~1,183 FLOP/byte crossover → **memory-bandwidth bound**

**Prefill at N=4096**:
- Large GEMMs with `coopA`/`coopB` cooperative launch: 665µs/kernel
- These kernels indicate full tile utilization with cooperative thread groups
- Flash Attention accounts for 9.2% of GPU time (scales with sequence length)
- **Compute bound** for N > ~512

## Measured Performance on Qwen2.5-7B, H100 80GB HBM3, BF16

### PyTorch (HF Transformers baseline)

| Input | Output | BS | TTFT (ms) | ITL (ms) | Tok/s   |
|-------|--------|----|-----------|----------|---------|
| 128   | 256    |  1 |      12.6 |     12.1 |    82.8 |
| 512   | 256    |  1 |      17.8 |     11.2 |    88.9 |
| 2048  | 256    |  1 |      56.7 |     12.3 |    80.4 |
| 4096  | 32     |  1 |     117.6 |     16.6 |    50.1 |
| 128   | 256    |  8 |      29.2 |     11.3 |   704.8 |
| 512   | 256    | 64 |     876.2 |     15.2 | 3,425.9 |

**Key observation**: ITL is nearly flat across input lengths (11–12ms) because decode
does not read the full KV cache on H100 — HBM bandwidth is sufficient to serve the
growing cache without meaningful slowdown until very long sequences.

### Throughput Scaling with Batch Size

Both engines scale linearly in tokens/sec up to batch sizes of 32–64:

| BS  | PyTorch tok/s | vLLM tok/s | vLLM / PyTorch |
|-----|--------------|------------|----------------|
|   1 |         81.1 |      167.3 |          2.06× |
|   4 |        354.2 |      641.0 |          1.81× |
|   8 |        698.0 |    1,264.6 |          1.81× |
|  16 |      1,246.0 |    2,476.8 |          1.99× |
|  32 |      2,203.8 |    4,878.8 |          2.21× |
|  64 |      3,425.9 |    8,185.9 |          2.39× |

vLLM maintains a ~2× throughput advantage across all batch sizes primarily due to
continuous batching and reduced Python overhead in the generation loop.

## GPU Kernel Time Breakdown (from Nsight Systems)

Profiled on pytorch engine, H100 80GB HBM3, BF16, 1 request.

### Decode-heavy: in=512, out=128, bs=1
- Total GPU kernel time: 6,137 ms
- nvjet GEMM kernels: 4,179 ms (**68.1%**)
- Flash Attention: 254 ms (4.1%)
- Elementwise (activations): ~600 ms (9.8%)

### Prefill-heavy: in=4096, out=32, bs=1
- Total GPU kernel time: 1,727 ms
- nvjet GEMM kernels: 940 ms (**54.4%**)
- Flash Attention: 159 ms (9.2%) ← 2.3× share vs decode-heavy

Flash Attention's fraction of GPU time grows with sequence length (O(n²) attention
cost) but remains secondary to linear layer GEMMs at these sequence lengths.

## Python Dispatch Overhead

From Nsight Systems CUDA API trace (decode-heavy profile):
- 855,648 `cudaLaunchKernel` calls × 5µs avg = **4.3 seconds** of pure Python→CUDA overhead
- This represents ~38% of wall time in the HF PyTorch engine
- 780 `cudaDeviceSynchronize` calls × 318µs = **248ms** of mandatory GPU idle time

Production runtimes (vLLM, TRT-LLM) reduce this overhead via:
- Continuous batching (fewer Python-level steps per token)
- CUDA graphs (capture entire decode step, replay without Python overhead)
- C++ execution engines
