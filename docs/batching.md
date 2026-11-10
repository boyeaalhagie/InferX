# Batching and Scheduling

## Why Batching Matters

LLM decode at bs=1 is severely memory-bandwidth bound: the full model (15 GB for 7B BF16)
must be streamed from HBM for each generated token, regardless of batch size.
Increasing batch size amortizes this weight-loading cost across more tokens per pass,
converting memory-bound GEMV operations into compute-bound GEMM operations.

**Arithmetic intensity** scales with batch size:
- bs=1: ~4 FLOP/byte (memory bound, well below H100's ~1,183 FLOP/byte crossover)
- bs=32: ~128 FLOP/byte (approaching compute bound for large weight matrices)
- bs=64+: tensor cores become the bottleneck

## Measured Throughput vs Batch Size

**Experiment C**: both engines, in=512 tokens, out=256 tokens.

| Batch Size | PyTorch tok/s | vLLM tok/s | PyTorch speedup vs bs=1 | vLLM speedup vs bs=1 |
|-----------|--------------|------------|------------------------|---------------------|
|  1        |          81  |        167 |                  1.0×  |                1.0× |
|  2        |         179  |        335 |                  2.2×  |                2.0× |
|  4        |         354  |        641 |                  4.4×  |                3.8× |
|  8        |         698  |      1,265 |                  8.6×  |                7.6× |
|  16       |       1,246  |      2,477 |                 15.4×  |               14.8× |
|  32       |       2,204  |      4,879 |                 27.2×  |               29.2× |
|  64       |       3,426  |      8,186 |                 42.3×  |               49.0× |

Both engines scale near-linearly up to bs=32. Diminishing returns appear at bs=64 (PyTorch
achieves 42× at bs=64 vs a theoretical 64× linear) due to compute saturation at large tiles.

vLLM's advantage grows with batch size (from 2× at bs=1 to ~2.4× at bs=64), reflecting
continuous batching's ability to keep the GPU saturated while PyTorch's static batching
introduces idle time between batches.

## Latency vs Throughput Tradeoff

Increasing batch size improves throughput but increases per-request TTFT:

| BS  | PyTorch TTFT | PyTorch ITL | Throughput |
|-----|-------------|-------------|-----------|
|  1  |     17.7 ms |     12.3 ms |      81/s |
|  4  |     55.9 ms |     11.1 ms |     354/s |
|  8  |    110.1 ms |     11.1 ms |     698/s |
| 16  |    218.5 ms |     12.0 ms |   1,246/s |
| 32  |    427.6 ms |     12.8 ms |   2,204/s |
| 64  |    876.2 ms |     15.2 ms |   3,426/s |

TTFT scales linearly with batch size (prefill processes all batch items together).
ITL remains nearly flat (~11–12ms) because decode batch items execute in parallel —
H100 compute is sufficient to process up to bs=64 decode steps simultaneously.

ITL rises slightly at bs=64 (12ms → 15ms), suggesting the tile sizes at bs=64 start
to pressure shared memory or register files.

## Continuous Batching (vLLM)

Static batching (PyTorch engine) processes each batch as an atomic unit:
1. Prefill all bs requests together
2. Decode all requests step-by-step until the last one finishes
3. Flush, start next batch → GPU idle between batches

Continuous batching (vLLM) interleaves requests:
1. New requests added to the running batch as slots free
2. Finished requests removed immediately
3. GPU never idles waiting for slow requests
4. Result: higher average GPU utilization, especially with variable-length outputs

**Measured TTFT impact**: vLLM TTFT (~800–1,600ms for bs=1–4) is higher than
PyTorch (~12–55ms) because the continuous batching scheduler queues requests
to form efficient batches. For interactive applications requiring <50ms TTFT,
the static HF engine at small batch sizes performs better.

## Input Length vs Output Length Effects

**Experiment D** (batch=4, engine comparison):

| Workload         | Input | Output | PyTorch tok/s | vLLM tok/s | TTFT ratio |
|-----------------|-------|--------|--------------|------------|------------|
| Short prompt     |  128  |   512  |        323.5 |      646.5 | vLLM 182×  |
| Medium prompt    | 1024  |   256  |        351.3 |      647.8 | vLLM  14×  |
| Long prompt      | 4096  |   128  |        258.7 |      600.9 | vLLM   1.9×|

The TTFT ratio inverts: vLLM is 182× slower at short prompts (scheduler overhead
dominates) but only 1.9× slower at long prompts (prefill cost dominates, scheduler
overhead becomes negligible).

**Recommendation**:
- Short prompts (< 256 tokens) + latency-sensitive: use HF PyTorch engine
- Long prompts or high batch throughput: use vLLM
- Production at scale: use vLLM or TRT-LLM with continuous batching

## Optimal Batch Size

For H100 80GB with Qwen2.5-7B BF16, throughput plateaus around bs=64–128.
Memory limit: 80GB − 15GB (weights) = 65GB for KV cache and activations.
At 56KB/token and 512 input + 256 output = 768 tokens max per request:
`65GB / (768 tokens × 56KB/token) ≈ 1,510 concurrent requests` (theoretical max).
In practice, activations and framework overhead limit to ~bs=128–256.
