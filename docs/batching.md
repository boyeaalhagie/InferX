# Batching and Scheduling

## Why Batching Matters

LLM decode is memory-bandwidth-bound: the full model (~15.2 GB for a 7B model in BF16) is
streamed from HBM for every decode step. Batching serves many sequences in the same step, so
each weight read is reused for more tokens.

**Arithmetic intensity** grows roughly linearly with batch size (each 2-byte weight does
2·B FLOPs):

| BS | Arithmetic intensity | vs H100 BF16 ridge (~295 FLOP/byte) |
|----|----------------------|-------------------------------------|
| 1  | ~1 FLOP/byte         | deeply memory-bound                 |
| 32 | ~32 FLOP/byte        | still memory-bound                  |
| 64 | ~64 FLOP/byte        | still memory-bound                  |

Even at BS=64, decode on an H100 has not reached the compute roof, which is why throughput
keeps climbing with batch size.

## Measured Throughput vs Batch Size

**Experiment C**: both engines, in=512 tokens, out=256 tokens.

| Batch Size | PyTorch tok/s | vLLM tok/s | PyTorch speedup vs BS=1 | vLLM speedup vs BS=1 |
|-----------|--------------|------------|------------------------|---------------------|
|  1        |          81  |        167 |                  1.0×  |                1.0× |
|  2        |         179  |        335 |                  2.2×  |                2.0× |
|  4        |         354  |        641 |                  4.4×  |                3.8× |
|  8        |         698  |      1,265 |                  8.6×  |                7.6× |
|  16       |       1,246  |      2,477 |                 15.4×  |               14.8× |
|  32       |       2,204  |      4,879 |                 27.2×  |               29.2× |
|  64       |       3,426  |      8,186 |                 42.3×  |               49.0× |

Both engines scale near-linearly up to BS=32. At BS=64, PyTorch reaches 42× (vs an ideal 64×)
as per-step time starts to rise (see below).

vLLM is already ~2× faster at BS=1, where batching can't be the reason. Its advantage comes
from lower per-step overhead (CUDA graphs, optimized kernels). See `inference-model.md`.

## Latency vs Throughput Tradeoff (PyTorch)

| BS  | TTFT     | ITL     | Throughput |
|-----|----------|---------|-----------|
|  1  |  17.7 ms | 12.3 ms |      81/s |
|  4  |  55.9 ms | 11.1 ms |     354/s |
|  8  | 110.1 ms | 11.1 ms |     698/s |
| 16  | 218.5 ms | 12.0 ms |   1,246/s |
| 32  | 427.6 ms | 12.8 ms |   2,204/s |
| 64  | 876.2 ms | 15.2 ms |   3,426/s |

- **TTFT grows ~linearly with batch size**: all BS prompts are prefilled together.
- **ITL stays ~flat from BS=1 to BS=32**: this is the signature of a memory-bound workload.
  The cost of a step is dominated by reading the weights once, so using them for 32 tokens
  instead of 1 is nearly free.
- **ITL rises at BS=64** (12.8 → 15.2 ms) as KV-cache reads and attention work for 64
  sequences start to add up.

## Continuous Batching (vLLM)

Static batching (PyTorch engine) processes each batch as an atomic unit:
1. Prefill all requests together
2. Decode step-by-step until the last request finishes
3. Start the next batch → slots of finished requests sit idle

Continuous batching (vLLM) adds and removes requests at every step, so slots are refilled as
soon as a request finishes. This matters most when requests arrive at different times and
have different output lengths.

**Caveat:** every experiment here submits all requests at once with identical prompt and
output lengths, so this benefit is *not* exercised by these benchmarks. Testing it needs a
workload with staggered arrivals and variable lengths.

**vLLM TTFT is not measured.** The offline `LLM.generate` API used here did not return
per-request timing metrics, so the harness falls back to recording the total batch latency in
`ttft_ms`. Don't compare vLLM `ttft_ms` values against PyTorch TTFT.

## Input Length vs Output Length (Experiment D)

Batch size 4:

| Workload      | Input | Output | PyTorch tok/s | vLLM tok/s | PyTorch TTFT |
|---------------|-------|--------|--------------|------------|--------------|
| Short prompt  |  128  |   512  |        323.5 |      646.5 |      17.6 ms |
| Medium prompt | 1024  |   256  |        351.3 |      647.8 |     107.8 ms |
| Long prompt   | 4096  |   128  |        258.7 |      600.9 |     446.2 ms |

- PyTorch TTFT grows with prompt length, as expected for prefill.
- Output tok/s here includes prefill time, so the long-prompt workload (4,096 in / 128 out)
  spends a much larger share of its time in prefill and reports lower output throughput.
- vLLM holds ~600–650 tok/s across all three workloads.

## Batch Size Headroom

Throughput was still rising at BS=64, the largest size tested.

Theoretical KV-cache capacity on an 80 GB H100: ~65 GB is left after weights. At 56 KB/token
and 768 tokens per request (512 in + 256 out):
`65 GB / (768 × 56 KB) ≈ 1,500` concurrent sequences. Activations, fragmentation, and
framework overhead lower this in practice.
