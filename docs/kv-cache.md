# KV Cache Engineering

## What the KV Cache Is

During Transformer inference, each attention layer computes:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

For autoregressive decode, the Keys and Values for all previous tokens are reused unchanged
at every step. The KV cache stores them so they don't have to be recomputed.

**KV cache size** for a single sequence:
```
bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × sizeof(dtype)
```

For Qwen2.5-7B at BF16 (28 layers, 4 KV heads via grouped-query attention, d_head=128):
```
bytes/token = 2 × 28 × 4 × 128 × 2 = 57,344 bytes ≈ 56 KB/token
```

| Sequence length | Per sequence | × 32 sequences |
|-----------------|--------------|----------------|
| 512             | ~29 MB       | ~0.9 GB        |
| 2,048           | ~117 MB      | ~3.8 GB        |
| 4,096           | ~235 MB      | ~7.5 GB        |

Grouped-query attention (4 KV heads instead of 28) keeps this 7× smaller than it would
otherwise be.

## How the Engines Store It

**PyTorch baseline:** the engine's decode loop uses the Hugging Face model's default dynamic
cache, which grows by appending one token per step (concatenating into larger tensors).

**PagedAttention (vLLM)** (Kwon et al. 2023) stores the KV cache in fixed-size blocks allocated
on demand:
- Block size: typically 16 tokens
- Blocks allocated incrementally as a sequence grows; nothing is copied
- A block table maps logical positions to physical blocks
- Enables sharing for beam search and prefix caching

vLLM pre-allocates its block pool at startup (`gpu_memory_utilization=0.90` here), which is why
its memory use reads ~72.7 GB regardless of workload.

## Measured Results (H100 80GB HBM3, Qwen2.5-7B BF16)

**Throughput, Experiment C (in=512, out=256):**

| BS  | PyTorch tok/s | vLLM tok/s | vLLM / PyTorch |
|-----|--------------|------------|----------------|
|   1 |         81.1 |      167.3 |          2.06× |
|   4 |        354.2 |      641.0 |          1.81× |
|   8 |        698.0 |    1,264.6 |          1.81× |
|  16 |      1,246.0 |    2,476.8 |          1.99× |
|  32 |      2,203.8 |    4,878.8 |          2.21× |
|  64 |      3,425.9 |    8,185.9 |          2.39× |

At these sizes, memory isn't the constraint: 64 sequences × 768 tokens × 56 KB ≈ 2.8 GB of
KV cache, far below what's free on an 80 GB GPU. So this gap is **not** caused by paging.
It's mostly lower per-step overhead in vLLM (see `inference-model.md`). PagedAttention pays off
when memory *is* the limit: long contexts, many concurrent users, or bigger models.

**vLLM throughput vs prompt length, Experiment B (BS=32, out=128):**

| Prompt tokens | tok/s |
|---------------|-------|
| 512           | 4,752 |
| 1,024         | 4,591 |
| 2,048         | 4,052 |
| 4,096         | 3,666 |

Throughput falls ~23% from 512 to 4,096 tokens: longer prompts mean longer prefill and more KV
cache to read at every decode step.

**Measured memory:** the GPU-memory numbers in `results/` are not reliable for the PyTorch
engine. The `nvidia-smi` poller in `bench_inference` reads every GPU on the 8-GPU node, not
just the assigned one. On one node, every run reports 67.35 GB regardless of workload. Use the
computed sizes above instead.

## TensorRT-LLM Paged KV Cache

TRT-LLM 0.21.0 enables paged KV cache via `--use_paged_context_fmha enable` at build time, and
its C++ runtime manages the block allocator.

Built engine configuration:
```json
{
  "max_batch_size": 32,
  "max_input_len": 4096,
  "max_seq_len": 4352,
  "use_paged_context_fmha": true
}
```

## Practical Implications

1. **Concurrency limit without paging** (reserving max length up front):
   `floor((80 GB − 15 GB) / 235 MB) ≈ 276` sequences at 4,096 tokens.
   With paging, memory is only used for tokens that actually exist.

2. **When the block pool runs out**, vLLM preempts requests. By default it frees their blocks
   and recomputes them later; it can also swap them to CPU memory. Either causes latency
   spikes that don't show up in average throughput.
