# KV Cache Engineering

## What the KV Cache Is

During Transformer inference, each attention layer computes:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

For autoregressive decode, the Keys and Values for all previously-generated tokens
are reused unchanged each step. The KV cache stores these tensors to avoid recomputation.

**KV cache size** for a single request:
```
bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × sizeof(dtype)
```

For Qwen2.5-7B at BF16 (28 layers, 4 KV heads, d_head=128):
```
bytes/token = 2 × 28 × 4 × 128 × 2 = 57,344 bytes ≈ 56 KB/token
```

At seq_len=4096: ~224 MB per request. At batch=32: ~7.2 GB — significant fraction
of 80GB HBM.

## Paged Attention (vLLM)

Standard KV cache allocates contiguous memory per request upfront (maximum seq_len × KV size).
This wastes memory when requests finish early and prevents fine-grained sharing.

**PagedAttention** (Kwon et al. 2023) stores KV cache in fixed-size blocks (pages)
allocated on demand:
- Block size: typically 16–32 tokens
- Blocks allocated incrementally as sequence grows
- Block table maps logical positions to physical blocks
- Enables KV cache sharing for beam search and prefix caching

### Measured Impact on Rosie (H100 80GB HBM3, Qwen2.5-7B BF16)

**Throughput at in=512 (best per batch size)**:

| BS  | PyTorch tok/s | vLLM tok/s | vLLM advantage |
|-----|--------------|------------|----------------|
|   1 |         88.9 |      168.1 |          1.89× |
|   4 |        357.8 |      662.6 |          1.85× |
|   8 |        698.0 |    1,264.6 |          1.81× |
|  16 |      1,246.0 |    2,515.9 |          2.02× |
|  32 |      2,203.8 |    4,878.8 |          2.21× |
|  64 |      3,425.9 |    8,185.9 |          2.39× |

The advantage grows with batch size because paged memory allows vLLM to run more
concurrent requests within the same 80GB HBM budget.

**TTFT comparison** (in=512, bs=1):
- PyTorch: 17.8ms ← very fast, no queue
- vLLM: 761ms ← continuous batching scheduler adds queuing overhead

vLLM's higher TTFT at bs=1 reflects the continuous batching scheduler's overhead
(waiting to form batches, Python scheduler loop). For latency-sensitive applications
at low batch sizes, the HF engine has lower TTFT.

## KV Cache Memory Scaling

Measured GPU memory peak (PyTorch engine, BF16):

| Input tokens | Batch size | GPU Memory Peak |
|-------------|------------|----------------|
| 128         | 1          | 15.0 GB        |
| 128         | 8          | 15.3 GB        |
| 512         | 1          | 15.0 GB        |
| 512         | 4          | 15.4 GB        |
| 2048        | 1          | 15.4 GB        |
| 2048        | 4          | 16.3 GB        |

Memory growth is modest relative to model weights (14.9 GB baseline for 7.6B params
in BF16 = 15.2 GB). The KV cache adds ~0.1–1.4 GB depending on batch × input_len.

## TensorRT-LLM Paged KV Cache

TRT-LLM 0.21.0 enables paged KV cache via `--use_paged_context_fmha enable` at build time.
The executor backend (in-flight batching) manages the block allocator at the C++ level,
eliminating Python scheduler overhead while maintaining memory efficiency.

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

1. **Batch size limit** without paged attention: `floor(80GB / (model_weights + max_KV_per_request))`
   For 7B model at BF16 with 4096 max seq: `floor((80-15) / 0.224)` ≈ 290 requests max

2. **vLLM page eviction**: when HBM fills, vLLM evicts KV blocks to CPU RAM and
   recomputes them on re-schedule. This causes latency spikes not visible in throughput
   averages. Use `--max-model-len` to cap sequence length and avoid eviction.

3. **For production**: set KV cache fraction explicitly (`--gpu-memory-utilization 0.85`
   for vLLM) to leave headroom for model weights + activations.
