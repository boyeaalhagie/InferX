# Quantization

## Overview

Quantization reduces model weight precision to decrease memory bandwidth pressure
and potentially increase throughput. For memory-bandwidth-bound decode workloads,
narrower dtype → smaller weights → faster HBM streaming → faster inference.

Formats evaluated in Experiment E:

| dtype | Bits/param | Model size (7B) | H100 support      |
|-------|-----------|-----------------|-------------------|
| bf16  | 16        | ~15.2 GB        | Native BF16 Tensor Cores |
| fp16  | 16        | ~15.2 GB        | Native FP16 Tensor Cores |
| int8  | 8         | ~7.6 GB         | INT8 Tensor Cores (via bitsandbytes) |

## Measured Results (Experiment E)

**Setup**: Qwen2.5-7B, H100 80GB HBM3, in=512, out=256, bs=4, 50 requests.

### PyTorch Engine

| dtype | TTFT (ms) | ITL (ms) | Tok/s   | GPU Mem (GB) |
|-------|-----------|----------|---------|-------------|
| bf16  |      55.9 |     11.1 |   357.3 |        15.0 |
| fp16  |      56.8 |     11.4 |   317.3 |        15.0 |
| int8  |      55.7 |     11.0 |   351.1 |         8.3 |

**Findings**:
- **BF16 vs FP16**: minimal throughput difference (12% advantage for BF16). H100 handles
  both natively via Tensor Cores. BF16 has wider exponent range, better for LLM weights.
- **INT8**: similar throughput to BF16 (~1.7% slower) with nearly half the memory (8.3 GB
  vs 15.0 GB). The expected speedup from halved bandwidth doesn't appear because
  `bitsandbytes` LLM.int8() uses mixed-precision: matrix multiplications are dequantized
  to fp16/bf16 at runtime, negating the bandwidth benefit.
- **Memory saving**: INT8 saves ~6.7 GB (45% reduction), enabling larger batch sizes.

### vLLM Engine

| dtype | TTFT (ms) | Tok/s   | GPU Mem (GB) |
|-------|-----------|---------|-------------|
| bf16  |     1,545 |   662.6 |        30.1 |
| fp16  |     1,585 |   646.1 |        30.1 |

vLLM pre-allocates KV cache on startup (default: 85% of GPU memory). The 30 GB
memory figure includes both model weights (15 GB) and pre-allocated KV cache (~15 GB).

## Why INT8 Doesn't Help Much Here

The bitsandbytes `LLM.int8()` quantization used in this experiment is a **post-training
quantization (PTQ)** scheme designed for memory reduction, not speed:

1. Weights stored as INT8
2. At inference, weight tensors are dequantized to FP16/BF16 for matrix multiply
3. Matrix multiplication runs in BF16 — same bandwidth as native BF16

For genuine INT8 speedup, use **static quantization** with integer arithmetic:
- TensorRT-LLM INT8 with calibration data (SmoothQuant or GPTQ)
- Requires a calibration step and accuracy evaluation
- Can achieve 1.5–2× throughput improvement at equivalent memory savings

## Weight-Only Quantization (INT4)

Not evaluated in this project (no GPTQ/AWQ models available on Rosie).
Expected behavior for 4-bit quantization:
- Memory: ~3.8 GB for 7B model → 4× reduction from BF16
- Throughput: 1.5–2× improvement on memory-bound decode workloads
- Accuracy: 0.1–0.5 point MMLU degradation (model-dependent)

## Precision Recommendation

For Qwen2.5-7B on H100 80GB:
- **BF16**: best default choice — fast, accurate, stable training if fine-tuning
- **FP16**: functionally equivalent, slightly lower throughput
- **INT8 (bitsandbytes)**: use only if 7.6 GB memory saving is needed to fit more
  concurrent requests; don't expect throughput improvement
- **INT8 (TRT-LLM static)**: worth evaluating if 1.5× throughput matters; requires
  calibration dataset and accuracy validation
- **INT4 (GPTQ/AWQ)**: best memory-to-performance ratio for latency-constrained deployments
