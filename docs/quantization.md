# Quantization

## Overview

Lower-precision weights mean fewer bytes to stream per decode step. Because decode is
memory-bound, that *can* speed up inference, but only if the hardware actually computes with
the smaller format instead of converting back first.

Formats evaluated in Experiment E:

| dtype | Bits/param | Weight size (7.6B params) | How it runs here                              |
|-------|-----------|---------------------------|-----------------------------------------------|
| bf16  | 16        | ~15.2 GB                  | Native BF16 Tensor Cores                      |
| fp16  | 16        | ~15.2 GB                  | Native FP16 Tensor Cores                      |
| int8  | 8         | ~7.6 GB (theoretical)     | bitsandbytes `LLM.int8()` (mixed precision)   |

## Measured Results (Experiment E)

**Setup**: Qwen2.5-7B, H100 80GB HBM3, in=512, out=256, BS=4.

### PyTorch Engine

| dtype | TTFT (ms) | ITL (ms) | Tok/s |
|-------|-----------|----------|-------|
| bf16  |      55.9 |     11.0 | 357.3 |
| fp16  |      56.8 |     12.4 | 317.3 |
| int8  |      55.7 |     11.2 | 351.1 |

### vLLM Engine

| dtype | Tok/s |
|-------|-------|
| bf16  | 662.6 |
| fp16  | 646.1 |

**Findings**:
- **INT8 gives no speedup** (351 vs 357 tok/s). bitsandbytes `LLM.int8()` is designed to save
  memory, not time: it splits each matrix multiply into an INT8 part and a 16-bit part for
  outlier features, with quantize/dequantize work around them. The smaller weights don't turn
  into faster decode.
- **FP16 vs BF16**: the H100 runs both at the same Tensor Core rate. FP16 measured 11% slower
  in PyTorch but only 2.5% slower in vLLM. The PyTorch BF16 and FP16 runs landed on
  different nodes, so treat that gap as run-to-run variation until it's rerun on one node.
- **Memory**: INT8's expected benefit is roughly halving weight memory (~15.2 → ~7.6 GB of
  weights). It was **not measured reliably** here: the GPU-memory poller reads every GPU on the
  node (see `kv-cache.md`). vLLM always reads ~72.7 GB because it pre-allocates 90% of GPU memory.

## Accuracy (FP16 vs BF16)

`evaluation/accuracy_eval.py`, 20 fixed prompts, 128 tokens, greedy decoding, BF16 as reference:

| dtype | ROUGE-L | Token overlap | Length ratio |
|-------|---------|---------------|--------------|
| fp16  | 0.850   | 0.873         | 0.995        |

Greedy decoding amplifies tiny numeric differences: once one token differs, the rest of the
text can take a different path. The outputs stay on topic and similar in length. INT8 accuracy
was not evaluated.

## Getting Real Speedups from Low Precision

Speedups need kernels that compute end-to-end in the low-precision format:
- TensorRT-LLM INT8 SmoothQuant, or weight-only INT8/INT4 (AWQ, GPTQ) with fused kernels
- FP8 on H100 (native FP8 Tensor Cores)
- All require a calibration step and an accuracy check

These were not evaluated in this project.

## Precision Recommendation

For Qwen2.5-7B on H100 80GB:
- **BF16**: default choice; fast and numerically robust
- **FP16**: same hardware speed; use if a toolchain requires it
- **INT8 (bitsandbytes)**: only to fit a model in less memory; don't expect a speedup
- **FP8 / INT8 / INT4 with dedicated kernels**: the path to actual speedups; needs calibration
  and accuracy validation
