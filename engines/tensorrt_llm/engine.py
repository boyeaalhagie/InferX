"""
TensorRT-LLM 0.21.0 inference engine.

Requires a pre-built TRT engine directory (produced by build_engine.sh).
Uses the ModelRunner API from tensorrt_llm.runtime.

Key TRT-LLM features active in this setup:
  - Paged KV cache (enabled by --use_paged_context_fmha at build time)
  - In-flight batching (via the executor backend)
  - CUDA graphs (captured automatically by TRT runtime)
  - Kernel fusion (via trtllm-build plugin flags)

Note: TTFT is approximated as 30% of total latency because ModelRunner
does not expose per-token timestamps. Use the streaming executor API
for exact prefill/decode timing (future work).
"""

import time
from pathlib import Path
from typing import List

import torch

from engines.base import InferenceEngine, GenerationResult, MemoryStats


class TRTLLMEngine(InferenceEngine):

    def __init__(
        self,
        engine_dir: str,
        dtype: str = "bf16",
        max_batch_size: int = 32,
        model_path: str = "",
    ):
        super().__init__(model_path=model_path or engine_dir, dtype=dtype)
        self.engine_dir = engine_dir
        self.max_batch_size = max_batch_size
        self._runner = None

    def load(self) -> None:
        engine_path = Path(self.engine_dir)
        if not engine_path.exists():
            raise FileNotFoundError(
                f"TRT-LLM engine directory not found: {self.engine_dir}\n"
                f"Build it first: sbatch engines/tensorrt_llm/build_engine.sh"
            )

        # Work around two import-time conflicts in the TRT-LLM 0.21.0 pip install:
        # 1. torch._dynamo circular import: pre-initialize _dynamo before anything else
        #    triggers it mid-way (transformers flex_attention uses @torch.compiler.disable).
        # 2. tensorflow/protobuf conflict: tensorrt_llm.quantization imports AutoProcessor
        #    which pulls in image_transforms which does `import tensorflow`; TRT-LLM
        #    downgraded protobuf to 4.24.4 but TF 2.21.0 needs 6.31.1+.
        import os, sys, types
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_JAX", "0")
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import torch._dynamo        # fully init before transformers touches it
        import torch._dynamo.config
        sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

        print(f"[trtllm] Loading engine from {self.engine_dir}")
        from tensorrt_llm.runtime import ModelRunner
        self._runner = ModelRunner.from_dir(
            engine_dir=self.engine_dir,
            rank=0,
        )

        # Load tokenizer from original HF model path if provided
        if self.model_path and self.model_path != self.engine_dir:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[trtllm] Engine loaded.")

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        **kwargs,
    ) -> GenerationResult:
        assert self._runner is not None, "Call load() first"
        assert self.tokenizer is not None, "Tokenizer required for TRT-LLM engine"

        from tensorrt_llm.runtime import SamplingConfig

        # Tokenize — ModelRunner expects List[Tensor(seq_len,)]
        input_ids_list = [
            self.tokenizer.encode(p, return_tensors=None)
            for p in prompts
        ]
        input_ids_tensors = [
            torch.tensor(ids, dtype=torch.int32) for ids in input_ids_list
        ]

        sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,          # greedy — consistent with other engines
        )

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs = self._runner.generate(
                batch_input_ids=input_ids_tensors,
                sampling_config=sampling_config,
            )

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0

        texts, output_token_counts = [], []
        # outputs shape: (batch, beam, seq_len) — beam=1 for greedy
        for i in range(len(prompts)):
            if hasattr(outputs, 'tolist'):
                # Tensor output
                out_ids = outputs[i][0].tolist()
            else:
                out_ids = outputs[i]
            # Strip input tokens
            out_ids = out_ids[len(input_ids_list[i]):]
            texts.append(self.tokenizer.decode(out_ids, skip_special_tokens=True))
            output_token_counts.append(len(out_ids))

        # Approximate TTFT: prefill ≈ 30% of total wall time
        # (ModelRunner doesn't expose per-token timestamps)
        ttft_ms = total_latency_ms * 0.30

        return GenerationResult(
            texts=texts,
            ttft_ms=ttft_ms,
            itl_ms_list=[],   # not available via ModelRunner
            total_latency_ms=total_latency_ms,
            output_token_counts=output_token_counts,
        )

    def teardown(self) -> None:
        if self._runner is not None:
            del self._runner
            self._runner = None
        torch.cuda.empty_cache()
        print("[trtllm] Engine unloaded.")

    def get_memory_stats(self) -> MemoryStats:
        return MemoryStats(
            gpu_allocated_gb=torch.cuda.memory_allocated() / 1e9,
            gpu_reserved_gb=torch.cuda.memory_reserved() / 1e9,
        )
