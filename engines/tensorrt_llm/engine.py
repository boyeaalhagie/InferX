"""
TensorRT-LLM inference engine.

Requires a pre-built TRT engine directory (produced by build_engine.sh).
Wraps the tensorrt_llm Python runtime API.

Key TRT-LLM features we instrument:
  - In-flight batching (executor API)
  - Paged KV cache
  - Quantization (INT8/FP8 set at engine build time)
  - CUDA graphs (captured automatically by TRT-LLM runtime)
"""

import time
from pathlib import Path
from typing import List

from engines.base import InferenceEngine, GenerationResult, MemoryStats


class TRTLLMEngine(InferenceEngine):

    def __init__(
        self,
        engine_dir: str,
        dtype: str = "bf16",
        max_batch_size: int = 8,
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
                f"Build it first with: engines/tensorrt_llm/build_engine.sh"
            )

        print(f"[trtllm] Loading engine from {self.engine_dir}")
        try:
            from tensorrt_llm import LLM as TRTLLM
            from tensorrt_llm.llmapi import LLM, BuildConfig

            # Use the high-level LLM API if available (TRT-LLM >= 0.10)
            self._runner = TRTLLM.from_dir(self.engine_dir)
        except ImportError:
            # Fall back to ModelRunner (older TRT-LLM versions)
            from tensorrt_llm.runtime import ModelRunner
            self._runner = ModelRunner.from_dir(
                engine_dir=self.engine_dir,
                rank=0,
            )

        # Load tokenizer from the original model path if provided
        if self.model_path and self.model_path != self.engine_dir:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, use_fast=True
                )
            except Exception as e:
                print(f"[trtllm] Warning: could not load tokenizer: {e}")

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

        # Tokenize
        input_ids_list = [
            self.tokenizer.encode(p, return_tensors=None)
            for p in prompts
        ]

        import torch
        input_ids_tensors = [torch.tensor(ids, dtype=torch.int32) for ids in input_ids_list]

        sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,  # greedy
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
        for i, output_ids in enumerate(outputs):
            # Strip input tokens from output
            out_ids = output_ids[len(input_ids_list[i]):]
            texts.append(self.tokenizer.decode(out_ids, skip_special_tokens=True))
            output_token_counts.append(len(out_ids))

        # TRT-LLM does not expose per-token timestamps through ModelRunner.
        # TTFT approximation: assume prefill is proportional to input length.
        # For accurate TTFT, use the executor streaming API (future work).
        ttft_ms = total_latency_ms * 0.3  # rough estimate — replace with streaming API

        return GenerationResult(
            texts=texts,
            ttft_ms=ttft_ms,
            itl_ms_list=[],  # not available via ModelRunner — use executor API
            total_latency_ms=total_latency_ms,
            output_token_counts=output_token_counts,
        )

    def teardown(self) -> None:
        if self._runner is not None:
            del self._runner
            self._runner = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("[trtllm] Engine unloaded.")
