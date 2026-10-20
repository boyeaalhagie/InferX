"""
vLLM inference engine.

Uses vllm.LLM (offline batched API) for static batch experiments.
Uses vllm.AsyncLLMEngine for concurrency / in-flight batching experiments.

vLLM exposes timing directly via RequestOutput.metrics — we use those
rather than wrapping with our own timers.
"""

import time
from typing import List

from engines.base import InferenceEngine, GenerationResult, MemoryStats


class VLLMEngine(InferenceEngine):

    def __init__(
        self,
        model_path: str,
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
    ):
        super().__init__(model_path=model_path, dtype=dtype)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None

    def load(self) -> None:
        from vllm import LLM
        from transformers import AutoTokenizer

        print(f"[vllm] Loading model ({self.dtype}, tp={self.tensor_parallel_size})")

        # dtype string mapping
        dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "int8": "int8"}
        vllm_dtype = dtype_map.get(self.dtype, "bfloat16")

        self._llm = LLM(
            model=self.model_path,
            dtype=vllm_dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        print("[vllm] Engine ready.")

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        **kwargs,
    ) -> GenerationResult:
        assert self._llm is not None, "Call load() first"

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,  # greedy
        )

        t_start = time.perf_counter()
        outputs = self._llm.generate(prompts, sampling_params)
        t_end = time.perf_counter()

        total_latency_ms = (t_end - t_start) * 1000.0

        # Extract timing from vLLM's RequestOutput.metrics
        ttfts, itl_ms_list = [], []
        texts, output_token_counts = [], []

        for output in outputs:
            texts.append(output.outputs[0].text)
            output_token_counts.append(len(output.outputs[0].token_ids))

            metrics = output.metrics
            if metrics is not None:
                if hasattr(metrics, "time_to_first_token") and metrics.time_to_first_token:
                    ttfts.append(metrics.time_to_first_token * 1000.0)
                if hasattr(metrics, "time_per_output_token") and metrics.time_per_output_token:
                    itl_ms_list.append(metrics.time_per_output_token * 1000.0)

        ttft_ms = sum(ttfts) / len(ttfts) if ttfts else total_latency_ms
        mean_itl = itl_ms_list  # already one-per-request from vLLM

        return GenerationResult(
            texts=texts,
            ttft_ms=ttft_ms,
            itl_ms_list=mean_itl,
            total_latency_ms=total_latency_ms,
            output_token_counts=output_token_counts,
        )

    def teardown(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("[vllm] Engine unloaded.")

    def get_memory_stats(self) -> MemoryStats:
        try:
            import torch
            stats = MemoryStats(
                gpu_allocated_gb=torch.cuda.memory_allocated() / 1e9,
                gpu_reserved_gb=torch.cuda.memory_reserved() / 1e9,
            )
            # vLLM exposes cache stats via internal engine — attempt to read
            if self._llm and hasattr(self._llm, "llm_engine"):
                cache_info = self._llm.llm_engine.get_cache_config()
                if cache_info:
                    # Rough estimate: num_gpu_blocks * block_size * dtype_bytes * 2 (K+V)
                    pass
            return stats
        except Exception:
            return MemoryStats()
