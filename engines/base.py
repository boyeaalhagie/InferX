"""
Abstract base class for all InferX inference engines.

Every engine (PyTorch, vLLM, TensorRT-LLM) implements this interface.
The harness bench_inference only ever calls these methods — it never
imports engine-specific code directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationResult:
    """Per-batch timing and output from one generate() call."""
    texts: List[str]                # generated texts, one per prompt in batch
    ttft_ms: float                  # time to first token (ms)
    itl_ms_list: List[float]        # per-inter-token latency values (ms)
    total_latency_ms: float         # wall time from call to last token (ms)
    output_token_counts: List[int]  # actual tokens generated per request


@dataclass
class MemoryStats:
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    kv_cache_gb: float = 0.0


class InferenceEngine(ABC):
    """
    Abstract inference engine.

    Subclasses must implement: load(), generate(), teardown().
    Optionally override: get_memory_stats().

    The tokenizer attribute (if set after load()) is used by the harness
    to build token-count-calibrated prompts.
    """

    def __init__(self, model_path: str, dtype: str = "bf16"):
        self.model_path = model_path
        self.dtype = dtype
        self.tokenizer = None  # set by subclass after load()

    @abstractmethod
    def load(self) -> None:
        """Load the model and any engine-specific runtime into GPU memory."""
        ...

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        **kwargs,
    ) -> GenerationResult:
        """
        Run inference on a batch of prompts.

        Args:
            prompts:        List of input strings (one per batch element).
            max_new_tokens: Maximum tokens to generate per prompt.
            **kwargs:       Engine-specific generation parameters.

        Returns:
            GenerationResult with timing fields populated.

        Timing contract:
            ttft_ms       = wall time from generate() entry to first token emitted
            itl_ms_list   = list of (t[i+1] - t[i]) for consecutive decode steps
            total_latency_ms = wall time from generate() entry to last token
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Release GPU memory and clean up resources."""
        ...

    def get_memory_stats(self) -> MemoryStats:
        """Return current GPU and KV-cache memory usage. Override in subclass."""
        try:
            import torch
            return MemoryStats(
                gpu_allocated_gb=torch.cuda.memory_allocated() / 1e9,
                gpu_reserved_gb=torch.cuda.memory_reserved() / 1e9,
            )
        except Exception:
            return MemoryStats()
