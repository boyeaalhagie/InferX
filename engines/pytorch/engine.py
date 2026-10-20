"""
PyTorch inference engine — intentionally naive baseline.

Uses HuggingFace transformers with a manual token-by-token decode loop
so we can capture per-token timestamps for accurate TTFT and ITL.

No flash attention, no torch.compile, no optimizations — that's the point.
This is the worst-case baseline every other engine is compared against.
"""

import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.base import InferenceEngine, GenerationResult, MemoryStats

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class PyTorchEngine(InferenceEngine):

    def __init__(self, model_path: str, dtype: str = "bf16"):
        super().__init__(model_path=model_path, dtype=dtype)
        self.model = None
        self._torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

    def load(self) -> None:
        print(f"[pytorch] Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[pytorch] Loading model ({self.dtype}) from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        print(f"[pytorch] Model loaded. "
              f"Params: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        **kwargs,
    ) -> GenerationResult:
        assert self.model is not None, "Call load() first"

        device = next(self.model.parameters()).device

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]

        # Manual token-by-token generation for accurate per-step timing.
        # We use greedy decoding (argmax) for reproducibility.
        past_key_values = None
        generated_ids = [[] for _ in range(batch_size)]
        token_timestamps: list[float] = []
        finished = [False] * batch_size

        # Synchronize GPU before starting the clock
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        t_first_token: float | None = None

        current_input_ids = input_ids
        current_attention_mask = attention_mask

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]  # (batch, vocab)
                next_tokens = torch.argmax(logits, dim=-1)  # greedy

                torch.cuda.synchronize()
                t_now = time.perf_counter()

                if t_first_token is None:
                    t_first_token = t_now
                token_timestamps.append(t_now)

                for i in range(batch_size):
                    tok = next_tokens[i].item()
                    if not finished[i]:
                        generated_ids[i].append(tok)
                        if tok == self.tokenizer.eos_token_id:
                            finished[i] = True

                if all(finished):
                    break

                # Prepare next step: only the new token, extend attention mask
                current_input_ids = next_tokens.unsqueeze(-1)
                ones = torch.ones(
                    (batch_size, 1),
                    dtype=current_attention_mask.dtype,
                    device=device,
                )
                current_attention_mask = torch.cat(
                    [current_attention_mask, ones], dim=1
                )

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        ttft_ms = (t_first_token - t_start) * 1000.0 if t_first_token else 0.0
        total_latency_ms = (t_end - t_start) * 1000.0

        # Inter-token latencies from the timestamp list
        itl_ms_list: list[float] = []
        if len(token_timestamps) > 1:
            itl_ms_list = [
                (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
                for i in range(1, len(token_timestamps))
            ]

        texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        return GenerationResult(
            texts=texts,
            ttft_ms=ttft_ms,
            itl_ms_list=itl_ms_list,
            total_latency_ms=total_latency_ms,
            output_token_counts=[len(ids) for ids in generated_ids],
        )

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        print("[pytorch] Model unloaded, VRAM released.")

    def get_memory_stats(self) -> MemoryStats:
        return MemoryStats(
            gpu_allocated_gb=torch.cuda.memory_allocated() / 1e9,
            gpu_reserved_gb=torch.cuda.memory_reserved() / 1e9,
        )
