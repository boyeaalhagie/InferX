"""
PyTorch + torch.compile engine — CUDA Graphs experiment.

Extends the baseline PyTorchEngine with torch.compile(mode="reduce-overhead"),
which captures repeated kernel sequences as CUDA graphs during the decode loop.
This eliminates Python interpreter overhead and CPU-side kernel launch latency
for each autoregressive step.

Experiment F: compare this engine vs pytorch baseline to isolate
the CPU launch overhead contribution to decode latency.
"""

import torch

from engines.pytorch.engine import PyTorchEngine


class PyTorchCompiledEngine(PyTorchEngine):
    """
    Identical to PyTorchEngine but with torch.compile applied after load().

    torch.compile(mode="reduce-overhead") uses CUDA graphs internally:
      - After a few warmup steps the kernel sequence is captured
      - Subsequent decode steps replay the graph, skipping Python + CUDA API overhead
      - This is what production runtimes (TRT-LLM, vLLM) leverage internally

    The tradeoff: first-call compilation overhead (amortized over many tokens),
    plus constraint that graph inputs must remain the same shape (guaranteed here
    since each decode step is always batch_size × 1 token).
    """

    def load(self) -> None:
        super().load()
        print("[pytorch_compiled] Applying torch.compile(mode='reduce-overhead')...")
        # reduce-overhead: enables CUDA graph capture for the forward pass
        # fullgraph=False allows partial compilation (safe for HF models with dynamic control)
        self.model = torch.compile(
            self.model,
            mode="reduce-overhead",
            fullgraph=False,
        )
        print("[pytorch_compiled] torch.compile applied. "
              "First generate() call will trigger compilation (expect ~60s delay).")
