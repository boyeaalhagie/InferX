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
        print("[pytorch_compiled] Applying torch.compile(backend='cudagraphs')...")
        # backend="cudagraphs": captures the forward pass as a CUDA graph.
        # Unlike mode="reduce-overhead" (which uses Triton), the cudagraphs
        # backend records GPU kernel sequences directly without JIT compilation.
        # This eliminates CPU-side kernel dispatch overhead per decode step
        # without requiring Triton or any additional compiler dependencies.
        self.model = torch.compile(
            self.model,
            backend="cudagraphs",
            fullgraph=False,
        )
        print("[pytorch_compiled] torch.compile applied (cudagraphs backend). "
              "First generate() call captures the graph (~10s).")
