#!/usr/bin/env python3
"""
Wrapper for trtllm-build that works around two import-time conflicts in
the TRT-LLM 0.21.0 / PyTorch 2.7.1 / TF 2.21.0 stack on Rosie:

1. tensorflow protobuf conflict:
   tensorrt_llm -> quantize_by_modelopt -> AutoProcessor ->
   image_transforms -> `import tensorflow` -> needs protobuf>=6.31.1
   but TRT-LLM installed 4.24.4.
   Fix: pre-populate sys.modules['tensorflow'] with a stub so the import
   is a no-op.

2. torch._dynamo circular import:
   tensorrt_llm -> quantize_by_modelopt -> AutoModel ->
   transformers.modeling_utils -> flex_attention ->
   @torch.compiler.disable -> `import torch._dynamo` triggers _dynamo
   mid-initialization; trace_rules.py accesses torch._dynamo.config
   before __init__.py finishes.
   Fix: explicitly import torch._dynamo (and its config sub-module)
   before anything else touches it, so it is fully initialised.

Usage: python3 trtllm_build_wrapper.py [trtllm-build args...]
"""
import sys
import types
import os

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_JAX', '0')
os.environ.setdefault('KERAS_BACKEND', 'torch')

# Fix 2 FIRST: fully initialize torch._dynamo before tensorflow is in sys.modules.
# torch._dynamo.trace_rules scans sys.modules at init time; a stub module with
# __spec__=None (what types.ModuleType gives) causes ValueError in find_spec.
# Importing _dynamo here — before the tensorflow stub — avoids that scan failure.
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch._dynamo.config  # noqa: E402  -- ensures config attr exists on the pkg

# Fix 1: NOW stub out tensorflow so the later import chain
#   tensorrt_llm -> quantize_by_modelopt -> AutoProcessor ->
#   image_transforms -> `import tensorflow`
# gets our stub instead of the real (broken) tensorflow.
sys.modules['tensorflow'] = types.ModuleType('tensorflow')

# Now it is safe to import TRT-LLM's build command.
from tensorrt_llm.commands.build import main  # noqa: E402

sys.exit(main())
