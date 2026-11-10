# **Inference Performance Engineering Platform**

### *A performance study and optimization framework for production-scale Transformer inference on NVIDIA GPUs*

This is the name I would use.

It sounds like engineering work rather than:

> "I built an LLM."
> 

The core question is:

> **How do we make Transformer inference faster, more memory-efficient, and more scalable on NVIDIA GPUs?**
> 

---

# 1. What are we actually building?

We're building an **inference performance engineering platform** around a real open-source Transformer model.

The system will let us run controlled experiments across:

- model size
- sequence length
- batch size
- input/output token lengths
- precision
- KV-cache configuration
- concurrency
- inference engine
- GPU

and measure:

- TTFT
- inter-token latency
- end-to-end latency
- throughput
- GPU utilization
- GPU memory
- KV-cache consumption
- scaling behavior

The basic architecture:

```
                    LLM Workload
                         │
                         ▼
                Inference Harness
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
        Baseline Path           Optimized Path
        PyTorch                 vLLM / TRT-LLM
             │                       │
             └───────────┬───────────┘
                         ▼
                    NVIDIA GPU
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         Nsight Systems       Nsight Compute
              │                     │
              └──────────┬──────────┘
                         ▼
                  Metrics Engine
                         │
                         ▼
                Performance Report
```

Notice something important:

**We're not building an LLM.**

We're building the **engineering infrastructure around inference**.

That's much closer to the role you're preparing for.

---

# 2. The central engineering problem

LLM inference has two fundamentally different phases:

```
              Request
                 │
                 ▼
             PREFILL
                 │
                 ▼
          First generated token
                 │
                 ▼
              DECODE
                 │
          ┌──────┴──────┐
          │             │
       token 1       token 2 ...
```

The performance characteristics of these phases are different.

That's something we're going to **measure**, not simply memorize.

For example, vLLM's current documentation explicitly describes prefill as more compute-oriented and decode as more memory-oriented, and its scheduling mechanisms attempt to balance the two. vLLM

That gives us our first major research question:

> **How does the bottleneck change between prefill and decode?**
> 

---

# 3. The project has five major areas

## A. Inference benchmarking

Establish a rigorous baseline.

## B. Memory/KV-cache engineering

Understand why memory becomes a constraint.

## C. Scheduling and batching

Investigate how concurrency changes throughput and latency.

## D. Kernel/runtime optimization

Connect the work directly to Project 1.

## E. Production inference engines

Compare:

- PyTorch
- vLLM
- TensorRT-LLM

where the ROSIE environment supports them.

---

# 4. Start with ONE model

This is important.

We are **not** going to benchmark 15 models.

Choose one model appropriate for the actual ROSIE hardware.

For example, depending on the GPU/memory available:

```
Qwen
Llama
Mistral
```

The exact model should be chosen **after we inspect your ROSIE software/GPU environment**.

The model needs to be:

- publicly available
- reproducible
- small enough to run comfortably
- large enough to expose real inference behavior

---

# 5. Build the inference harness

This is one of the most important parts.

We don't want:

```
python run_model.py
```

and then manually record numbers.

We want:

```
./bench_inference \
    --model <model> \
    --engine pytorch \
    --input-tokens 2048 \
    --output-tokens 256 \
    --batch-size 8 \
    --requests 100
```

And the system produces a structured record.

For example:

```
{
  "model": "...",
  "engine": "pytorch",
  "gpu": "...",
  "dtype": "bf16",
  "batch_size": 8,
  "input_tokens": 2048,
  "output_tokens": 256,
  "requests": 100,
  "ttft_ms": 0.0,
  "itl_ms": 0.0,
  "latency_ms": 0.0,
  "tokens_per_second": 0.0,
  "gpu_memory_gb": 0.0
}
```

Again:

**actual numbers come from ROSIE.**

---

# 6. The metrics

This project needs much better metrics than simply:

> "It generated text in 2 seconds."
> 

### TTFT

**Time to First Token**

```
request
   │
   ├───────────────┐
                   ▼
              first token
```

This captures the initial response delay.

---

### Inter-token latency

How long does it take between generated tokens?

```
token 1 → token 2 → token 3 → token 4
    ↑         ↑         ↑
   ITL       ITL       ITL
```

---

### End-to-end latency

Total request duration.

---

### Throughput

For example:

```
tokens / second
```

But we will be precise about whether we're measuring:

- input tokens
- output tokens
- total tokens
- requests/sec

---

### GPU utilization

What is the GPU actually doing?

---

### GPU memory

How much memory is being consumed?

---

### KV-cache memory

This becomes particularly important.

---

# 7. KV Cache becomes a major investigation

For autoregressive Transformers, previous key/value states are stored so they don't have to be recomputed every generation step.

Conceptually:

```
Prompt
  ↓
Transformer
  ↓
K,V
  ↓
┌─────────────────────┐
│     KV Cache        │
│                     │
│ previous tokens     │
│ previous tokens     │
│ previous tokens     │
└─────────────────────┘
          ↓
       Decode
```

NVIDIA's TensorRT documentation describes KV-cache tensors as persistent storage used during Transformer inference, and TensorRT-LLM exposes optimized KV-cache mechanisms for LLM workloads. NVIDIA Docs

We're going to experimentally investigate:

> **How does KV-cache memory scale with sequence length and concurrency?**
> 

---

# 8. KV-cache experiment

We vary:

```
sequence length
        ×
number of concurrent requests
```

For example:

```
512
1024
2048
4096
8192
```

and:

```
1
2
4
8
16
32
```

where hardware permits.

Measure:

```
GPU memory
KV-cache memory
throughput
TTFT
ITL
```

Then visualize:

```
KV Cache Memory
       ↑
       │
       │             /
       │          /
       │       /
       │    /
       │ /
       └────────────────→
            Sequence Length
```

The important part is explaining the shape.

---

# 9. Batch-size study

Now:

```
Batch = 1
Batch = 2
Batch = 4
Batch = 8
Batch = 16
...
```

Measure:

```
latency
throughput
GPU utilization
memory
TTFT
```

NVIDIA's TensorRT performance guidance explicitly identifies batching as one of the most important inference optimizations because it increases parallel work and can improve GPU efficiency. NVIDIA also notes that very small batches can be inefficient for matrix operations, while the optimal batch can depend on hardware and caching behavior. NVIDIA Docs

So our question becomes:

> **Where does batching stop helping?**
> 

That is much more interesting than:

> "Batching makes inference faster."
> 

---

# 10. Continuous / in-flight batching

Now we're getting closer to production inference.

Instead of:

```
Request 1
████████████████
                 Request 2
                 ███████████████
```

we want to investigate scheduling requests while generation is already happening.

Conceptually:

```
GPU execution

Request A: ███████████████████
Request B:       █████████████
Request C:            ███████████████
Request D:                    ██████████
```

TensorRT-LLM specifically supports **in-flight batching**, while vLLM has sophisticated scheduling around active requests and KV-cache availability. NVIDIA Docs

We should benchmark:

> Static batching vs dynamic/in-flight batching.
> 

# 11. Prefill vs Decode

This becomes one of the project's strongest experiments.

Create workloads such as:

### Short prompt

```
128 input tokens
```

### Medium prompt

```
1,024 input tokens
```

### Long prompt

```
4,096+
```

Then measure:

```
Prefill latency
Decode latency
TTFT
ITL
GPU utilization
memory bandwidth
```

The goal:

> **Understand how workload characteristics change the GPU bottleneck.**
> 

This directly builds on Project 1's performance-analysis methodology.

---

# 12. Precision study

Now investigate:

```
FP32
FP16
BF16
INT8
```

Potentially additional formats depending on the actual GPU and supported software.

But this is **not** simply:

> "INT8 is faster."
> 

We're going to measure:

```
Performance
+
Memory
+
Accuracy
```

because inference optimization always involves tradeoffs.

NVIDIA's TensorRT documentation explicitly discusses precision control and accuracy/performance tradeoffs, including INT8 and newer low-precision approaches. NVIDIA Docs

---

# 13. Accuracy evaluation

This is critical.

Suppose:

```
FP16:
100 tokens/sec

INT8:
170 tokens/sec
```

Great.

But:

> Did model quality change?
> 

We therefore create a small evaluation set and compare:

- output similarity
- task accuracy
- perplexity where appropriate
- exact-match metrics for deterministic tasks where appropriate

We don't need a giant evaluation framework.

We need enough to make responsible performance claims.

---

# 14. PyTorch vs vLLM vs TensorRT-LLM

Now the project becomes very interesting.

We can compare:

```
             Model
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
   PyTorch    vLLM   TRT-LLM
       │       │        │
       └───────┼────────┘
               ▼
             GPU
               │
               ▼
          Measurements
```

But we don't compare them with one random command.

We establish identical workloads:

```
model
input tokens
output tokens
batch/concurrency
GPU
precision
```

Then run each engine.

---

# 15. What we're actually trying to learn

Suppose we eventually get something like:

```
                Throughput
PyTorch          X
vLLM             X
TensorRT-LLM     X
```

The interesting question is:

> **Why?**
> 

Maybe one system has:

- better batching
- better KV-cache management
- kernel fusion
- optimized attention
- Tensor Core utilization
- CUDA graphs
- better scheduling
- reduced Python overhead

That investigation becomes the project.

---

# 16. TensorRT-LLM

This should be a **major part**, because it directly aligns with NVIDIA's ecosystem.

NVIDIA describes TensorRT-LLM as its specialized toolkit for optimizing LLM inference on NVIDIA GPUs, with features including **in-flight batching, paged KV caching, quantization, and multi-GPU/multi-node support**. NVIDIA Docs

We will investigate rather than blindly use:

### Engine building

### Quantization

### KV cache

### batching

### Tensor Core utilization

### multi-GPU if hardware allows

---

# 17. vLLM

vLLM gives us another valuable reference point.

Its current documentation exposes tuning around:

- KV-cache capacity
- batching
- tensor parallelism
- pipeline parallelism
- chunked prefill
- scheduling
- async execution. vLLM

This lets us ask:

> **How do different inference engines solve the same systems problems?**
> 

That's a strong engineering question.

---

# 18. CUDA Graphs

Another experiment:

```
Normal launch
       vs
CUDA Graph
```

Why?

Because repeated kernel launches can introduce CPU-side launch overhead.

NVIDIA's TensorRT performance documentation explicitly identifies CUDA graphs as an inference optimization technique alongside batching and multi-streaming. NVIDIA Docs

Measure:

```
latency
throughput
CPU overhead
GPU utilization
```

---

# 19. Kernel-level connection to Project 1

This is where the two projects connect.

Project 1:

> **Why is this kernel slow?**
> 

Project 2:

> **Why is the entire Transformer inference system slow?**
> 

If Nsight Systems shows:

```
attention kernel
██████████
```

we can drill into that kernel with Nsight Compute.

Then:

```
Project 2
      ↓
Find bottleneck
      ↓
Identify kernel
      ↓
Project 1 analysis
      ↓
Nsight Compute
      ↓
GPU-level explanation
```

That's exactly how I want the projects connected.

---

# 20. Nsight Systems

For this project, **Nsight Systems becomes especially important**.

We're now looking at the entire application timeline:

```
CPU
──────────────────────────────

GPU
    ┌─────┐ ┌─────────┐ ┌────┐
────┘     └─┘         └─┘  └───

Memory transfers
      ↓↓↓          ↓↓↓

Kernels
       ███ █████ ████ ███
```

We want to answer:

> Is the GPU actually busy?
> 

> Are there gaps?
> 

> Is CPU orchestration limiting throughput?
> 

> Are memory transfers overlapping with computation?
> 

> Are kernels launching too frequently?
> 

That is a systems question.

---

# 21. Multi-GPU — but only if ROSIE makes sense

I **do not** want to force multi-GPU into this project.

If your available ROSIE environment gives us appropriate multi-GPU access, then we can investigate:

```
1 GPU
2 GPU
4 GPU
```

and compare scaling.

TensorRT-LLM supports multi-GPU and multi-node inference. NVIDIA Docs

But if your actual ROSIE allocation doesn't make this practical:

**we skip it.**

One excellent single-GPU investigation is better than fake distributed work.

# 22. The benchmark matrix

Eventually our experiment matrix could look like:

```
Model
 ├── sequence length
 │     ├── 128
 │     ├── 512
 │     ├── 1024
 │     └── 4096
 │
 ├── batch/concurrency
 │     ├── 1
 │     ├── 2
 │     ├── 4
 │     └── 8
 │
 ├── precision
 │     ├── FP16
 │     ├── BF16
 │     └── INT8
 │
 └── engine
       ├── PyTorch
       ├── vLLM
       └── TensorRT-LLM
```

But we **don't run the entire Cartesian product blindly**.

The benchmark harness should allow us to choose targeted experiments based on what we learn.

---

# 23. Performance regression

Just like Project 1:

```
baseline
   ↓
optimization
   ↓
benchmark
   ↓
compare
```

If a change causes:

```
TTFT: +12%
```

the system flags:

> **Latency regression**
> 

If throughput improves:

```
+18%
```

but memory increases:

```
+31%
```

the report should show both.

---

# 24. The performance report

We want reports like:

```
================================================
LLM INFERENCE PERFORMANCE REPORT
================================================

Model:
<model>

GPU:
<actual ROSIE GPU>

Engine:
TensorRT-LLM

Precision:
BF16

Input:
2048 tokens

Output:
256 tokens

Concurrency:
8

------------------------------------------------
PERFORMANCE
------------------------------------------------

TTFT:             X ms
ITL:              X ms
Output Throughput X tok/s
GPU Memory:       X GB

------------------------------------------------
BOTTLENECK
------------------------------------------------

Primary:
<measured finding>

------------------------------------------------
CHANGE
------------------------------------------------

<optimization>

------------------------------------------------
RESULT
------------------------------------------------

Throughput:       +X%
TTFT:             -X%
Memory:           +X%

------------------------------------------------
TRADEOFF
------------------------------------------------

<analysis>
```

This becomes portfolio-quality engineering evidence.

---

# 25. Repository

I'd structure it differently from Project 1:

```
inference-performance/
│
├── README.md
│
├── docs/
│   ├── architecture.md
│   ├── inference-model.md
│   ├── benchmarking.md
│   ├── kv-cache.md
│   ├── batching.md
│   ├── quantization.md
│   └── profiling.md
│
├── models/
│
├── engines/
│   ├── pytorch/
│   ├── vllm/
│   └── tensorrt_llm/
│
├── benchmarks/
│
├── workloads/
│
├── evaluation/
│
├── profiling/
│   ├── nsight-systems/
│   └── nsight-compute/
│
├── results/
│
├── configs/
│
├── scripts/
│
└── reports/
```

---

# 26. The engineering questions

This project should force you to answer questions like:

### Inference

> Why is prefill different from decode?
> 

> Why can decode become memory-bound?
> 

> Why does batch size affect throughput?
> 

> Why can increasing concurrency hurt latency?
> 

### KV cache

> Why does KV-cache memory grow with sequence length?
> 

> Why does KV-cache capacity limit concurrency?
> 

> What happens when KV cache becomes exhausted?
> 

Current vLLM documentation explicitly discusses KV-cache exhaustion, preemption/recomputation, and how increasing cache capacity or changing concurrency affects performance. vLLM

### Systems

> Why are there GPU idle gaps?
> 

> Why might CUDA graphs help?
> 

> Where is CPU overhead?
> 

### GPU

> Which kernel is actually dominating execution?
> 

> Is it compute-bound or memory-bound?
> 

> What did Nsight Compute tell you?
> 

### Runtime

> Why does TensorRT-LLM outperform a basic PyTorch implementation?
> 

> What does fusion accomplish?
> 

> Why does quantization change performance?
> 

Those are the conversations we want.

---

# 27. What we are NOT doing

Again, we're keeping the scope disciplined.

### ❌ Not building a chatbot

### ❌ Not fine-tuning an LLM

### ❌ Not building an agent

### ❌ Not training a model

### ❌ Not implementing an entire inference engine from scratch

### ❌ Not blindly benchmarking 20 models

### ❌ Not throwing every NVIDIA technology into one repository

Instead:

> **One model. One GPU environment. One reproducible inference harness. A small number of carefully designed experiments.**
> 

That's how we get depth.

---

# 28. The final architecture

At the end:

```
             INFERENCE PERFORMANCE
                   PLATFORM
                       │
         ┌─────────────┴─────────────┐
         │                           │
    Model Workload              Benchmark
         │                       Harness
         │                           │
         └─────────────┬─────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       PyTorch        vLLM       TRT-LLM
          │            │            │
          └────────────┼────────────┘
                       ▼
                 NVIDIA GPU
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     Nsight Systems        Nsight Compute
            │                     │
            └──────────┬──────────┘
                       ▼
                Metrics Engine
                       │
                       ▼
              Performance Reports
```

# 29. How it connects to Project 1

This is the key:

### Project 1

**Kernel Performance Engineering Suite**

You learn:

> **GPU-level performance**
> 

```
CUDA
Memory
Warps
SMs
Shared Memory
Registers
Roofline
Nsight Compute
```

↓

### Project 2

**Inference Performance Engineering Platform**

You learn:

> **AI-system-level performance**
> 

```
Transformer
Prefill
Decode
KV Cache
Batching
Scheduling
Quantization
CUDA Graphs
vLLM
TensorRT-LLM
Nsight Systems
```

↓

### Project 3

Then we can build:

**ROSIE Agent**

> **Can an AI agent autonomously perform this entire engineering loop?**
> 

That's the progression.

---

# 30. What makes Project 2 stand out

The strongest part isn't:

> "I used TensorRT-LLM."
> 

Thousands of people can say that.

It's:

> **"I built a reproducible inference-performance framework, characterized the transition between prefill and decode, investigated KV-cache and batching constraints, compared inference runtimes under controlled workloads, profiled system-level execution with Nsight Systems, drilled into critical kernels with Nsight Compute, and quantified the latency/throughput/memory tradeoffs of different optimizations."**
> 

That is the kind of statement that gives an NVIDIA interviewer **multiple directions to drill into**.

And because Project 1 comes first, you'll have the GPU fundamentals underneath it rather than simply knowing how to run a framework.