# Profiling

## Tools Available on Rosie

| Tool | Location | Status | Notes |
|------|----------|--------|-------|
| Nsight Systems (nsys) | `/usr/local/cuda/bin/nsys` inside `msoe-pytorch-25.05-py3.sif` | ✅ Working | Full kernel timeline, CUDA API trace |
| Nsight Compute (ncu) | `/usr/local/bin/ncu` inside `msoe-pytorch-25.05-py3.sif` | ❌ ERR_NVGPUCTRPERM | HW counters locked to root |
| nvidia-smi | Host + container | ✅ Working | GPU utilization, memory |
| torch.profiler | Python in-process | ✅ Working | No root required |

**ERR_NVGPUCTRPERM**: Rosie restricts NVIDIA hardware performance counter access to
root/admin. `ncu` can connect to processes but cannot collect SM throughput, DRAM
bandwidth, or occupancy metrics. Contact HPC admin to enable per-job access:
`sudo sh -c 'echo 0 > /proc/driver/nvidia/params/RmProfilingAdminOnly'`

## Nsight Systems Usage

### Quick Profile

```bash
# Submit via SLURM (recommended)
sbatch profiling/nsight-systems/profile_nsys.sh

# Environment variables:
ENGINE=pytorch           # pytorch | vllm
INPUT_TOKENS=512
OUTPUT_TOKENS=128
BATCH_SIZE=1
REQUESTS=1               # Keep low for profiling
WARMUP=0                 # Set to 0 to capture all kernels
LAUNCH_SKIP=0            # Skip first N CUDA launches (skip model load)
LAUNCH_COUNT=0           # Capture N launches (0 = unlimited)
```

### What nsys Captures

- **GPU kernel timeline**: every CUDA kernel with start/end timestamps, block/grid dims
- **CUDA API trace**: `cudaLaunchKernel`, `cudaDeviceSynchronize`, memcpy
- **NVTX annotations**: torch.profiler regions if `with torch.profiler.record_function()`
- **Memory transfers**: H2D/D2H copies with size and duration

### Reading Results

nsys produces `.nsys-rep` and `.sqlite` files:

```bash
# Open in Nsight Systems GUI (needs X11/remote desktop)
nsys-ui profiling/nsight-systems/profile_pytorch_in512_out128_bs1_297168.nsys-rep

# CLI stats summary
nsys stats profile_pytorch_in512_out128_bs1_297168.nsys-rep

# Query SQLite directly for kernel breakdown
python3 - <<'EOF'
import sqlite3
db = 'profiling/nsight-systems/profile_pytorch_in512_out128_bs1_297168.sqlite'
conn = sqlite3.connect(db)
rows = conn.execute('''
  SELECT s.value, count(*), round(sum(k.end-k.start)/1e6,1) as total_ms
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON s.id = k.demangledName
  GROUP BY s.value ORDER BY total_ms DESC LIMIT 15
''').fetchall()
for r in rows: print(f'{r[2]:8.1f}ms  {r[1]:6d}x  {r[0][:70]}')
conn.close()
EOF
```

## Profiling Results Summary

Profiles collected: 4 runs (jobs 297166–297169)

### Kernel Breakdown: in=512, out=128, bs=1, pytorch engine

```
Total GPU kernel time: 6,137 ms

Kernel                                           Calls   Total   Pct
nvjet_tst_256x8_64x6_4x1_v_bz_TNT              42,672  2,125ms  34.6%  GEMM (decode)
nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT       21,336  1,004ms  16.4%  GEMM split-K
nvjet_tst_384x8_64x4_2x1_v_bz_TNT                 762    272ms   4.4%  GEMM (large)
elementwise_kernel (activations)                43,344    260ms   4.2%  SiLU/RoPE/etc
nvjet_tst_64x8_64x16_4x1_v_bz_TNT              21,336    252ms   4.1%  GEMM
flash_fwd_splitkv_kernel                        21,336    162ms   2.6%  Flash Attention decode
---------
nvjet GEMM total:   4,179ms  (68.1%)
Flash Attention:      254ms  ( 4.1%)
```

### Kernel Breakdown: in=4096, out=32, bs=1, pytorch engine

```
Total GPU kernel time: 1,727 ms

nvjet GEMM total:    940ms  (54.4%)  ← lower share (fewer decode steps)
Flash Attention:     159ms  ( 9.2%)  ← higher share (longer sequence)
```

### Key Findings

1. **GEMM dominates decode** (68% at bs=1). Decode is memory-bandwidth bound:
   tiny `bz` (batch=1) tiles cannot saturate H100 tensor cores.

2. **FlashAttention grows with sequence length**: 4.1% at in=512 vs 9.2% at in=4096.
   Not the bottleneck at current sequence lengths but would dominate at >16K tokens.

3. **Python dispatch overhead**: 855,648 `cudaLaunchKernel` calls × 5µs = 4.3 seconds
   pure overhead in a ~11-second decode run (~38% of wall time).

4. **Synchronization gaps**: 780 `cudaDeviceSynchronize` × 318µs = 248ms idle time.
   Production runtimes pipeline decode steps to eliminate this.

## Collected Profile Files

```
profiling/nsight-systems/
├── profile_pytorch_in512_out128_bs1_297166.nsys-rep   # First run (reference)
├── profile_pytorch_in512_out128_bs1_297166.sqlite
├── profile_pytorch_in4096_out32_bs1_297167.nsys-rep   # First long-prompt run
├── profile_pytorch_in4096_out32_bs1_297167.sqlite
├── profile_pytorch_in512_out128_bs1_297168.nsys-rep   # Clean re-run (used for analysis)
├── profile_pytorch_in512_out128_bs1_297168.sqlite
├── profile_pytorch_in4096_out32_bs1_297169.nsys-rep   # Clean long-prompt (used for analysis)
└── profile_pytorch_in4096_out32_bs1_297169.sqlite
```
