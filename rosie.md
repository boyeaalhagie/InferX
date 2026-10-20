# Rosie HPC — Agent Reference Guide
## For TerraCare Development on MSOE's Supercomputer

---

## 1. Cluster Overview

Rosie is MSOE's GPU supercomputer. All GPU compute must go through the SLURM job scheduler — never run heavy jobs directly on the login/management node.

**Hardware:**

| Partition | Nodes | GPUs | Notes |
|---|---|---|---|
| `dgxh100` | 2 (dh-dgxh100-[1-2]) | 8x H100 per node (16 total) | **Use this for TerraCare** |
| `dgx` | 3 (dh-dgx1-[1-3]) | 8x V100 per node | Fallback if H100 unavailable |
| `teaching` | 20 (dh-node[1-20]) | 4x T4 per node | Dev/test only |
| `batch` | 20 (dh-node[1-20]) | 4x T4 per node | Non-interactive batch |
| `highmem` | 2 (dh-node[19-20]) | 4x T4 per node | High memory jobs |

## 3. SLURM Quick Reference

```bash
sinfo                              # View partitions and node status
squeue                             # View all jobs in queue
squeue -u boyea@ad.msoe.edu        # View your jobs only
scancel JOBID                      # Cancel a job
sacct -u boyea@ad.msoe.edu         # View job history
```

---

## 4. SRUN — Run Simple Tasks

Use `srun` for quick commands and interactive sessions. Always run from the management node.

```bash
# Run a command on the default partition
srun hostname

# Check GPUs on H100 node
srun --partition=dgxh100 --gres=gpu:h100:1 nvidia-smi

# Check GPUs on V100 node
srun --partition=dgx --gres=gpu:v100:8 nvidia-smi

# Interactive session (teaching nodes only — NOT dgxh100)
srun --pty --partition=teaching bash

# Interactive session with T4 GPU
srun --pty --gres=gpu:t4:1 --partition=teaching bash

# Run a Python script with 1 H100
srun --partition=dgxh100 --gres=gpu:h100:1 python /data/ai_club/TerraCare/scripts/run_hdbscan.py

# Email alerts
srun --mail-user=boyea@ad.msoe.edu --mail-type=BEGIN,END,FAIL --partition=dgxh100 --gres=gpu:h100:8 python script.py
```

> **Do NOT use `--pty` (interactive) on dgxh100 nodes. Use `sbatch` for all H100 jobs.**

---

## 5. SBATCH — Submit Batch Jobs

Use `sbatch` for all H100 jobs. Write a script, submit it, check the output file.

```bash
sbatch job_script.sh               # Submit a job
cat job_JOBID.out                  # View output
tail -f job_JOBID.out              # Follow output live
less job_JOBID.out                 # Page through output
```

---

## 6. SBATCH Script Templates

### Template 1 — H100 GPU Job (TerraCare Standard)

```bash
#!/bin/bash

#SBATCH --job-name="terracare_hdbscan"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-gpu=8

## SCRIPT START
srun hostname; pwd; date
srun nvidia-smi
srun python /data/ai_club/TerraCare/scripts/run_hdbscan.py
## SCRIPT END
```

### Template 2 — H100 Job with Singularity Container

```bash
#!/bin/bash

#SBATCH --job-name="terracare_cnn"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-gpu=8

CONTAINER="/data/containers/msoe-tensorflow.sif"
PYTHON_FILE="/data/ai_club/TerraCare/scripts/train_resnet.py"

## SCRIPT START
srun hostname; pwd; date
srun singularity exec --nv -B /data:/data ${CONTAINER} python3 ${PYTHON_FILE}
## SCRIPT END
```

### Template 3 — CPU-Only Job (Data Prep / Ingest)

```bash
#!/bin/bash

#SBATCH --job-name="terracare_ingest"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=batch
#SBATCH --nodes=1

## SCRIPT START
srun python /data/ai_club/TerraCare/scripts/ingest_buildings.py
## SCRIPT END
```

### Template 4 — T4 GPU Job (Dev/Test on Small Data)

```bash
#!/bin/bash

#SBATCH --job-name="terracare_test"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyea@ad.msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4

## SCRIPT START
srun python /data/ai_club/TerraCare/scripts/test_pipeline.py
## SCRIPT END
```

---

## 7. Singularity Containers

Rosie uses Singularity (not Docker). Containers are `.sif` files. Rosie runs **Singularity version 3.3**.

### Pre-built containers available

| Image | Path |
|---|---|
| Ubuntu 18.04 | `/data/containers/ubuntu_18.04.sif` |
| Ubuntu 20.04 | `/data/containers/ubuntu_20.04.sif` |
| TensorFlow | `/data/containers/msoe-tensorflow.sif` |

### Key flags

| Flag | Purpose |
|---|---|
| `--nv` | Mount NVIDIA GPUs into container — always use for GPU jobs |
| `-B /data:/data` | Bind mount cluster data pool into container — always include |

### Common commands

```bash
# Run a command inside a container
singularity exec -B /data:/data /data/containers/ubuntu_20.04.sif cat /etc/os-release

# Run with GPU access
singularity exec --nv -B /data:/data /data/containers/msoe-tensorflow.sif nvidia-smi

# Interactive CPU shell
singularity shell -B /data:/data /data/containers/ubuntu_20.04.sif

# Interactive GPU shell (via srun on teaching node)
srun --pty --gres=gpu:t4:1 singularity shell --nv -B /data:/data /data/containers/msoe-tensorflow.sif

# Build a custom container from DockerHub
singularity build MyContainer.sif docker://ubuntu:latest
```

### Custom container definition file (example for RAPIDS)

```
Bootstrap: docker
From: tensorflow:19.07-py3
Registry: nvcr.io
Namespace: nvidia

%post
    apt-get update && apt-get -y upgrade
    apt-get -y install python3 python3-pip
    python3 -m pip install --upgrade pip
    python3 -m pip install cuml cupy pandas pyarrow torch
```

---


---

---

## 9. Tips

- Output saves as `job_JOBID.out` in the directory where you ran `sbatch`
- Use `tail -f job_JOBID.out` to watch a running job live
- Use `sinfo` to confirm dgxh100 nodes are available before submitting
- Email alerts (`--mail-type=ALL`) are strongly recommended for long H100 jobs
- Web portal available at: https://dh-ood.hpc.msoe.edu