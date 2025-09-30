# Parallel Training Status

## Submitted Jobs

### Job 3015108: Multi-GPU Training (2 GPUs, Single Node)
- **Status**: Pending (waiting for GPU resources)
- **Configuration**: 
  - 1 node × 2 GPUs (any available: A100/A40/L40S)
  - 32 CPUs, 256GB RAM
  - 24 hour time limit
- **Expected Performance**:
  - Training time: ~6-7 hours (vs 12h single GPU)
  - Speedup: 1.8-1.9x
  - Throughput: ~80 samples/sec
- **Technology**:
  - PyTorch DistributedDataParallel (DDP)
  - NCCL backend for GPU communication
  - Mixed precision training (FP16)
  - 16 parallel data loaders

### Available Multi-Node Configuration (Not Yet Submitted)
- **Script**: `train_multi_node.slurm`
- **Configuration**: 2 nodes × 2 GPUs = 4 GPUs total
- **Expected Performance**:
  - Training time: ~3-4 hours
  - Speedup: 3.4-3.8x
  - Throughput: ~160 samples/sec
- **When to use**: After multi-GPU job completes successfully

## HPC Resources Detected

### Available GPUs:
- **A100-80GB**: 2 GPUs per node, 64 cores, InfiniBand HDR/NDR
- **A100-40GB**: 2 GPUs per node, 64 cores, InfiniBand
- **A40**: 2 GPUs per node, 48GB memory
- **L40S**: 3 GPUs per node, 48GB memory (latest Lovelace)
- **V100/P100**: Legacy options

### Parallel Infrastructure:
- ✅ CUDA 12.6.3
- ✅ OpenMPI 5.0.5
- ✅ InfiniBand HDR/NDR (200 Gbps inter-node)
- ✅ NCCL (NVIDIA Collective Communications)
- ✅ 64 CPU cores per node (AMD EPYC 7513/9534)

## Performance Expectations

| Configuration | GPUs | Time | Speedup | Efficiency | GPU-Hours |
|---------------|------|------|---------|------------|-----------|
| Single GPU | 1 | 12h | 1.0x | 100% | 12 |
| Multi-GPU (current) | 2 | 6.5h | 1.85x | 92% | 13 |
| Multi-Node | 4 | 3.5h | 3.43x | 86% | 14 |
| Multi-Node Large | 8 | 2h | 6.0x | 75% | 16 |

## Next Steps

1. ✅ **Multi-GPU job submitted** (Job 3015108) - waiting for resources
2. ⏳ **Monitor multi-GPU training** - will start when GPUs available
3. 📊 **Evaluate speedup** - compare with single GPU baseline
4. 🚀 **Scale to multi-node** - if 2-GPU shows good efficiency

## Commands to Monitor

```bash
# Check job status
squeue -u yliu0158

# Watch multi-GPU training output
tail -f logs/train_multi_gpu_3015108.out

# Check GPU utilization (when running)
srun --jobid=3015108 nvidia-smi

# Cancel job if needed
scancel 3015108
```

## Optimization Applied

### NCCL Settings:
```bash
export NCCL_IB_DISABLE=0          # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5       # GPU Direct RDMA
export NCCL_P2P_LEVEL=NVL         # Use NVLink
```

### PyTorch DDP:
- DistributedSampler for data partitioning
- Automatic gradient synchronization
- Mixed precision with GradScaler

### Data Loading:
- 16 parallel workers per GPU
- Persistent workers (no restart overhead)
- Pin memory for faster GPU transfer
- Prefetch factor = 4

---
Last Updated: $(date)
