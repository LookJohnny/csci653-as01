# SLURM Training Scripts

Organized training pipeline for the hot-seller prediction system.

## 🚀 Quick Start

### Run Everything (Recommended)
```bash
sbatch run_all.slurm
```

This submits all jobs with proper dependencies. Jobs run automatically in the correct order.

---

## 📂 Directory Structure

```
slurm/
├── 01_data_preparation/      # Prepare datasets (REQUIRED FIRST)
├── 02_baseline_models/       # Simple models (70-80% accuracy)
├── 03_advanced_models/       # Deep learning (85-93% accuracy)
├── 04_boosting_models/       # XGBoost, LightGBM (88-91% accuracy)
├── 05_ensemble/              # Final ensemble (93-95% accuracy)
└── 06_distributed/           # Multi-GPU/Multi-node training
```

---

## 📝 Training Stages

### Stage 1: Data Preparation ⏱️ 30-60 min
```bash
sbatch 01_data_preparation/prepare_data.slurm
```
**Output**: `panel_weekly.csv`, `panel_enhanced.csv`

### Stage 2: Baseline Models ⏱️ 6-10 hours
```bash
sbatch 02_baseline_models/train_transformer.slurm    # 75-80% acc
sbatch 02_baseline_models/train_autots.slurm         # 70-75% acc
```

### Stage 3: Advanced Models ⏱️ 12-30 hours
```bash
sbatch 03_advanced_models/train_bert_enhanced.slurm  # 85-88% acc
sbatch 03_advanced_models/train_transformer_bert.slurm  # 86-89% acc
sbatch 03_advanced_models/train_gnn_multitask.slurm  # 87-90% acc
sbatch 03_advanced_models/train_ultimate.slurm       # 92-93% acc (2 GPUs)
```

### Stage 4: Boosting Models ⏱️ 1-3 hours
```bash
sbatch 04_boosting_models/train_xgboost.slurm        # 88-90% acc
sbatch 04_boosting_models/train_lightgbm.slurm       # 89-91% acc
```

### Stage 5: Ensemble ⏱️ 4-6 hours
```bash
sbatch 05_ensemble/train_ensemble.slurm              # 93-95% acc ✨
```
**Run this AFTER all other models complete!**

---

## ⚡ Performance Options

### Standard Training (1 GPU per model)
```bash
sbatch run_all.slurm
```
- **Total Time**: ~30-36 hours (parallel execution)
- **Resources**: 1-2 GPUs, 32 CPUs

### Fast Training (Multi-GPU)
```bash
sbatch 06_distributed/train_multi_gpu.slurm
```
- **Speedup**: 1.8-2x faster (2-4 GPUs)
- **Total Time**: ~15-20 hours

### Ultra-Fast Training (Multi-Node)
```bash
sbatch 06_distributed/train_multi_node.slurm
```
- **Speedup**: 6x faster (16 GPUs across 4 nodes)
- **Total Time**: ~5-8 hours

---

## 📊 Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch specific job
tail -f /home1/yliu0158/amazon2023/amazon23/logs/<job_name>_*.out

# Cancel jobs
scancel <JOB_ID>
scancel -u $USER  # Cancel all
```

---

## 🎯 Expected Results

| Model | Accuracy | Time | Resources |
|-------|----------|------|-----------|
| Baseline | 70-80% | ~8h | 1 GPU or 16 CPUs |
| Advanced | 85-93% | ~18h | 1-2 GPUs |
| Boosting | 88-91% | ~2h | 32 CPUs |
| **Ensemble** | **93-95%** | ~5h | 1 GPU |

---

## 📚 Full Documentation

See [SLURM_TRAINING_GUIDE.md](../SLURM_TRAINING_GUIDE.md) for complete details.

---

**Ready to train? Run:**
```bash
sbatch run_all.slurm
```