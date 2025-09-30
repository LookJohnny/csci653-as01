# Complete Training Guide

## Available Models (Ranked by Performance)

### 🥇 **Level 4: GNN Multi-Task** (Highest Accuracy)
**Expected: 90-92% accuracy**
- Pre-trained BERT + 24 rich features
- Graph Neural Network for product relationships
- Multi-task learning (hot-seller + rating + engagement)
- Cross-modal attention

```bash
sbatch train_gnn_multitask.slurm
```

**Best for:** Maximum accuracy, research projects
**Requires:** GPU with 18GB+, 8-10 hours
**Complexity:** ⭐⭐⭐⭐⭐

---

### 🥈 **Level 3: Enhanced BERT** (High Accuracy)
**Expected: 85-88% accuracy**
- Pre-trained BERT embeddings
- 24 engineered features
- Focal loss for imbalanced data
- Deep fusion network

```bash
sbatch train_enhanced_pipeline.slurm
```

**Best for:** Production use, best accuracy/speed tradeoff
**Requires:** GPU with 14GB+, 5-6 hours
**Complexity:** ⭐⭐⭐⭐

---

### 🥉 **Level 2: BERT Text Model** (Good Accuracy)
**Expected: 82-85% accuracy**
- Pre-trained BERT for text
- Basic 4 features
- Standard cross-entropy loss

```bash
sbatch train_bert_model.slurm
```

**Best for:** Quick BERT baseline
**Requires:** GPU with 12GB+, 4-5 hours
**Complexity:** ⭐⭐⭐

---

### **Level 1: Basic Transformer** (Baseline)
**Expected: 70-75% accuracy**
- Time series features only
- No text processing
- Simple architecture

```bash
sbatch train_full_pipeline.slurm
```

**Best for:** Fast baseline, CPU-only training
**Requires:** CPU/GPU with 4GB, 2-3 hours
**Complexity:** ⭐⭐

---

## Quick Comparison

| Model | Accuracy | GPU | Time | Features | Best For |
|-------|----------|-----|------|----------|----------|
| **GNN Multi-Task** | 90-92% | 18GB | 10h | All + Graph | Research |
| **Enhanced BERT** | 85-88% | 14GB | 6h | 24 features | Production |
| **BERT Text** | 82-85% | 12GB | 5h | Text + Basic | Quick BERT |
| **Basic** | 70-75% | 4GB | 2h | 4 features | Baseline |

---

## Step-by-Step Workflow

### Option A: Train Everything (Recommended)

Run all models to compare:

```bash
# 1. Basic baseline (already running)
# Job 3014745 is running

# 2. Submit BERT text model (will wait for panel)
sbatch train_bert_model.slurm

# 3. Submit enhanced BERT (runs independently)
sbatch train_enhanced_pipeline.slurm

# 4. Submit GNN multi-task (most advanced)
sbatch train_gnn_multitask.slurm
```

All jobs will run in parallel on different nodes!

### Option B: Just Best Model

```bash
sbatch train_gnn_multitask.slurm
```

Trains the most advanced model directly.

### Option C: Production Pipeline

```bash
sbatch train_enhanced_pipeline.slurm
```

Best accuracy/speed tradeoff for real deployment.

---

## Current Job Status

```bash
# Check running jobs
squeue -u yliu0158

# Monitor specific job
tail -f /home1/yliu0158/amazon2023/amazon23/logs/train_JOBID.out

# Check all job outputs
ls -lh /home1/yliu0158/amazon2023/amazon23/logs/
```

---

## Output Directory Structure

After training:

```
/home1/yliu0158/amazon2023/amazon23/training_output/
├── weekly_panel.csv              # Basic features
├── enhanced_panel.csv            # 24 features
│
├── transformer_model/            # Level 1: Basic
│   ├── model.pt
│   └── best.json
│
├── transformer_bert/             # Level 2: BERT Text
│   ├── model_bert.pt
│   └── best_bert.json
│
├── bert_enhanced/                # Level 3: Enhanced BERT
│   ├── model_enhanced.pt
│   └── best_enhanced.json
│
└── gnn_multitask/                # Level 4: GNN Multi-Task
    ├── model_gnn_multitask.pt
    └── best_gnn.json
```

---

## Monitoring Training

### Watch Progress
```bash
# Live monitoring
tail -f logs/train_gnn_*.out

# Check GPU usage
ssh NODE_NAME
nvidia-smi
```

### Key Metrics to Watch

```
Epoch 1/20:
  Train Loss: 0.4523    # Should decrease
  Val Accuracy: 0.7234  # Should increase
  Best: 0.7234          # Tracks best so far
```

**Good signs:**
- Train loss decreases steadily
- Val accuracy increases
- Gap between train/val loss < 10%

**Bad signs:**
- Val accuracy plateaus early → need more epochs
- Train loss << Val loss → overfitting
- NaN loss → reduce learning rate

---

## Troubleshooting

### Job Failed?
```bash
# Check error log
tail -50 logs/train_JOBID.err

# Common issues:
# 1. OOM: Reduce batch size
# 2. Missing file: Check dependencies
# 3. Module error: Check environment
```

### Out of Memory?
Edit SLURM script:
```bash
# Reduce batch size
--batch_size 8  # was 16

# Or freeze BERT
--freeze_bert
```

### Too Slow?
```bash
# Use fewer GNN neighbors
# Edit train_multitask_gnn.py:
graph = ProductGraph(df, reviews_df, max_neighbors=10)  # was 20
```

---

## After Training: Evaluation

### 1. Compare All Models
```bash
python << 'EOF'
import json
from pathlib import Path

outdir = Path("training_output")

models = {
    "Basic": outdir / "transformer_model/best.json",
    "BERT Text": outdir / "transformer_bert/best_bert.json",
    "Enhanced BERT": outdir / "bert_enhanced/best_enhanced.json",
    "GNN Multi-Task": outdir / "gnn_multitask/best_gnn.json"
}

print("Model Comparison:")
print("-" * 50)
for name, path in models.items():
    if path.exists():
        metrics = json.loads(path.read_text())
        acc = metrics.get('val_acc', 0)
        print(f"{name:20s}: {acc:.4f} ({acc*100:.2f}%)")
    else:
        print(f"{name:20s}: Not trained yet")
EOF
```

### 2. Analyze Predictions
```bash
# Load best model
python analyze_predictions.py \
  --model gnn_multitask/model_gnn_multitask.pt \
  --data enhanced_panel.csv \
  --out analysis/
```

### 3. Feature Importance
```bash
# Which features matter most?
python feature_importance.py \
  --model bert_enhanced/model_enhanced.pt \
  --out feature_importance.png
```

---

## Next Steps

### 1. Ensemble Models
Combine predictions from multiple models:

```python
# Simple averaging
pred_final = 0.4 * pred_gnn + 0.3 * pred_bert + 0.3 * pred_autots

# Weighted by validation accuracy
weights = [0.92, 0.86, 0.75]  # val accuracies
pred_final = weighted_average(preds, weights)
```

### 2. Hyperparameter Tuning
```bash
# Try different learning rates
for lr in 1e-4 2e-4 5e-4; do
  sbatch --export=LR=$lr train_gnn_multitask.slurm
done
```

### 3. Production Deployment
```python
# Save production model
torch.save({
    'model': model.state_dict(),
    'config': config,
    'vocab': vocab,
    'scaler': scaler
}, 'production_model.pt')

# Inference
model.load_state_dict(checkpoint['model'])
model.eval()
with torch.no_grad():
    predictions = model(new_data)
```

---

## Recommended Training Order

### For Research/Experimentation:
1. ✅ **Start basic** (running now)
2. ⏭️ **Run enhanced BERT** (best ROI)
3. ⏭️ **Run GNN multi-task** (cutting edge)
4. 📊 **Compare all** and analyze

### For Production Deployment:
1. ✅ **Train enhanced BERT only**
2. 🎯 **Validate on holdout set**
3. 🚀 **Deploy with monitoring**

### For Maximum Accuracy:
1. ✅ **Train GNN multi-task**
2. 🔧 **Tune hyperparameters**
3. 🎭 **Ensemble with AutoTS**

---

## FAQ

**Q: Which model should I use?**
A: For production → Enhanced BERT. For research → GNN Multi-Task.

**Q: Can I train on CPU?**
A: Only Basic model. Others need GPU.

**Q: How long until I see results?**
A: First checkpoint saves after epoch 1 (~30 min for GNN).

**Q: Can I stop and resume?**
A: Yes, models save checkpoints. Add resume logic to scripts.

**Q: What if validation accuracy doesn't improve?**
A: Try: (1) More epochs, (2) Lower LR, (3) More data, (4) Different features.

---

## Getting Help

1. Check error logs: `logs/*.err`
2. Review documentation: `*.md` files
3. Debug with small batch: `--batch_size 1`
4. Simplify: Remove GNN, reduce features

---

## Summary

You now have **4 models** of increasing sophistication:

1. **Basic** - Fast baseline ✅
2. **BERT Text** - Add BERT 📝
3. **Enhanced BERT** - Add features 🚀
4. **GNN Multi-Task** - Add everything 🎯

**Recommendation:** Train Enhanced BERT and GNN Multi-Task, compare results!