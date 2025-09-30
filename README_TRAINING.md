# Training Job Monitoring Guide

## Current Training Job
**Job ID:** 3012925
**Status:** Running
**Dataset:** 6.7M reviews from 5 categories (All_Beauty, Amazon_Fashion, Appliances, Arts_Crafts_and_Sewing, Automotive)

---

## Quick Check Commands

### 1. Check if job is still running:
```bash
squeue -u $USER
```

### 2. Monitor live progress:
```bash
tail -f /home1/yliu0158/amazon2023/amazon23/logs/train_3012925.out
```

### 3. Check for errors:
```bash
tail -f /home1/yliu0158/amazon2023/amazon23/logs/train_3012925.err
```

### 4. Run comprehensive status check:
```bash
/home1/yliu0158/amazon2023/csci653-as01/check_training_results.sh
```

---

## Training Pipeline Stages

### ✅ Stage 1: Data Preparation (COMPLETED)
- Combined 5 categories into single dataset
- **Output:** `/home1/yliu0158/amazon2023/amazon23/combined_reviews.csv` (6.7M rows)

### ⏳ Stage 2: Weekly Panel Building (IN PROGRESS)
- Aggregates reviews by week per product
- Calculates rolling metrics, growth scores
- Assigns hot-seller labels (top 5%)
- **Output:** `/home1/yliu0158/amazon2023/amazon23/training_output/weekly_panel.csv`

### 📋 Stage 3: Transformer Training (PENDING)
- Trains time series transformer
- 20 epochs, batch size 128
- Predicts hot-seller probability
- **Output:** `/home1/yliu0158/amazon2023/amazon23/training_output/transformer_model/`

### 📋 Stage 4: Forecast Pipeline (PENDING)
- Runs TFT (Temporal Fusion Transformer)
- Runs AutoTS ensemble
- Generates blended predictions
- **Output:** `/home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/`

---

## Expected Output Files

Once training completes, you'll have:

```
/home1/yliu0158/amazon2023/amazon23/
├── combined_reviews.csv              # 6.7M combined reviews
└── training_output/
    ├── weekly_panel.csv              # Time-aggregated data
    ├── transformer_model/
    │   ├── model.pt                  # Trained transformer weights
    │   ├── metrics.json              # Training metrics
    │   └── config.json               # Model configuration
    └── forecast_output/
        ├── pred_tft_val.csv          # TFT predictions
        ├── pred_autots_val.csv       # AutoTS predictions
        ├── pred_blend_val.csv        # Ensemble predictions
        ├── metrics.json              # Forecast metrics (SMAPE, MAE, RMSE)
        ├── data_schema.json          # Dataset metadata
        └── plots/
            ├── forecast_vs_actual.png
            ├── residual_hist_blend.png
            ├── smape_violin.png
            └── tft_importance.png    # Feature importance
```

---

## How to Use the Results

### 1. View Forecast Metrics:
```bash
cat /home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/metrics.json | python -m json.tool
```

### 2. View Predictions:
```bash
head -50 /home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/pred_blend_val.csv
```

### 3. Analyze with Python:
```python
import pandas as pd
import json

# Load predictions
preds = pd.read_parquet('/home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/pred_blend_val.csv')

# Load metrics
with open('/home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/metrics.json') as f:
    metrics = json.load(f)

print("Global SMAPE:", metrics['global']['smape'])
print("\nTop 10 predicted hot sellers:")
print(preds.groupby('series_id')['yhat_blend'].sum().sort_values(ascending=False).head(10))
```

---

## Troubleshooting

### Job failed or stuck?
```bash
# Check job status
sacct -j 3012925 --format=JobID,State,ExitCode,Elapsed

# View full log
cat /home1/yliu0158/amazon2023/amazon23/logs/train_3012925.out

# View full error log
cat /home1/yliu0158/amazon2023/amazon23/logs/train_3012925.err
```

### Cancel job if needed:
```bash
scancel 3012925
```

### Resubmit training:
```bash
sbatch /home1/yliu0158/amazon2023/csci653-as01/train_full_pipeline.slurm
```

---

## Source Data

The training uses data downloaded from:
- **Source:** HuggingFace McAuley-Lab/Amazon-Reviews-2023
- **Location:** `/home1/yliu0158/amazon2023/amazon2023_stage/`
- **Size:** 8.1 GB (23 categories, 41 parquet files)

To train on different categories, edit:
```bash
/home1/yliu0158/amazon2023/csci653-as01/train_full_pipeline.slurm
```
Line 51-52: Change which categories to include in the training loop.

---

## Contact

For issues or questions:
- Check logs first
- Review error messages
- Consult the main README: `/home1/yliu0158/amazon2023/csci653-as01/README.md`