#!/bin/bash
#SBATCH --job-name=train_darkhorse
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=main

set -e
set -u
set -o pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
DATA_DIR="/home1/yliu0158/amazon2023/amazon2023_stage"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"
TRAIN_OUT="${OUT_DIR}/training_output"

mkdir -p "${TRAIN_OUT}"

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

echo "Step 1: Converting parquet files to combined parquet..."
python << 'EOF'
import pandas as pd
from pathlib import Path
import os
import shutil

data_dir = Path("/home1/yliu0158/amazon2023/amazon2023_stage")
out_dir = Path("/home1/yliu0158/amazon2023/amazon23")
local_tmp = Path(f"/tmp/SLURM_{os.getenv('SLURM_JOB_ID', 'local')}")
local_tmp.mkdir(parents=True, exist_ok=True)

# Find all review parquet files (not chunks)
parquet_files = sorted([f for f in data_dir.glob("*_reviews.parquet") if "_chunk" not in f.name])
print(f"Found {len(parquet_files)} review files")

# Read and combine all parquet files
dfs = []
for pf in parquet_files:
    print(f"Reading {pf.name}...")
    df = pd.read_parquet(pf)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {combined.shape}")
print(f"Columns: {combined.columns.tolist()}")

# Save to local temp first (parquet is more robust than CSV)
local_parquet = local_tmp / "combined_reviews.parquet"
print(f"Writing to local temp: {local_parquet}")
combined.to_parquet(local_parquet, index=False, engine='pyarrow', compression='snappy')
print(f"Successfully wrote to local storage")

# Copy to final destination
final_parquet = out_dir / "combined_reviews.parquet"
print(f"Copying to final destination: {final_parquet}")
shutil.copy2(local_parquet, final_parquet)
print(f"Successfully copied to {final_parquet}")
print(f"Total rows: {len(combined):,}")

# Clean up local temp
local_parquet.unlink()
print("Cleaned up local temp file")
EOF

echo ""
echo "Step 2: Building weekly panel dataset..."
python "${SCRIPT_DIR}/build_weekly_dataset.py" \
  --input "${OUT_DIR}/combined_reviews.parquet" \
  --out "${TRAIN_OUT}/weekly_panel.csv" \
  --top_quantile 0.95 \
  --min_reviews 1

echo ""
echo "Step 3: Training Transformer model..."
python "${SCRIPT_DIR}/train_transformer.py" \
  --data "${TRAIN_OUT}/weekly_panel.csv" \
  --out "${TRAIN_OUT}/transformer_model" \
  --seq_len 32 \
  --batch_size 128 \
  --epochs 20 \
  --lr 0.001

echo ""
echo "Step 4: Running Forecast Pipeline (TFT + AutoTS)..."
python "${SCRIPT_DIR}/forecast_pipeline.py" \
  --dataset "${TRAIN_OUT}/weekly_panel.csv" \
  --series-col parent_asin \
  --time-col week_start \
  --horizon 12 \
  --val-size 12 \
  --outdir "${TRAIN_OUT}/forecast_output"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${TRAIN_OUT}"
echo "Trained models:"
ls -lh "${TRAIN_OUT}/"
echo ""
echo "Forecast results:"
ls -lh "${TRAIN_OUT}/forecast_output/"