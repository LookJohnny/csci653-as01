#!/bin/bash
#SBATCH --job-name=train_fast
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_fast_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_fast_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --partition=main

# ============================================================================
# FAST PARALLEL PIPELINE - Uses 64 CPUs with Dask for parallel processing
# Expected speedup: 4-8x faster than 16 CPU version
# ============================================================================

set -e
set -u
set -o pipefail

# Parallel processing settings (use all 32 CPUs)
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

export TMPDIR="/tmp/SLURM_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
mkdir -p "${OUT_DIR}/logs"

echo "=========================================="
echo "FAST PARALLEL PIPELINE (32 CPUs + Dask)"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "TMPDIR = $TMPDIR"
echo "CPUs = 32"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

# Install Dask for parallel processing
echo "Installing Dask for parallel computation..."
pip install -q dask[complete] pyarrow fastparquet

# Check if combined file already exists (from previous job)
COMBINED_FILE="${OUT_DIR}/combined_reviews.parquet"

if [ -f "${COMBINED_FILE}" ]; then
    echo "✓ Combined reviews file already exists: ${COMBINED_FILE}"
    echo "  Skipping Step 1 (combining parquet files)"
else
    echo "Step 1: Converting parquet files to combined parquet..."
    python <<EOF
import pandas as pd
from pathlib import Path
import shutil
import os

data_dir = Path("${OUT_DIR}")
parquet_files = sorted(data_dir.glob("*_reviews.parquet"))

print(f"Found {len(parquet_files)} review files")

dfs = []
for pf in parquet_files:
    print(f"Reading {pf.name}...")
    df = pd.read_parquet(pf)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {combined.shape}")
print(f"Columns: {combined.columns.tolist()}")

# Write to local temp first (avoid NFS issues)
local_parquet = Path("${TMPDIR}/combined_reviews.parquet")
print(f"Writing to local temp: {local_parquet}")
combined.to_parquet(local_parquet, index=False, engine='pyarrow', compression='snappy')
print("Successfully wrote to local storage")

# Copy to final destination
final_parquet = Path("${COMBINED_FILE}")
print(f"Copying to final destination: {final_parquet}")
shutil.copy2(local_parquet, final_parquet)
print(f"Successfully copied to {final_parquet}")
print(f"Total rows: {len(combined):,}")

# Cleanup
local_parquet.unlink()
print("Cleaned up local temp file")
EOF
fi

echo ""
echo "Step 2: Building weekly panel dataset (FAST with Dask, 32 partitions)..."
echo ""

python "${SCRIPT_DIR}/build_weekly_dataset_fast.py" \
  --input "${COMBINED_FILE}" \
  --out "${OUT_DIR}/panel_weekly.csv" \
  --top_quantile 0.95 \
  --min_reviews 1 \
  --npartitions 32

echo ""
echo "Step 3: Training Transformer model..."
echo ""

python "${SCRIPT_DIR}/train_transformer.py" \
  --data "${OUT_DIR}/panel_weekly.csv" \
  --out "${OUT_DIR}/model_transformer" \
  --seq_len 32 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --batch_size 128 \
  --epochs 20 \
  --lr 0.0001

echo ""
echo "Step 4: Training AutoTS model..."
echo ""

python "${SCRIPT_DIR}/train_autots.py" \
  --data "${OUT_DIR}/panel_weekly.csv" \
  --out "${OUT_DIR}/model_autots"

echo ""
echo "=========================================="
echo "✓ FAST PIPELINE COMPLETE!"
echo "=========================================="
echo "Output directory: ${OUT_DIR}"
echo ""
echo "Files created:"
ls -lh "${OUT_DIR}"/model_* "${OUT_DIR}"/*.csv "${OUT_DIR}"/*.parquet 2>/dev/null || true

# Cleanup temp directory
rm -rf "${TMPDIR}"