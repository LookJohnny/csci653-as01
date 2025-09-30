#!/bin/bash
#SBATCH --job-name=01_prepare_data
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/01_prepare_data_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/01_prepare_data_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --partition=main

# ============================================================================
# STAGE 1: DATA PREPARATION
# ============================================================================
# Combines all review parquet files and builds the weekly panel dataset
# with 54+ engineered features using Dask parallel processing (32 CPUs)
# Expected time: 30-60 minutes
# ============================================================================

set -e
set -u
set -o pipefail

# Parallel processing settings
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
echo "STAGE 1: DATA PREPARATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs: 32 (with Dask parallel processing)"
echo "Memory: 128GB"
echo "=========================================="
echo ""

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

# Install dependencies
pip install -q dask[complete] pyarrow fastparquet

COMBINED_FILE="${OUT_DIR}/combined_reviews.parquet"

# Step 1: Combine parquet files (if not already done)
if [ -f "${COMBINED_FILE}" ]; then
    echo "✓ Combined reviews file already exists: ${COMBINED_FILE}"
    echo "  Skipping combination step"
else
    echo "Step 1: Combining parquet files..."
    python <<EOF
import pandas as pd
from pathlib import Path
import shutil

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

# Write to local temp first
local_parquet = Path("${TMPDIR}/combined_reviews.parquet")
print(f"Writing to: {local_parquet}")
combined.to_parquet(local_parquet, index=False, engine='pyarrow', compression='snappy')

# Copy to final destination
final_parquet = Path("${COMBINED_FILE}")
shutil.copy2(local_parquet, final_parquet)
print(f"✓ Saved to {final_parquet}")
print(f"  Total rows: {len(combined):,}")
EOF
fi

echo ""
echo "Step 2: Building weekly panel with enhanced features (Dask, 32 partitions)..."
python "${SCRIPT_DIR}/build_weekly_dataset_fast.py" \
  --input "${COMBINED_FILE}" \
  --out "${OUT_DIR}/panel_weekly.csv" \
  --top_quantile 0.95 \
  --min_reviews 1 \
  --npartitions 32

echo ""
echo "Step 3: Building enhanced features dataset..."
if [ -f "${SCRIPT_DIR}/build_enhanced_features.py" ]; then
    python "${SCRIPT_DIR}/build_enhanced_features.py" \
      --input "${OUT_DIR}/panel_weekly.csv" \
      --out "${OUT_DIR}/panel_enhanced.csv"
else
    echo "  Enhanced features script not found, skipping..."
fi

echo ""
echo "=========================================="
echo "✓ DATA PREPARATION COMPLETE"
echo "=========================================="
echo "Output files:"
ls -lh "${OUT_DIR}"/panel_*.csv "${OUT_DIR}"/*.parquet 2>/dev/null || true
echo ""
echo "Files ready for model training!"

# Cleanup
rm -rf "${TMPDIR}"