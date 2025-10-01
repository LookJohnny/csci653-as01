#!/bin/bash
#SBATCH --job-name=build_weekly
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/build_weekly_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/build_weekly_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --partition=epyc-64

set -e
set -u
set -o pipefail

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

mkdir -p "${OUT_DIR}/logs"

echo "=========================================="
echo "📊 BUILDING WEEKLY PANEL DATASET"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "CPUs = 32"
echo "Memory = 240G"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

# Install Dask for parallel processing
pip install -q dask[complete] pyarrow fastparquet

echo ""
echo "Building weekly panel dataset (32 parallel partitions)..."
echo ""

python "${SCRIPT_DIR}/build_weekly_dataset_fast.py" \
  --input "${OUT_DIR}/combined_reviews.parquet" \
  --out "${OUT_DIR}/panel_weekly.parquet" \
  --top_quantile 0.95 \
  --min_reviews 1 \
  --npartitions 32

echo ""
echo "=========================================="
echo "✓ WEEKLY PANEL COMPLETE!"
echo "=========================================="
echo ""

if [ -f "${OUT_DIR}/panel_weekly.parquet" ]; then
  echo "Output file: ${OUT_DIR}/panel_weekly.parquet"
  ls -lh "${OUT_DIR}/panel_weekly.parquet"
fi

echo ""
echo "Next: Run model training batches"
echo "  sbatch ${SCRIPT_DIR}/train_batch1_cpu.slurm"
echo ""
echo "=========================================="