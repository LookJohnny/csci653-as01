#!/bin/bash
#SBATCH --job-name=batch1_cpu
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/batch1_cpu_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/batch1_cpu_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
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
TRAIN_OUT="${OUT_DIR}/training_output"
BATCH1_OUT="${TRAIN_OUT}/batch1_cpu"

mkdir -p "${BATCH1_OUT}"

echo "=========================================="
echo "🔥 BATCH 1: CPU-HEAVY MODELS (XGBoost + LightGBM)"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "CPUs = 32"
echo "Memory = 128G"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

# Install dependencies
pip install -q xgboost lightgbm scikit-learn pandas pyarrow fastparquet

# Build enhanced features if needed
FEATURE_FILE="${TRAIN_OUT}/enhanced_panel.csv"
if [ ! -f "${FEATURE_FILE}" ]; then
    echo ""
    echo "=== Building Enhanced Features ==="
    echo ""
    python "${SCRIPT_DIR}/build_enhanced_features.py" \
      --input "${OUT_DIR}/combined_reviews.parquet" \
      --out "${FEATURE_FILE}" \
      --top_quantile 0.95 \
      --min_reviews 1
fi

# Add external features (holidays, seasons)
EXTERNAL_FILE="${TRAIN_OUT}/panel_with_external.csv"
if [ ! -f "${EXTERNAL_FILE}" ]; then
    echo ""
    echo "=== Adding External Features (holidays, seasons) ==="
    echo ""
    python "${SCRIPT_DIR}/external_features.py" \
      --input "${FEATURE_FILE}" \
      --output "${EXTERNAL_FILE}" \
      --date_col week_start
fi

echo ""
echo "=== Training XGBoost + LightGBM ==="
echo ""

python "${SCRIPT_DIR}/ensemble_predictor.py" \
  --data "${EXTERNAL_FILE}" \
  --out_dir "${BATCH1_OUT}" \
  --train_tree_models

echo ""
echo "=========================================="
echo "✓ BATCH 1 COMPLETE!"
echo "=========================================="
echo ""
echo "Models trained:"
echo "  ✓ XGBoost: ${BATCH1_OUT}/xgboost_model.pkl"
echo "  ✓ LightGBM: ${BATCH1_OUT}/lightgbm_model.txt"
echo ""
echo "Next: Submit Batch 2 (GPU models)"
echo "  sbatch ${SCRIPT_DIR}/train_batch2_gpu.slurm"
echo ""
echo "=========================================="