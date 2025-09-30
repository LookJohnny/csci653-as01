#!/bin/bash
#SBATCH --job-name=train_ensemble
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_ensemble_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_ensemble_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --partition=main

set -e
set -u
set -o pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"
TRAIN_OUT="${OUT_DIR}/training_output"
ENSEMBLE_OUT="${TRAIN_OUT}/ensemble"

mkdir -p "${ENSEMBLE_OUT}"

echo "=========================================="
echo "🎯 COMPLETE ENSEMBLE TRAINING PIPELINE"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi
echo "=========================================="
echo ""
echo "Pipeline Steps:"
echo "  1. Build enhanced features (24 features)"
echo "  2. Add external data features (holidays, seasons)"
echo "  3. Train XGBoost model"
echo "  4. Train LightGBM model"
echo "  5. Train Ultimate deep model"
echo "  6. Build ensemble (weighted average + stacking)"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1

source "${VENV_DIR}/bin/activate"

# Install additional dependencies
pip install -q transformers>=4.30.0 tqdm xgboost lightgbm scikit-learn matplotlib

echo ""
echo "=== STEP 1: Enhanced Features ==="
echo ""

FEATURE_FILE="${TRAIN_OUT}/enhanced_panel.csv"
if [ ! -f "${FEATURE_FILE}" ]; then
    python "${SCRIPT_DIR}/build_enhanced_features.py" \
      --input "${OUT_DIR}/combined_reviews.parquet" \
      --out "${FEATURE_FILE}" \
      --top_quantile 0.95 \
      --min_reviews 1
fi

echo ""
echo "=== STEP 2: External Features ==="
echo ""

EXTERNAL_FILE="${TRAIN_OUT}/panel_with_external.csv"
python "${SCRIPT_DIR}/external_features.py" \
  --input "${FEATURE_FILE}" \
  --output "${EXTERNAL_FILE}" \
  --date_col week_start

echo ""
echo "=== STEP 3: Train Tree Models ==="
echo ""

python "${SCRIPT_DIR}/ensemble_predictor.py" \
  --data "${EXTERNAL_FILE}" \
  --out_dir "${ENSEMBLE_OUT}" \
  --train_tree_models

echo ""
echo "=== STEP 4: Train Ultimate Deep Model ==="
echo ""

python "${SCRIPT_DIR}/train_ultimate.py" \
  --data "${EXTERNAL_FILE}" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${ENSEMBLE_OUT}/ultimate" \
  --bert_model "distilbert-base-uncased" \
  --seq_len 32 \
  --text_max_len 128 \
  --d_model 256 \
  --batch_size 16 \
  --epochs 25 \
  --lr 0.0002 \
  --use_mixup \
  --use_cutmix \
  --use_amp \
  --patience 12

echo ""
echo "=========================================="
echo "✓ ENSEMBLE TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Models trained:"
echo "  ✓ XGBoost: ${ENSEMBLE_OUT}/xgboost_model.pkl"
echo "  ✓ LightGBM: ${ENSEMBLE_OUT}/lightgbm_model.txt"
echo "  ✓ Ultimate: ${ENSEMBLE_OUT}/ultimate/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate each model individually"
echo "  2. Build ensemble predictions"
echo "  3. Compare ensemble vs individual models"
echo ""
echo "=========================================="