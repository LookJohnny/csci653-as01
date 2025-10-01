#!/bin/bash
#SBATCH --job-name=batch4_ensemble
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/batch4_ensemble_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/batch4_ensemble_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

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
BATCH4_OUT="${TRAIN_OUT}/batch4_ensemble"

mkdir -p "${BATCH4_OUT}"

echo "=========================================="
echo "🎯 BATCH 4: ENSEMBLE COMBINATION"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi
echo ""
echo "Ensemble Methods:"
echo "  ✓ Weighted Average"
echo "  ✓ Stacking (Meta-learner)"
echo "  ✓ Rank Averaging"
echo ""
echo "Models to combine:"
echo "  1. XGBoost (Batch 1)"
echo "  2. LightGBM (Batch 1)"
echo "  3. Transformer (Batch 2)"
echo "  4. Transformer+BERT (Batch 2)"
echo "  5. GNN+Multi-task (Batch 2)"
echo "  6. Ultimate (Batch 3)"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1

source "${VENV_DIR}/bin/activate"

# Install dependencies
pip install -q torch scikit-learn xgboost lightgbm pandas numpy

# Verify all models exist
BATCH1_DIR="${TRAIN_OUT}/batch1_cpu"
BATCH2_DIR="${TRAIN_OUT}/batch2_gpu"
BATCH3_DIR="${TRAIN_OUT}/batch3_ultimate"

echo ""
echo "Checking for trained models..."
echo ""

MISSING=0
for model_path in \
  "${BATCH1_DIR}/xgboost_model.pkl" \
  "${BATCH1_DIR}/lightgbm_model.txt" \
  "${BATCH2_DIR}/transformer/model.pt" \
  "${BATCH2_DIR}/transformer_bert/best_model.pt" \
  "${BATCH2_DIR}/gnn_multitask/best_model.pt" \
  "${BATCH3_DIR}/ultimate/best_model.pt"
do
  if [ ! -f "$model_path" ]; then
    echo "  ✗ MISSING: $model_path"
    MISSING=1
  else
    echo "  ✓ Found: $model_path"
  fi
done

if [ $MISSING -eq 1 ]; then
  echo ""
  echo "ERROR: Some models are missing. Please run all previous batches first."
  echo ""
  echo "Run in order:"
  echo "  1. sbatch ${SCRIPT_DIR}/train_batch1_cpu.slurm"
  echo "  2. sbatch ${SCRIPT_DIR}/train_batch2_gpu.slurm"
  echo "  3. sbatch ${SCRIPT_DIR}/train_batch3_ultimate.slurm"
  echo "  4. sbatch ${SCRIPT_DIR}/train_batch4_ensemble.slurm"
  exit 1
fi

echo ""
echo "=== Building Ensemble ==="
echo ""

python "${SCRIPT_DIR}/build_ensemble.py" \
  --batch1_dir "${BATCH1_DIR}" \
  --batch2_dir "${BATCH2_DIR}" \
  --batch3_dir "${BATCH3_DIR}" \
  --data "${TRAIN_OUT}/panel_with_external.csv" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out_dir "${BATCH4_OUT}" \
  --methods weighted stacking rank

echo ""
echo "=========================================="
echo "✓✓✓ ALL BATCHES COMPLETE! ✓✓✓"
echo "=========================================="
echo ""
echo "Total Models Trained: 6"
echo "  ✓ XGBoost"
echo "  ✓ LightGBM"
echo "  ✓ Transformer"
echo "  ✓ Transformer + BERT"
echo "  ✓ GNN + Multi-task"
echo "  ✓ Ultimate (BERT + GNN + Multi-task + 9 advanced techniques)"
echo ""
echo "Ensemble Methods: 3"
echo "  ✓ Weighted Average"
echo "  ✓ Stacking Meta-learner"
echo "  ✓ Rank Averaging"
echo ""
echo "Final Output:"
echo "  ${BATCH4_OUT}/ensemble_predictions.csv"
echo "  ${BATCH4_OUT}/ensemble_metrics.json"
echo ""
echo "=========================================="