#!/bin/bash
#SBATCH --job-name=train_ultimate
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_ultimate_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_ultimate_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
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

mkdir -p "${TRAIN_OUT}"

echo "=========================================="
echo "🚀 ULTIMATE MODEL TRAINING"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi
echo "=========================================="
echo ""
echo "Advanced Techniques Enabled:"
echo "  ✓ Multi-Task Learning"
echo "  ✓ Graph Neural Network"
echo "  ✓ Contrastive Learning"
echo "  ✓ Focal Loss"
echo "  ✓ Mixup Augmentation"
echo "  ✓ CutMix Augmentation"
echo "  ✓ Early Stopping"
echo "  ✓ LR Warmup + Cosine Decay"
echo "  ✓ Mixed Precision Training"
echo "  ✓ Gradient Accumulation"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1

source "${VENV_DIR}/bin/activate"

pip install -q transformers>=4.30.0 tqdm

# Build enhanced features if needed
FEATURE_FILE="${TRAIN_OUT}/enhanced_panel.csv"
if [ ! -f "${FEATURE_FILE}" ]; then
    echo ""
    echo "Building enhanced features..."
    python "${SCRIPT_DIR}/build_enhanced_features.py" \
      --input "${OUT_DIR}/combined_reviews.parquet" \
      --out "${FEATURE_FILE}" \
      --top_quantile 0.95 \
      --min_reviews 1
fi

echo ""
echo "Starting Ultimate Model Training..."
echo ""

python "${SCRIPT_DIR}/train_ultimate.py" \
  --data "${FEATURE_FILE}" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${TRAIN_OUT}/ultimate_model" \
  --bert_model "distilbert-base-uncased" \
  --seq_len 32 \
  --text_max_len 128 \
  --d_model 256 \
  --max_neighbors 20 \
  --batch_size 16 \
  --epochs 30 \
  --lr 0.0002 \
  --weight_decay 0.0001 \
  --accumulation_steps 4 \
  --num_workers 8 \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --contrast_temp 0.07 \
  --w_focal 1.0 \
  --w_contrast 0.15 \
  --w_rating 0.3 \
  --w_engagement 0.2 \
  --use_mixup \
  --use_cutmix \
  --mixup_alpha 0.3 \
  --aug_prob 0.5 \
  --use_amp \
  --patience 15 \
  --min_delta 0.0005 \
  --warmup_ratio 0.1

echo ""
echo "=========================================="
echo "✓ TRAINING COMPLETE!"
echo "=========================================="
echo "Output: ${TRAIN_OUT}/ultimate_model"
echo ""
echo "Files created:"
ls -lh "${TRAIN_OUT}/ultimate_model/"
echo ""
echo "Best model metrics:"
if [ -f "${TRAIN_OUT}/ultimate_model/training_metrics.json" ]; then
    cat "${TRAIN_OUT}/ultimate_model/training_metrics.json"
fi
echo ""
echo "=========================================="