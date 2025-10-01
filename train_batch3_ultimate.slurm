#!/bin/bash
#SBATCH --job-name=batch3_ultimate
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/batch3_ultimate_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/batch3_ultimate_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
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
BATCH3_OUT="${TRAIN_OUT}/batch3_ultimate"

mkdir -p "${BATCH3_OUT}"

echo "=========================================="
echo "🔥 BATCH 3: ULTIMATE MODEL (BERT + GNN + Multi-Task)"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi
echo ""
echo "Advanced Techniques Enabled:"
echo "  ✓ Multi-Task Learning (4 tasks)"
echo "  ✓ Graph Neural Network"
echo "  ✓ Contrastive Learning"
echo "  ✓ Focal Loss"
echo "  ✓ Mixup Augmentation"
echo "  ✓ CutMix Augmentation"
echo "  ✓ Early Stopping"
echo "  ✓ LR Warmup + Cosine Decay"
echo "  ✓ Mixed Precision Training (AMP)"
echo "  ✓ Gradient Accumulation"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1

source "${VENV_DIR}/bin/activate"

# Install dependencies
pip install -q torch torchvision torchaudio transformers>=4.30.0 tqdm scikit-learn

# Use enhanced features from Batch 1
FEATURE_FILE="${TRAIN_OUT}/panel_with_external.csv"

if [ ! -f "${FEATURE_FILE}" ]; then
    echo "ERROR: ${FEATURE_FILE} not found. Run Batch 1 first."
    exit 1
fi

echo ""
echo "=== Training Ultimate Model ==="
echo ""

python "${SCRIPT_DIR}/train_ultimate.py" \
  --data "${FEATURE_FILE}" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${BATCH3_OUT}/ultimate" \
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
echo "✓ BATCH 3 COMPLETE!"
echo "=========================================="
echo ""
echo "Model trained:"
echo "  ✓ Ultimate: ${BATCH3_OUT}/ultimate/best_model.pt"
echo ""
if [ -f "${BATCH3_OUT}/ultimate/training_metrics.json" ]; then
    echo "Best metrics:"
    cat "${BATCH3_OUT}/ultimate/training_metrics.json"
    echo ""
fi
echo ""
echo "Next: Submit Batch 4 (Ensemble)"
echo "  sbatch ${SCRIPT_DIR}/train_batch4_ensemble.slurm"
echo ""
echo "=========================================="