#!/bin/bash
#SBATCH --job-name=03_bert_enhanced
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/03_bert_enhanced_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/03_bert_enhanced_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --partition=gpu

# ============================================================================
# STAGE 3A: BERT + ENHANCED FEATURES
# ============================================================================
# DistilBERT for text + enhanced tabular features + focal loss
# Expected Accuracy: 85-88%
# Training Time: ~12-15 hours (1 GPU)
# ============================================================================

set -e
set -u

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "=========================================="
echo "STAGE 3A: BERT + ENHANCED FEATURES"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1.1

source "${VENV_DIR}/bin/activate"

python "${SCRIPT_DIR}/train_bert_enhanced.py" \
  --data "${OUT_DIR}/panel_enhanced.csv" \
  --out "${OUT_DIR}/models/bert_enhanced" \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-5

echo "✓ BERT Enhanced training complete!"