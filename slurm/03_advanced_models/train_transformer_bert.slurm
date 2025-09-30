#!/bin/bash
#SBATCH --job-name=03_trans_bert
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/03_transformer_bert_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/03_transformer_bert_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --partition=gpu

# ============================================================================
# STAGE 3B: TRANSFORMER + BERT COMBINED
# ============================================================================
# Combines Transformer time-series with BERT text understanding
# Expected Accuracy: 86-89%
# Training Time: ~14-16 hours (1 GPU)
# ============================================================================

set -e
set -u

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "=========================================="
echo "STAGE 3B: TRANSFORMER + BERT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1.1

source "${VENV_DIR}/bin/activate"

python "${SCRIPT_DIR}/train_transformer_bert.py" \
  --data "${OUT_DIR}/panel_enhanced.csv" \
  --out "${OUT_DIR}/models/transformer_bert" \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-4

echo "✓ Transformer+BERT training complete!"