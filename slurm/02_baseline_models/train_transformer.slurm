#!/bin/bash
#SBATCH --job-name=02_transformer
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/02_transformer_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/02_transformer_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu

# ============================================================================
# STAGE 2A: BASELINE TRANSFORMER MODEL
# ============================================================================
# Basic Transformer architecture for time-series prediction
# Expected Accuracy: 75-80%
# Training Time: ~8-10 hours (1 GPU)
# ============================================================================

set -e
set -u

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "=========================================="
echo "STAGE 2A: BASELINE TRANSFORMER"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1.1

source "${VENV_DIR}/bin/activate"

python "${SCRIPT_DIR}/train_transformer.py" \
  --data "${OUT_DIR}/panel_weekly.csv" \
  --out "${OUT_DIR}/models/baseline_transformer" \
  --seq_len 32 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --batch_size 128 \
  --epochs 20 \
  --lr 0.0001

echo ""
echo "✓ Baseline Transformer training complete!"