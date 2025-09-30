#!/bin/bash
#SBATCH --job-name=03_gnn_multitask
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/03_gnn_multitask_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/03_gnn_multitask_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu

# ============================================================================
# STAGE 3C: GNN + MULTI-TASK LEARNING
# ============================================================================
# Graph Neural Network with product relationships + multi-task learning
# Expected Accuracy: 87-90%
# Training Time: ~18-20 hours (1 GPU)
# ============================================================================

set -e
set -u

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "=========================================="
echo "STAGE 3C: GNN + MULTI-TASK"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1.1

source "${VENV_DIR}/bin/activate"

pip install -q torch-geometric torch-scatter torch-sparse

python "${SCRIPT_DIR}/train_multitask_gnn.py" \
  --data "${OUT_DIR}/panel_enhanced.csv" \
  --out "${OUT_DIR}/models/gnn_multitask" \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-3

echo "✓ GNN Multi-task training complete!"