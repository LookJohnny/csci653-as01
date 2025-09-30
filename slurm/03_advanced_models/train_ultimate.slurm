#!/bin/bash
#SBATCH --job-name=03_ultimate
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/03_ultimate_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/03_ultimate_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=1-06:00:00
#SBATCH --partition=gpu

# ============================================================================
# STAGE 3D: ULTIMATE MODEL (BERT + GNN + MULTI-TASK + ALL TECHNIQUES)
# ============================================================================
# Combines: BERT, GNN, Multi-task, Contrastive Learning, Mixup, CutMix
# Uses: Focal Loss, Label Smoothing, Mixed Precision, Gradient Accumulation
# Expected Accuracy: 92-93%
# Training Time: ~24-30 hours (2 GPUs with DDP)
# ============================================================================

set -e
set -u

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "=========================================="
echo "STAGE 3D: ULTIMATE MODEL"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: 2"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1.1

source "${VENV_DIR}/bin/activate"

pip install -q torch-geometric torch-scatter torch-sparse

# Multi-GPU training with DistributedDataParallel
export MASTER_PORT=29500
export MASTER_ADDR=localhost

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=${MASTER_PORT} \
  "${SCRIPT_DIR}/train_ultimate.py" \
  --data "${OUT_DIR}/panel_enhanced.csv" \
  --out "${OUT_DIR}/models/ultimate" \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --distributed

echo "✓ Ultimate Model training complete!"