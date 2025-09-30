#!/bin/bash
#SBATCH --job-name=02_autots
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/02_autots_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/02_autots_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --partition=main

# ============================================================================
# STAGE 2B: AUTOTS BASELINE MODEL
# ============================================================================
# Automated time-series forecasting with AutoTS
# Expected Accuracy: 70-75%
# Training Time: ~4-6 hours (CPU)
# ============================================================================

set -e
set -u

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "=========================================="
echo "STAGE 2B: AUTOTS BASELINE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs: 16"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

python "${SCRIPT_DIR}/train_autots.py" \
  --data "${OUT_DIR}/panel_weekly.csv" \
  --out "${OUT_DIR}/models/baseline_autots"

echo ""
echo "✓ AutoTS training complete!"