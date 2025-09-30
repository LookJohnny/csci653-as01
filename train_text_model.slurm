#!/bin/bash
#SBATCH --job-name=train_text_model
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_text_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_text_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=main

set -e
set -u
set -o pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
DATA_DIR="/home1/yliu0158/amazon2023/amazon2023_stage"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"
TRAIN_OUT="${OUT_DIR}/training_output"

mkdir -p "${TRAIN_OUT}"

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1

source "${VENV_DIR}/bin/activate"

echo ""
echo "Training Transformer with Text Embeddings..."
echo ""

# Wait for weekly panel to be ready (if first job is still running)
PANEL_FILE="${TRAIN_OUT}/weekly_panel.csv"
if [ ! -f "${PANEL_FILE}" ]; then
    echo "Waiting for weekly panel dataset..."
    while [ ! -f "${PANEL_FILE}" ]; do
        sleep 60
    done
    echo "Panel dataset ready!"
fi

python "${SCRIPT_DIR}/train_transformer_with_text.py" \
  --data "${PANEL_FILE}" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${TRAIN_OUT}/transformer_with_text" \
  --seq_len 32 \
  --text_len 100 \
  --vocab_size 15000 \
  --min_freq 3 \
  --embed_dim 128 \
  --d_model 128 \
  --batch_size 64 \
  --epochs 25 \
  --lr 0.0005

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${TRAIN_OUT}/transformer_with_text"
ls -lh "${TRAIN_OUT}/transformer_with_text/"