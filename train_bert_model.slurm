#!/bin/bash
#SBATCH --job-name=train_bert
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_bert_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_bert_%A.err
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
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"
TRAIN_OUT="${OUT_DIR}/training_output"

mkdir -p "${TRAIN_OUT}"

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1

source "${VENV_DIR}/bin/activate"

# Install transformers if not already installed
pip install -q transformers>=4.30.0 tqdm

echo ""
echo "Training Transformer with Pre-trained BERT..."
echo ""

# Wait for weekly panel to be ready
PANEL_FILE="${TRAIN_OUT}/weekly_panel.csv"
if [ ! -f "${PANEL_FILE}" ]; then
    echo "Waiting for weekly panel dataset..."
    while [ ! -f "${PANEL_FILE}" ]; do
        sleep 60
    done
    sleep 10  # Extra buffer
    echo "Panel dataset ready!"
fi

python "${SCRIPT_DIR}/train_transformer_bert.py" \
  --data "${PANEL_FILE}" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${TRAIN_OUT}/transformer_bert" \
  --bert_model "distilbert-base-uncased" \
  --seq_len 32 \
  --text_max_len 128 \
  --d_model 256 \
  --batch_size 32 \
  --epochs 15 \
  --lr 0.0002 \
  --num_workers 8 \
  --accumulation_steps 2

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${TRAIN_OUT}/transformer_bert"
ls -lh "${TRAIN_OUT}/transformer_bert/"