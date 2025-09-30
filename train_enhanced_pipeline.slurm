#!/bin/bash
#SBATCH --job-name=train_enhanced
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_enhanced_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_enhanced_%A.err
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
echo "Enhanced Model Training Pipeline"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.1

source "${VENV_DIR}/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -q transformers>=4.30.0 tqdm

echo ""
echo "Step 1: Building Enhanced Feature Dataset..."
echo ""

python "${SCRIPT_DIR}/build_enhanced_features.py" \
  --input "${OUT_DIR}/combined_reviews.parquet" \
  --out "${TRAIN_OUT}/enhanced_panel.csv" \
  --top_quantile 0.95 \
  --min_reviews 1

echo ""
echo "Enhanced features created!"
echo "Dataset info:"
wc -l "${TRAIN_OUT}/enhanced_panel.csv"
head -1 "${TRAIN_OUT}/enhanced_panel.csv" | tr ',' '\n' | nl

echo ""
echo "Step 2: Training Enhanced BERT Model..."
echo ""

python "${SCRIPT_DIR}/train_bert_enhanced.py" \
  --data "${TRAIN_OUT}/enhanced_panel.csv" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${TRAIN_OUT}/bert_enhanced" \
  --bert_model "distilbert-base-uncased" \
  --seq_len 32 \
  --text_max_len 128 \
  --d_model 256 \
  --batch_size 24 \
  --epochs 20 \
  --lr 0.0002 \
  --num_workers 8 \
  --accumulation_steps 4

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${TRAIN_OUT}/bert_enhanced"
echo ""
echo "Model files:"
ls -lh "${TRAIN_OUT}/bert_enhanced/"
echo ""
echo "Best metrics:"
cat "${TRAIN_OUT}/bert_enhanced/best_enhanced.json"