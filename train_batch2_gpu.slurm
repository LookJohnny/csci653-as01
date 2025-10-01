#!/bin/bash
#SBATCH --job-name=batch2_gpu
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/batch2_gpu_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/batch2_gpu_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
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
BATCH2_OUT="${TRAIN_OUT}/batch2_gpu"

mkdir -p "${BATCH2_OUT}"

echo "=========================================="
echo "🚀 BATCH 2: GPU MODELS (Transformer, BERT, GNN)"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi
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
echo "=== MODEL 1: Transformer (Time Series) ==="
echo ""

python "${SCRIPT_DIR}/train_transformer.py" \
  --data "${FEATURE_FILE}" \
  --out "${BATCH2_OUT}/transformer" \
  --seq_len 32 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --batch_size 128 \
  --epochs 20 \
  --lr 0.0001

echo ""
echo "=== MODEL 2: Transformer + BERT (with Text) ==="
echo ""

python "${SCRIPT_DIR}/train_transformer_bert.py" \
  --data "${FEATURE_FILE}" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${BATCH2_OUT}/transformer_bert" \
  --bert_model "distilbert-base-uncased" \
  --seq_len 32 \
  --text_max_len 128 \
  --d_model 128 \
  --batch_size 16 \
  --epochs 20 \
  --lr 0.0001

echo ""
echo "=== MODEL 3: GNN + Multi-Task ==="
echo ""

python "${SCRIPT_DIR}/train_multitask_gnn.py" \
  --data "${FEATURE_FILE}" \
  --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
  --out "${BATCH2_OUT}/gnn_multitask" \
  --bert_model "distilbert-base-uncased" \
  --seq_len 32 \
  --text_max_len 128 \
  --d_model 128 \
  --max_neighbors 20 \
  --batch_size 16 \
  --epochs 20 \
  --lr 0.0001

echo ""
echo "=========================================="
echo "✓ BATCH 2 COMPLETE!"
echo "=========================================="
echo ""
echo "Models trained:"
echo "  ✓ Transformer: ${BATCH2_OUT}/transformer/model.pt"
echo "  ✓ Transformer+BERT: ${BATCH2_OUT}/transformer_bert/best_model.pt"
echo "  ✓ GNN+Multi-task: ${BATCH2_OUT}/gnn_multitask/best_model.pt"
echo ""
echo "Next: Submit Batch 3 (Ultimate model)"
echo "  sbatch ${SCRIPT_DIR}/train_batch3_ultimate.slurm"
echo ""
echo "=========================================="