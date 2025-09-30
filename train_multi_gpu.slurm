#!/bin/bash
#SBATCH --job-name=train_multi_gpu
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_multi_gpu_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_multi_gpu_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ============================================================================
# OPTIMIZED MULTI-GPU TRAINING
# Uses PyTorch DistributedDataParallel (DDP) for 2x A100 GPUs
# Expected speedup: 1.8-1.9x (near-linear scaling)
# ============================================================================

set -e
set -u
set -o pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# NCCL optimization for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# PyTorch optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"
TRAIN_OUT="${OUT_DIR}/training_output"

mkdir -p "${TRAIN_OUT}"

echo "=========================================="
echo "⚡ MULTI-GPU TRAINING (2x A100)"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURM_GPUS = $SLURM_GPUS"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.6.3

source "${VENV_DIR}/bin/activate"

pip install -q transformers>=4.30.0 tqdm

# Build features if needed
FEATURE_FILE="${TRAIN_OUT}/enhanced_panel.csv"
if [ ! -f "${FEATURE_FILE}" ]; then
    echo "Building enhanced features..."
    python "${SCRIPT_DIR}/build_enhanced_features.py" \
      --input "${OUT_DIR}/combined_reviews.parquet" \
      --out "${FEATURE_FILE}" \
      --top_quantile 0.95 \
      --min_reviews 1
fi

echo ""
echo "Starting distributed training on 2 GPUs..."
echo ""

# Launch with torch.distributed
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    "${SCRIPT_DIR}/train_ultimate.py" \
    --data "${FEATURE_FILE}" \
    --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
    --out "${TRAIN_OUT}/ultimate_multi_gpu" \
    --bert_model "distilbert-base-uncased" \
    --seq_len 32 \
    --text_max_len 128 \
    --d_model 256 \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.0002 \
    --use_mixup \
    --use_cutmix \
    --use_amp \
    --num_workers 16 \
    --accumulation_steps 2 \
    --distributed

echo ""
echo "=========================================="
echo "✓ MULTI-GPU TRAINING COMPLETE!"
echo "=========================================="
echo "Output: ${TRAIN_OUT}/ultimate_multi_gpu"
ls -lh "${TRAIN_OUT}/ultimate_multi_gpu/"