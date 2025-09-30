#!/bin/bash
#SBATCH --job-name=train_multi_node
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/train_multi_node_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/train_multi_node_%A.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# ============================================================================
# MULTI-NODE DISTRIBUTED TRAINING
# Uses 2 nodes × 2 GPUs = 4 GPUs total (A100)
# Expected speedup: 3.5-3.8x vs single GPU
# Uses NCCL for GPU communication
# ============================================================================

set -e
set -u
set -o pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# NCCL optimization for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude virtual interfaces
export NCCL_IB_HCA=mlx5  # InfiniBand adapter

# PyTorch DDP settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Debugging

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"
TRAIN_OUT="${OUT_DIR}/training_output"

mkdir -p "${TRAIN_OUT}"

echo "=========================================="
echo "🚀 MULTI-NODE TRAINING (2 Nodes × 2 GPUs)"
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURM_NNODES = $SLURM_NNODES"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "SLURM_GPUS_PER_NODE = 2"
echo ""
echo "Node information:"
scontrol show hostnames $SLURM_JOB_NODELIST
echo ""
echo "GPU information:"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16 cuda/12.6.3 openmpi/5.0.5

source "${VENV_DIR}/bin/activate"

pip install -q transformers>=4.30.0 tqdm

# Build features (only on first node)
FEATURE_FILE="${TRAIN_OUT}/enhanced_panel.csv"
if [ "$SLURM_NODEID" == "0" ] && [ ! -f "${FEATURE_FILE}" ]; then
    echo "Building enhanced features..."
    python "${SCRIPT_DIR}/build_enhanced_features.py" \
      --input "${OUT_DIR}/combined_reviews.parquet" \
      --out "${FEATURE_FILE}" \
      --top_quantile 0.95 \
      --min_reviews 1
fi

# Wait for all nodes
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 sleep 5

echo ""
echo "Starting distributed training on 4 GPUs (2 nodes)..."
echo ""

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Launch with srun (MPI-aware)
srun python "${SCRIPT_DIR}/train_ultimate.py" \
    --data "${FEATURE_FILE}" \
    --reviews_file "${OUT_DIR}/combined_reviews.parquet" \
    --out "${TRAIN_OUT}/ultimate_multi_node" \
    --bert_model "distilbert-base-uncased" \
    --seq_len 32 \
    --text_max_len 128 \
    --d_model 256 \
    --batch_size 24 \
    --epochs 15 \
    --lr 0.0003 \
    --use_mixup \
    --use_cutmix \
    --use_amp \
    --num_workers 16 \
    --accumulation_steps 1 \
    --distributed \
    --world_size 4 \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT"

echo ""
echo "=========================================="
echo "✓ MULTI-NODE TRAINING COMPLETE!"
echo "=========================================="
echo "Total GPUs used: 4 (2 nodes × 2 GPUs)"
echo "Expected training time: ~3x faster than single GPU"
echo ""
echo "Output: ${TRAIN_OUT}/ultimate_multi_node"
ls -lh "${TRAIN_OUT}/ultimate_multi_node/"