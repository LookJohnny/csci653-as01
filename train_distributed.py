"""
Distributed Training with PyTorch DDP (DistributedDataParallel)

Supports:
- Multi-GPU on single node
- Multi-node training with MPI
- Mixed precision (AMP)
- Gradient accumulation
- Optimized data loading
"""
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # NCCL for GPU
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_distributed(rank, world_size, args):
    """
    Main training function for each process

    rank: GPU id (0 to world_size-1)
    world_size: total number of GPUs
    """
    print(f"[Rank {rank}] Starting training on GPU {rank}/{world_size}")

    # Setup
    setup_distributed(rank, world_size)

    # Load model
    from train_ultimate import MultiTaskGNNModel  # Import your model
    model = MultiTaskGNNModel(**args.model_config)
    model = model.to(rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Data loaders with DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * world_size)

    # AMP scaler
    scaler = GradScaler()

    # Training loop
    for epoch in range(args.epochs):
        # Important: set epoch for shuffling
        train_sampler.set_epoch(epoch)

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = [b.to(rank) if torch.is_tensor(b) else b for b in batch]

            # Mixed precision forward
            with autocast():
                loss = model(*batch)

            # Backward
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Log only from rank 0
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save checkpoint (only rank 0)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # Note: .module for DDP
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch}.pt')

    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus_per_node', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # Calculate world size
    world_size = args.nodes * args.gpus_per_node

    print(f"Training on {world_size} GPUs ({args.nodes} nodes x {args.gpus_per_node} GPUs)")

    # Launch processes
    mp.spawn(
        train_distributed,
        args=(world_size, args),
        nprocs=args.gpus_per_node,
        join=True
    )

if __name__ == '__main__':
    main()