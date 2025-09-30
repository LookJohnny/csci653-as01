"""
Ultimate Training Script
Combines ALL advanced techniques:
- Multi-task GNN
- Contrastive learning
- Focal loss
- Mixup/CutMix
- Early stopping
- Warmup + cosine scheduling
- Label smoothing
- Gradient accumulation
- Mixed precision
"""
import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

# Import custom modules
from train_multitask_gnn import (
    ProductGraph, MultiTaskDataset, collate_multitask, MultiTaskGNNModel
)
from advanced_trainer import (
    FocalLoss, SupConLoss, mixup_data, cutmix_data,
    EarlyStopping, WarmupScheduler, LabelSmoothingLoss
)

class UltimateTrainer:
    """Ultimate trainer with all techniques"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # Loss functions
        self.focal_loss = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )

        self.contrastive_loss = SupConLoss(
            temperature=config.get('contrast_temp', 0.07)
        )

        self.mse_loss = nn.MSELoss()

        # Loss weights
        self.w_focal = config.get('w_focal', 1.0)
        self.w_contrast = config.get('w_contrast', 0.15)
        self.w_rating = config.get('w_rating', 0.3)
        self.w_engagement = config.get('w_engagement', 0.2)

        # Augmentation config
        self.use_mixup = config.get('use_mixup', True)
        self.use_cutmix = config.get('use_cutmix', True)
        self.mixup_alpha = config.get('mixup_alpha', 0.3)
        self.aug_prob = config.get('aug_prob', 0.5)

        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.0005),
            mode='max',
            verbose=True
        )

        # Metrics tracking
        self.train_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0

    def train_epoch(self, train_loader, optimizer, scheduler, epoch, accumulation_steps=1):
        """Train one epoch with all techniques"""
        self.model.train()

        total_loss = 0
        total_hs_loss = 0
        total_rt_loss = 0
        total_eng_loss = 0
        total_contrast_loss = 0
        n_samples = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Unpack
            xs, input_ids, attn_mask, pids, neighbors, n_neighbors, y_hs, y_rt, y_eng = batch

            xs = xs.to(self.device)
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            pids = pids.to(self.device)
            neighbors = neighbors.to(self.device)
            n_neighbors = n_neighbors.to(self.device)
            y_hs = y_hs.to(self.device)
            y_rt = y_rt.to(self.device)
            y_eng = y_eng.to(self.device)

            # Data augmentation
            use_aug = self.use_mixup or self.use_cutmix
            do_aug = use_aug and np.random.rand() < self.aug_prob

            if do_aug:
                if self.use_mixup and (not self.use_cutmix or np.random.rand() < 0.5):
                    # Mixup
                    xs, y_hs_a, y_hs_b, lam = mixup_data(xs, y_hs, self.mixup_alpha)
                    y_rt_a, y_rt_b = y_rt, y_rt[torch.randperm(y_rt.size(0))]
                    y_eng_a, y_eng_b = y_eng, y_eng[torch.randperm(y_eng.size(0))]
                else:
                    # CutMix
                    xs, y_hs_a, y_hs_b, lam = cutmix_data(xs, y_hs, self.mixup_alpha)
                    y_rt_a, y_rt_b = y_rt, y_rt[torch.randperm(y_rt.size(0))]
                    y_eng_a, y_eng_b = y_eng, y_eng[torch.randperm(y_eng.size(0))]
            else:
                y_hs_a, y_hs_b = y_hs, None
                y_rt_a, y_rt_b = y_rt, None
                y_eng_a, y_eng_b = y_eng, None
                lam = 1.0

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Get predictions and embeddings
                hs_logit, rt_pred, eng_pred = self.model(xs, input_ids, attn_mask, pids, neighbors, n_neighbors)

                # Get embeddings for contrastive learning
                # (Need to modify model to return embeddings)
                # For now, use shared backbone output
                with torch.no_grad():
                    embeddings = self.model.shared_backbone(
                        torch.cat([
                            self.model.text_proj(self.model.bert(input_ids, attn_mask).last_hidden_state[:, 0, :]),
                            self.model.ts_encoder(self.model.ts_proj(xs))[:, -1, :]
                        ], dim=1)
                    )

                # Multi-task losses
                if do_aug and y_hs_b is not None:
                    # Mixup/CutMix loss
                    loss_hs = lam * self.focal_loss(hs_logit, y_hs_a) + (1 - lam) * self.focal_loss(hs_logit, y_hs_b)
                    loss_rt = lam * self.mse_loss(rt_pred, y_rt_a) + (1 - lam) * self.mse_loss(rt_pred, y_rt_b)
                    loss_eng = lam * self.mse_loss(eng_pred, y_eng_a) + (1 - lam) * self.mse_loss(eng_pred, y_eng_b)
                else:
                    loss_hs = self.focal_loss(hs_logit, y_hs)
                    loss_rt = self.mse_loss(rt_pred, y_rt)
                    loss_eng = self.mse_loss(eng_pred, y_eng)

                # Contrastive loss
                loss_contrast = self.contrastive_loss(embeddings, y_hs.squeeze().long())

                # Combined loss
                loss = (
                    self.w_focal * loss_hs +
                    self.w_rating * loss_rt +
                    self.w_engagement * loss_eng +
                    self.w_contrast * loss_contrast
                )

                # Normalize by accumulation
                loss = loss / accumulation_steps

            # Backward with mixed precision
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            # Track metrics
            total_loss += loss.item() * xs.size(0) * accumulation_steps
            total_hs_loss += loss_hs.item() * xs.size(0) * accumulation_steps
            total_rt_loss += loss_rt.item() * xs.size(0) * accumulation_steps
            total_eng_loss += loss_eng.item() * xs.size(0) * accumulation_steps
            total_contrast_loss += loss_contrast.item() * xs.size(0)
            n_samples += xs.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/n_samples:.4f}',
                'hs': f'{total_hs_loss/n_samples:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_metrics = {
            'loss': total_loss / n_samples,
            'hs_loss': total_hs_loss / n_samples,
            'rt_loss': total_rt_loss / n_samples,
            'eng_loss': total_eng_loss / n_samples,
            'contrast_loss': total_contrast_loss / n_samples
        }

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate model"""
        self.model.eval()

        correct = 0
        total = 0
        val_loss = 0

        for batch in tqdm(val_loader, desc="Validation"):
            xs, input_ids, attn_mask, pids, neighbors, n_neighbors, y_hs, y_rt, y_eng = batch

            xs = xs.to(self.device)
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            pids = pids.to(self.device)
            neighbors = neighbors.to(self.device)
            n_neighbors = n_neighbors.to(self.device)
            y_hs = y_hs.to(self.device)

            with autocast(enabled=self.use_amp):
                hs_logit, _, _ = self.model(xs, input_ids, attn_mask, pids, neighbors, n_neighbors)
                loss = self.focal_loss(hs_logit, y_hs)

            pred = (torch.sigmoid(hs_logit) >= 0.5).float()
            correct += (pred == y_hs).sum().item()
            total += y_hs.numel()
            val_loss += loss.item() * xs.size(0)

        acc = correct / total
        avg_loss = val_loss / total

        return {'acc': acc, 'loss': avg_loss}

    def train(self, train_loader, val_loader, optimizer, scheduler, epochs, save_dir):
        """Complete training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("ULTIMATE TRAINING START")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Mixup: {self.use_mixup}, CutMix: {self.use_cutmix}")
        print(f"Contrastive weight: {self.w_contrast}")
        print(f"Early stopping patience: {self.early_stopping.patience}")
        print(f"Mixed precision: {self.use_amp}")
        print("="*60 + "\n")

        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, scheduler, epoch,
                accumulation_steps=self.config.get('accumulation_steps', 2)
            )

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Log
            print(f"\nTrain Metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            print(f"\nVal Metrics:")
            print(f"  Accuracy: {val_metrics['acc']:.4f}")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Best Accuracy: {self.best_val_acc:.4f}")

            # Track
            self.train_losses.append(train_metrics['loss'])
            self.val_accs.append(val_metrics['acc'])

            # Save best model
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_metrics['acc'],
                    'val_loss': val_metrics['loss'],
                    'train_metrics': train_metrics,
                    'config': self.config
                }

                torch.save(checkpoint, save_dir / 'best_model.pt')
                print(f"\n✓ Saved best model! Val Acc: {val_metrics['acc']:.4f}")

            # Early stopping
            if self.early_stopping(val_metrics['acc'], epoch):
                print(f"\n{'='*60}")
                print("EARLY STOPPING TRIGGERED")
                print(f"Best epoch: {self.early_stopping.best_epoch}")
                print(f"Best val acc: {self.early_stopping.best_score:.4f}")
                print(f"{'='*60}")
                break

        # Save final metrics
        final_metrics = {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.early_stopping.best_epoch,
            'train_losses': self.train_losses,
            'val_accs': self.val_accs
        }

        (save_dir / 'training_metrics.json').write_text(json.dumps(final_metrics, indent=2))

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Saved to: {save_dir}")
        print(f"{'='*60}\n")

        return final_metrics

def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data, parse_dates=['week_start'])
    reviews_df = pd.read_parquet(args.reviews_file,
                                 columns=['parent_asin', 'user_id', 'title', 'text', 'helpful_vote'])

    print("Building product graph...")
    graph = ProductGraph(df, reviews_df, max_neighbors=args.max_neighbors)

    print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Split
    last_week = df['week_start'].max()
    cutoff = last_week - pd.Timedelta(weeks=26)
    train_df = df[df['week_start'] <= cutoff].copy()
    val_df = df[df['week_start'] > cutoff].copy()

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    # Datasets
    train_ds = MultiTaskDataset(train_df, reviews_df, tokenizer, graph,
                                seq_len=args.seq_len, text_max_len=args.text_max_len)
    val_ds = MultiTaskDataset(val_df, reviews_df, tokenizer, graph,
                              seq_len=args.seq_len, text_max_len=args.text_max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_multitask,
                             pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_multitask,
                           pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MultiTaskGNNModel(
        bert_model_name=args.bert_model,
        n_products=graph.n_products,
        d_model=args.d_model,
        freeze_bert=args.freeze_bert
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': args.lr / 10},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    # Scheduler with warmup
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=args.lr * 0.01,
        warmup_strategy='linear',
        decay_strategy='cosine'
    )

    # Training config
    config = {
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'contrast_temp': args.contrast_temp,
        'w_focal': args.w_focal,
        'w_contrast': args.w_contrast,
        'w_rating': args.w_rating,
        'w_engagement': args.w_engagement,
        'use_mixup': args.use_mixup,
        'use_cutmix': args.use_cutmix,
        'mixup_alpha': args.mixup_alpha,
        'aug_prob': args.aug_prob,
        'use_amp': args.use_amp,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'accumulation_steps': args.accumulation_steps
    }

    # Trainer
    trainer = UltimateTrainer(model, device, config)

    # Train
    metrics = trainer.train(train_loader, val_loader, optimizer, scheduler,
                           epochs=args.epochs, save_dir=args.out)

    print("Done!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reviews_file", required=True)
    ap.add_argument("--out", required=True)

    # Model args
    ap.add_argument("--bert_model", default="distilbert-base-uncased")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--text_max_len", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--max_neighbors", type=int, default=20)
    ap.add_argument("--freeze_bert", action="store_true")

    # Training args
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)

    # Loss args
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--contrast_temp", type=float, default=0.07)
    ap.add_argument("--w_focal", type=float, default=1.0)
    ap.add_argument("--w_contrast", type=float, default=0.15)
    ap.add_argument("--w_rating", type=float, default=0.3)
    ap.add_argument("--w_engagement", type=float, default=0.2)

    # Augmentation args
    ap.add_argument("--use_mixup", action="store_true", default=True)
    ap.add_argument("--use_cutmix", action="store_true", default=True)
    ap.add_argument("--mixup_alpha", type=float, default=0.3)
    ap.add_argument("--aug_prob", type=float, default=0.5)

    # Training technique args
    ap.add_argument("--use_amp", action="store_true", default=True)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=0.0005)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)

    args = ap.parse_args()
    main(args)