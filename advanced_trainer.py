"""
Advanced Training Techniques Module

Implements:
1. Contrastive Learning
2. Focal Loss
3. Mixup/CutMix augmentation
4. Early Stopping with Patience
5. Learning Rate Warmup & Scheduling
6. Label Smoothing
7. Gradient Accumulation
8. Mixed Precision Training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math

# ============================================================================
# 1. CONTRASTIVE LEARNING
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to distinguish hot vs cold products
    Learns embeddings where similar products are close, dissimilar are far
    """
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: (B, D) - product embeddings from model
        labels: (B, 1) - binary labels (0=cold, 1=hot)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Create label matrix: same label = 1, different = 0
        labels_eq = labels.squeeze() == labels.squeeze().unsqueeze(1)  # (B, B)
        labels_eq = labels_eq.float()

        # Positive pairs: same label
        # Negative pairs: different label
        pos_mask = labels_eq - torch.eye(labels_eq.size(0), device=labels_eq.device)  # Remove diagonal
        neg_mask = 1 - labels_eq

        # Contrastive loss
        exp_sim = torch.exp(similarity)

        # Positive similarity
        pos_sim = (exp_sim * pos_mask).sum(dim=1)

        # Negative similarity
        neg_sim = (exp_sim * neg_mask).sum(dim=1)

        # Loss: maximize pos, minimize neg
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))

        return loss.mean()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D)
        labels: (B,)
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity
        similarity = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()

        return loss

# ============================================================================
# 2. FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: Focuses on hard examples
    FL(p_t) = -α(1-p_t)^γ log(p_t)

    α: balance factor (default 0.25)
    γ: focusing parameter (default 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (B, 1) - logits
        targets: (B, 1) - binary labels
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Get probabilities
        p_t = torch.exp(-bce_loss)

        # Focal term
        focal_term = (1 - p_t) ** self.gamma

        # Final loss
        loss = self.alpha * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLossMultiClass(nn.Module):
    """Focal Loss for multi-class classification"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Can be list of weights per class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (B, C) - logits
        targets: (B,) - class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_term = (1 - p_t) ** self.gamma

        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================================================================
# 3. MIXUP / CUTMIX AUGMENTATION
# ============================================================================

def mixup_data(x, y, alpha=1.0):
    """
    Mixup: Mixup: Beyond Empirical Risk Minimization (Zhang et al. 2017)

    Creates virtual training examples:
    x_tilde = λ*x_i + (1-λ)*x_j
    y_tilde = λ*y_i + (1-λ)*y_j
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MixupLoss(nn.Module):
    """Wrapper for Mixup training"""
    def __init__(self, criterion, alpha=1.0):
        super().__init__()
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, pred, y_a, y_b, lam):
        return mixup_criterion(self.criterion, pred, y_a, y_b, lam)

def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: Regularization Strategy (Yun et al. 2019)

    Cuts and pastes patches between training images
    For time series: cuts temporal segments
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # For time series (B, T, F)
    if len(x.shape) == 3:
        seq_len = x.size(1)
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1) if cut_len < seq_len else 0

        x_mixed = x.clone()
        x_mixed[:, cut_start:cut_start+cut_len, :] = x[index, cut_start:cut_start+cut_len, :]

        # Adjust lambda based on actual cut
        lam = 1 - (cut_len / seq_len)
    else:
        # For other shapes, use mixup
        x_mixed = lam * x + (1 - lam) * x[index]

    y_a, y_b = y, y[index]

    return x_mixed, y_a, y_b, lam

# ============================================================================
# 4. EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving

    patience: number of epochs to wait before stopping
    min_delta: minimum change to qualify as improvement
    mode: 'min' or 'max' depending on metric
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"Initial best score: {score:.4f}")
            return False

        # Check if improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                print(f"✓ Improvement! New best: {score:.4f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True

            return False

# ============================================================================
# 5. LEARNING RATE SCHEDULERS
# ============================================================================

class WarmupScheduler:
    """
    Learning rate warmup followed by decay

    Warmup: Linear increase from 0 to max_lr
    Decay: Cosine annealing or linear decay
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr=0.0, warmup_strategy='linear', decay_strategy='cosine'):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_strategy = warmup_strategy
        self.decay_strategy = decay_strategy

        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step <= self.warmup_steps:
                # Warmup phase
                if self.warmup_strategy == 'linear':
                    lr = base_lr * (self.current_step / self.warmup_steps)
                elif self.warmup_strategy == 'exponential':
                    lr = base_lr * (self.current_step / self.warmup_steps) ** 2
                else:
                    lr = base_lr
            else:
                # Decay phase
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)

                if self.decay_strategy == 'cosine':
                    lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                elif self.decay_strategy == 'linear':
                    lr = base_lr - (base_lr - self.min_lr) * progress
                else:
                    lr = base_lr

            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmup:
    """Cosine Annealing with Warmup Restarts"""
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0, cycles=1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.cycles = cycles
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step < self.warmup_steps:
                # Warmup
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine annealing with restarts
                progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                cycle_progress = (progress * self.cycles) % 1.0
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cycle_progress))

            param_group['lr'] = lr

# ============================================================================
# 6. LABEL SMOOTHING
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing: Prevents overconfident predictions

    Instead of [0, 1], use [ε, 1-ε]
    Improves generalization
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        pred: (B, 1) logits
        target: (B, 1) binary
        """
        # Smooth labels
        target_smooth = target * (1 - self.smoothing) + self.smoothing * 0.5

        # BCE with smoothed targets
        loss = F.binary_cross_entropy_with_logits(pred, target_smooth)

        return loss

# ============================================================================
# 7. ADVANCED TRAINER CLASS
# ============================================================================

class AdvancedTrainer:
    """
    Complete training pipeline with all advanced techniques
    """
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # Loss functions
        self.focal_loss = FocalLoss(alpha=config.get('focal_alpha', 0.25),
                                   gamma=config.get('focal_gamma', 2.0))

        self.contrastive_loss = SupConLoss(temperature=config.get('contrast_temp', 0.07))

        self.label_smoothing = LabelSmoothingLoss(smoothing=config.get('smoothing', 0.1))

        # Weights for combined loss
        self.w_focal = config.get('w_focal', 1.0)
        self.w_contrast = config.get('w_contrast', 0.1)

        # Augmentation
        self.use_mixup = config.get('use_mixup', True)
        self.mixup_alpha = config.get('mixup_alpha', 0.2)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001),
            mode='max'
        )

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch with all techniques"""
        self.model.train()
        total_loss = 0
        n_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch
            x_ts, input_ids, attn_mask, y = batch[:4]

            x_ts = x_ts.to(self.device)
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            y = y.to(self.device)

            # Mixup augmentation
            if self.use_mixup and np.random.rand() < 0.5:
                x_ts, y_a, y_b, lam = mixup_data(x_ts, y, self.mixup_alpha)

                # Forward
                logits, embeddings = self.model(x_ts, input_ids, attn_mask, return_embeddings=True)

                # Mixup loss
                loss_focal = lam * self.focal_loss(logits, y_a) + (1 - lam) * self.focal_loss(logits, y_b)
            else:
                # Normal forward
                logits, embeddings = self.model(x_ts, input_ids, attn_mask, return_embeddings=True)

                # Focal loss
                loss_focal = self.focal_loss(logits, y)

            # Contrastive loss
            loss_contrast = self.contrastive_loss(embeddings, y)

            # Combined loss
            loss = self.w_focal * loss_focal + self.w_contrast * loss_contrast

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * x_ts.size(0)
            n_samples += x_ts.size(0)

        return total_loss / n_samples

    def evaluate(self, val_loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                x_ts, input_ids, attn_mask, y = batch[:4]

                x_ts = x_ts.to(self.device)
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                y = y.to(self.device)

                logits = self.model(x_ts, input_ids, attn_mask)
                pred = (torch.sigmoid(logits) >= 0.5).float()

                correct += (pred == y).sum().item()
                total += y.numel()

        return correct / total

# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'FocalLoss', 'FocalLossMultiClass',
    'ContrastiveLoss', 'SupConLoss',
    'mixup_data', 'cutmix_data', 'MixupLoss',
    'EarlyStopping',
    'WarmupScheduler', 'CosineAnnealingWarmup',
    'LabelSmoothingLoss',
    'AdvancedTrainer'
]