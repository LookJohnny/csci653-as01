"""
Enhanced BERT model with rich feature set:
- Pre-trained BERT for text
- Sentiment, engagement, image features
- Category embeddings
- Advanced time series features
"""
import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

# ---- Enhanced Dataset ----
class EnhancedPanelDataset(Dataset):
    def __init__(self, df, reviews_df, tokenizer, seq_len=32, min_weeks=12, text_max_len=128):
        self.seq_len = seq_len
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer

        # Enhanced feature set
        self.feature_cols = [
            # Basic metrics
            'reviews', 'rating_mean', 'rating_std',

            # Engagement features
            'helpful_sum', 'helpful_mean', 'pct_with_helpful',
            'engagement_score',

            # Content features
            'avg_text_length', 'pct_long_reviews',

            # Image features
            'pct_with_images', 'image_count',

            # Sentiment features
            'sentiment_mean', 'sentiment_std',

            # Quality features
            'verified_ratio', 'quality_score',
            'pct_5star', 'pct_1star', 'pct_extreme',

            # Growth features
            'rev_prev4', 'review_momentum',
            'helpful_growth', 'sentiment_trend'
        ]

        # Build text per product
        print("Aggregating product texts...")
        self.product_texts = {}
        if reviews_df is not None:
            for pid, group in tqdm(reviews_df.groupby('parent_asin'), desc="Products"):
                # Use top 5 most helpful reviews
                if 'helpful_vote' in group.columns:
                    top = group.nlargest(5, 'helpful_vote')
                else:
                    top = group.head(5)

                texts = []
                for _, row in top.iterrows():
                    title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
                    text = str(row.get('text', ''))[:300] if pd.notna(row.get('text')) else ''
                    if title:
                        texts.append(title)
                    if text:
                        texts.append(text)

                combined = ' '.join(texts)[:500]
                self.product_texts[pid] = combined

        # Build sequences
        self.samples = []
        print("Building sequences...")

        for pid, g in tqdm(df.groupby('parent_asin'), desc="Sequences"):
            g = g.sort_values('week_start').reset_index(drop=True)
            if len(g) < min_weeks:
                continue

            # Extract features
            try:
                X = torch.tensor(g[self.feature_cols].values, dtype=torch.float32)
            except KeyError as e:
                print(f"Missing feature: {e}, skipping product {pid}")
                continue

            # Handle NaN/Inf
            X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

            y = torch.tensor(g['label_top5'].values, dtype=torch.float32)

            text = self.product_texts.get(pid, "")

            for t in range(seq_len-1, len(g)):
                x_seq = X[t-seq_len+1:t+1]
                y_t = y[t]
                if torch.isnan(y_t):
                    continue
                self.samples.append((x_seq, text, y_t))

        print(f"Created {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, text, y = self.samples[idx]
        text_encoded = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return (
            x_seq,
            text_encoded['input_ids'].squeeze(0),
            text_encoded['attention_mask'].squeeze(0),
            y
        )

def collate_enhanced(batch):
    xs, input_ids, attention_masks, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    ys = torch.stack(ys).unsqueeze(1)
    return xs, input_ids, attention_masks, ys

# ---- Enhanced Model ----
class EnhancedBERTModel(nn.Module):
    def __init__(self, bert_model_name='distilbert-base-uncased',
                 n_features=24, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.2, freeze_bert=False):
        super().__init__()

        # BERT for text
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Text projection with deeper network
        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # Time series feature projection
        self.ts_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Time series transformer
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=num_layers
        )

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Fusion with residual connections
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_ts, input_ids, attention_mask):
        batch_size = x_ts.size(0)

        # Text encoding
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_out.last_hidden_state[:, 0, :]  # CLS token
        text_feat = self.text_proj(text_emb)  # (B, d)

        # Time series encoding
        ts_emb = self.ts_proj(x_ts)  # (B, T, d)
        ts_emb = self.pos_encoder(ts_emb)
        ts_enc = self.ts_encoder(ts_emb)  # (B, T, d)

        # Pooling strategies
        ts_last = ts_enc[:, -1, :]  # Last timestep
        ts_mean = ts_enc.mean(dim=1)  # Average pooling

        # Cross-attention: text attends to time series
        text_expanded = text_feat.unsqueeze(1)  # (B, 1, d)
        cross_out, _ = self.cross_attn(
            query=text_expanded,
            key=ts_enc,
            value=ts_enc
        )
        cross_feat = cross_out.squeeze(1)  # (B, d)

        # Fusion
        combined = torch.cat([cross_feat, ts_last, ts_mean], dim=1)  # (B, 3*d)
        logit = self.fusion(combined)

        return logit

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ---- Training ----
def train_loop(model, train_loader, val_loader, device, epochs=15, lr=2e-4, accumulation_steps=2):
    # Differential learning rates
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': lr / 10},
        {'params': other_params, 'lr': lr}
    ], weight_decay=1e-4)

    # Warmup + cosine schedule
    total_steps = len(train_loader) * epochs // accumulation_steps
    warmup_steps = total_steps // 10

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr / 10, lr],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Focal loss for imbalanced data
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal_loss.mean()

    criterion = FocalLoss()
    best_val = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0
        n_samples = 0

        optimizer.zero_grad()

        for batch_idx, (xb_ts, input_ids, attention_mask, yb) in enumerate(
            tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        ):
            xb_ts = xb_ts.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            yb = yb.to(device)

            logit = model(xb_ts, input_ids, attention_mask)
            loss = criterion(logit, yb) / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            tot_loss += loss.item() * xb_ts.size(0) * accumulation_steps
            n_samples += xb_ts.size(0)

        avg_loss = tot_loss / max(1, n_samples)

        # Validation
        model.eval()
        corr = 0
        total = 0

        with torch.no_grad():
            for xb_ts, input_ids, attention_mask, yb in tqdm(val_loader, desc="Validation"):
                xb_ts = xb_ts.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                yb = yb.to(device)

                logit = model(xb_ts, input_ids, attention_mask)
                p = torch.sigmoid(logit)
                pred = (p >= 0.5).float()
                corr += (pred.eq(yb)).sum().item()
                total += yb.numel()

        acc = corr / max(1, total)

        print(f"\nEpoch {ep}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Accuracy: {acc:.4f}")
        print(f"  Best Val Acc: {best_val:.4f}")

        if acc > best_val:
            best_val = acc
            yield {"epoch": ep, "val_acc": acc, "train_loss": avg_loss}

def main(args):
    print("Loading panel data...")
    df = pd.read_csv(args.data, parse_dates=['week_start'])

    print("Loading reviews...")
    reviews_df = pd.read_parquet(
        args.reviews_file,
        columns=['parent_asin', 'title', 'text', 'helpful_vote']
    )

    print(f"Loaded {len(df):,} panel rows, {len(reviews_df):,} reviews")

    # Tokenizer
    print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Split
    last_week = df['week_start'].max()
    cutoff = last_week - pd.Timedelta(weeks=26)
    train_df = df[df['week_start'] <= cutoff].copy()
    val_df = df[df['week_start'] > cutoff].copy()

    print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,}")

    # Datasets
    train_ds = EnhancedPanelDataset(train_df, reviews_df, tokenizer,
                                    seq_len=args.seq_len, text_max_len=args.text_max_len)
    val_ds = EnhancedPanelDataset(val_df, reviews_df, tokenizer,
                                  seq_len=args.seq_len, text_max_len=args.text_max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_enhanced,
                             pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_enhanced,
                           pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = EnhancedBERTModel(
        bert_model_name=args.bert_model,
        n_features=24,  # Number of enhanced features
        d_model=args.d_model,
        freeze_bert=args.freeze_bert
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    best = None
    for ckpt in train_loop(model, train_loader, val_loader, device,
                          epochs=args.epochs, lr=args.lr,
                          accumulation_steps=args.accumulation_steps):
        best = ckpt
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': vars(args),
            'metrics': ckpt
        }, outdir / "model_enhanced.pt")

    if best:
        (outdir / "best_enhanced.json").write_text(json.dumps(best, indent=2))

    print(f"\nDone! Saved to {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reviews_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bert_model", default="distilbert-base-uncased")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--text_max_len", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--freeze_bert", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    args = ap.parse_args()
    main(args)