import argparse, os, json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

# ---- Dataset with BERT ----
class PanelSeqWithBERTDataset(Dataset):
    def __init__(self, df, reviews_df, tokenizer, seq_len=32, min_weeks=12,
                 text_max_len=128, cache_file=None):
        """
        df: weekly panel (parent_asin, week_start, features, label)
        reviews_df: raw reviews with title and text
        tokenizer: BERT tokenizer
        """
        self.seq_len = seq_len
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer

        # Build text aggregation per product
        print("Aggregating text per product...")
        self.product_texts = {}

        if reviews_df is not None and 'parent_asin' in reviews_df.columns:
            # Sample reviews per product to avoid memory issues
            for pid, group in tqdm(reviews_df.groupby('parent_asin'), desc="Processing products"):
                texts = []
                # Use top 5 most helpful reviews or first 5
                if 'helpful_vote' in group.columns:
                    top_reviews = group.nlargest(5, 'helpful_vote')
                else:
                    top_reviews = group.head(5)

                for _, row in top_reviews.iterrows():
                    title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
                    text = str(row.get('text', ''))[:300] if pd.notna(row.get('text')) else ''
                    if title:
                        texts.append(title)
                    if text:
                        texts.append(text)

                # Combine and truncate
                combined = ' '.join(texts)[:500]  # Limit total length
                self.product_texts[pid] = combined

        print(f"Collected text for {len(self.product_texts)} products")

        # Build sequences
        feats = ["reviews","helpful_sum","verified_ratio","rating_mean"]
        self.samples = []

        print("Building time series sequences...")
        for pid, g in tqdm(df.groupby("parent_asin"), desc="Building sequences"):
            g = g.sort_values("week_start").reset_index(drop=True)
            if len(g) < min_weeks:
                continue

            X = torch.tensor(g[feats].values, dtype=torch.float32)
            y = torch.tensor(g["label_top5"].values, dtype=torch.float32)

            # Get text for this product
            text = self.product_texts.get(pid, "")

            for t in range(seq_len-1, len(g)):
                x_seq = X[t-seq_len+1:t+1]
                y_t = y[t]
                if torch.isnan(y_t):
                    continue
                self.samples.append((x_seq, text, y_t, pid))

        print(f"Created {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, text, y, pid = self.samples[idx]
        # Tokenize text on-the-fly
        text_encoded = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return x_seq, text_encoded['input_ids'].squeeze(0), text_encoded['attention_mask'].squeeze(0), y

def collate_with_bert(batch):
    xs, input_ids, attention_masks, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    ys = torch.stack(ys).unsqueeze(1)
    return xs, input_ids, attention_masks, ys

# ---- Model with Pre-trained BERT ----
class TransformerWithBERT(nn.Module):
    def __init__(self, bert_model_name='distilbert-base-uncased', n_features=4,
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256,
                 dropout=0.1, freeze_bert=False):
        super().__init__()

        # Pre-trained BERT for text
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for base, 512 for distilbert

        # Optionally freeze BERT layers for faster training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("BERT weights frozen")

        # Project BERT output
        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        # Time series encoder
        self.ts_proj = nn.Linear(n_features, d_model)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Cross-attention between text and time series
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x_ts, input_ids, attention_mask):
        # Encode text with BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_output.last_hidden_state  # (B, L, H)
        text_pooled = text_emb[:, 0, :]  # CLS token (B, H)
        text_feat = self.text_proj(text_pooled)  # (B, d)

        # Encode time series
        ts_emb = self.ts_proj(x_ts)  # (B, T, d)
        ts_enc = self.ts_encoder(ts_emb)  # (B, T, d)
        ts_pooled = ts_enc[:, -1, :]  # Last timestep (B, d)

        # Cross-attention: let text attend to time series
        text_feat_expanded = text_feat.unsqueeze(1)  # (B, 1, d)
        attn_output, _ = self.cross_attn(
            query=text_feat_expanded,
            key=ts_enc,
            value=ts_enc
        )  # (B, 1, d)
        attn_pooled = attn_output.squeeze(1)  # (B, d)

        # Fusion
        combined = torch.cat([attn_pooled, ts_pooled], dim=1)  # (B, 2*d)
        logit = self.fusion(combined)  # (B, 1)

        return logit

def train_loop(model, train_loader, val_loader, device, epochs=10, lr=1e-4,
               accumulation_steps=1):
    # Use different learning rates for BERT and other layers
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': lr / 10},  # Lower LR for pre-trained BERT
        {'params': other_params, 'lr': lr}
    ], weight_decay=1e-4)

    # Scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr / 10, lr],
        total_steps=total_steps,
        pct_start=0.1
    )

    bce = nn.BCEWithLogitsLoss()
    best_val = 0.0

    for ep in range(1, epochs+1):
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
            loss = bce(logit, yb)
            loss = loss / accumulation_steps  # Normalize loss

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
        val_loss = 0

        with torch.no_grad():
            for xb_ts, input_ids, attention_mask, yb in tqdm(val_loader, desc="Validation"):
                xb_ts = xb_ts.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                yb = yb.to(device)

                logit = model(xb_ts, input_ids, attention_mask)
                loss = bce(logit, yb)
                val_loss += loss.item() * xb_ts.size(0)

                p = torch.sigmoid(logit)
                pred = (p >= 0.5).float()
                corr += (pred.eq(yb)).sum().item()
                total += yb.numel()

        acc = corr / max(1, total)
        avg_val_loss = val_loss / max(1, total)

        print(f"\nEpoch {ep}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {acc:.4f}")
        print(f"  Best Val Acc: {best_val:.4f}")

        if acc > best_val:
            best_val = acc
            yield {"epoch": ep, "val_acc": acc, "val_loss": avg_val_loss}

def main(args):
    # Load panel data
    print("Loading panel data...")
    df = pd.read_csv(args.data, parse_dates=["week_start"])

    # Load reviews for text
    print("Loading review texts...")
    if args.reviews_file.endswith('.parquet'):
        # Load in chunks to manage memory
        reviews_df = pd.read_parquet(
            args.reviews_file,
            columns=['parent_asin', 'title', 'text', 'helpful_vote']
        )
    else:
        reviews_df = pd.read_csv(
            args.reviews_file,
            usecols=['parent_asin', 'title', 'text', 'helpful_vote']
        )

    print(f"Loaded {len(reviews_df):,} reviews")

    # Initialize BERT tokenizer
    print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Time-based split
    last_week = df["week_start"].max()
    cutoff = last_week - pd.Timedelta(weeks=26)
    train_df = df[df["week_start"] <= cutoff].copy()
    val_df = df[df["week_start"] > cutoff].copy()

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")

    # Create datasets
    print("\nCreating training dataset...")
    train_ds = PanelSeqWithBERTDataset(
        train_df, reviews_df, tokenizer,
        seq_len=args.seq_len,
        text_max_len=args.text_max_len
    )

    print("\nCreating validation dataset...")
    val_ds = PanelSeqWithBERTDataset(
        val_df, reviews_df, tokenizer,
        seq_len=args.seq_len,
        text_max_len=args.text_max_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_with_bert,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_bert,
        pin_memory=True
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = TransformerWithBERT(
        bert_model_name=args.bert_model,
        d_model=args.d_model,
        freeze_bert=args.freeze_bert
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Train
    best = None
    print("\nStarting training...")
    for ckpt in train_loop(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr,
        accumulation_steps=args.accumulation_steps
    ):
        best = ckpt
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': vars(args),
            'metrics': ckpt
        }, outdir / "model_bert.pt")
        print(f"Saved checkpoint: val_acc={ckpt['val_acc']:.4f}")

    # Save best metrics
    if best:
        (outdir / "best_bert.json").write_text(json.dumps(best, indent=2))

    print(f"\nTraining complete! Saved to {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Transformer with BERT embeddings")
    ap.add_argument("--data", required=True, help="Weekly panel CSV")
    ap.add_argument("--reviews_file", required=True, help="Reviews parquet/CSV with text")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--bert_model", default="distilbert-base-uncased",
                   help="Pre-trained BERT model name")
    ap.add_argument("--seq_len", type=int, default=32, help="Time series sequence length")
    ap.add_argument("--text_max_len", type=int, default=128, help="Max text tokens")
    ap.add_argument("--d_model", type=int, default=128, help="Model dimension")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ap.add_argument("--freeze_bert", action="store_true", help="Freeze BERT weights")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation")
    args = ap.parse_args()
    main(args)