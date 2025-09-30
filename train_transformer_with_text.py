import argparse, os, math, json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---- Text Processing ----
class SimpleVocab:
    def __init__(self, texts, max_vocab=10000, min_freq=5):
        """Build vocabulary from texts"""
        counter = Counter()
        for text in texts:
            if pd.notna(text):
                tokens = self.tokenize(text)
                counter.update(tokens)

        # Keep most common words
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in counter.most_common(max_vocab):
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"Vocabulary size: {len(self.word2idx)}")

    def tokenize(self, text):
        """Simple tokenization"""
        if pd.isna(text):
            return []
        text = str(text).lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def encode(self, text, max_len=50):
        """Convert text to indices"""
        tokens = self.tokenize(text)[:max_len]
        indices = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]
        return torch.tensor(indices, dtype=torch.long)

# ---- Dataset with Text ----
class PanelSeqWithTextDataset(Dataset):
    def __init__(self, df, reviews_df, vocab, seq_len=32, min_weeks=12, text_len=50):
        """
        df: weekly panel (parent_asin, week_start, features, label)
        reviews_df: raw reviews with title and text
        vocab: vocabulary for text encoding
        """
        self.seq_len = seq_len
        self.text_len = text_len
        self.vocab = vocab

        # Build text aggregation per product
        print("Aggregating text per product...")
        self.product_texts = {}
        if reviews_df is not None and 'parent_asin' in reviews_df.columns:
            for pid, group in reviews_df.groupby('parent_asin'):
                # Concatenate top review texts
                texts = []
                for _, row in group.head(10).iterrows():  # Use top 10 reviews
                    if pd.notna(row.get('title')):
                        texts.append(str(row['title']))
                    if pd.notna(row.get('text')):
                        texts.append(str(row['text'])[:200])  # Limit text length
                combined = ' '.join(texts)
                self.product_texts[pid] = combined

        # Build sequences
        feats = ["reviews","helpful_sum","verified_ratio","rating_mean"]
        self.samples = []
        print("Building sequences...")
        for pid, g in df.groupby("parent_asin"):
            g = g.sort_values("week_start").reset_index(drop=True)
            if len(g) < min_weeks:
                continue
            X = torch.tensor(g[feats].values, dtype=torch.float32)
            y = torch.tensor(g["label_top5"].values, dtype=torch.float32)

            # Get text for this product
            text = self.product_texts.get(pid, "")
            text_encoded = self.vocab.encode(text, max_len=self.text_len)

            for t in range(seq_len-1, len(g)):
                x_seq = X[t-seq_len+1:t+1]
                y_t = y[t]
                if torch.isnan(y_t):
                    continue
                self.samples.append((x_seq, text_encoded, y_t))

        print(f"Created {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_with_text(batch):
    xs, texts, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    ys = torch.stack(ys).unsqueeze(1)
    return xs, texts_padded, ys

# ---- Enhanced Model with Text ----
class TransformerWithText(nn.Module):
    def __init__(self, vocab_size, n_features=4, embed_dim=128, d_model=128,
                 nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()

        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                      dim_feedforward=256, dropout=dropout,
                                      batch_first=True),
            num_layers=2
        )

        # Time series encoder
        self.ts_proj = nn.Linear(n_features, d_model)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                      dim_feedforward=dim_feedforward,
                                      dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_ts, x_text):
        # Encode text
        text_emb = self.text_embedding(x_text)  # (B, L, E)
        text_enc = self.text_encoder(text_emb)  # (B, L, E)
        text_pooled = text_enc.mean(dim=1)  # (B, E) - average pooling

        # Encode time series
        ts_emb = self.ts_proj(x_ts)  # (B, T, d)
        ts_enc = self.ts_encoder(ts_emb)  # (B, T, d)
        ts_pooled = ts_enc[:, -1, :]  # (B, d) - last token

        # Fusion
        combined = torch.cat([text_pooled, ts_pooled], dim=1)  # (B, E+d)
        logit = self.fusion(combined)  # (B, 1)

        return logit

def train_loop(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    best_val = 0.0

    for ep in range(1, epochs+1):
        model.train()
        tot = 0; n = 0
        for xb_ts, xb_text, yb in train_loader:
            xb_ts = xb_ts.to(device)
            xb_text = xb_text.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logit = model(xb_ts, xb_text)
            loss = bce(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb_ts.size(0)
            n += xb_ts.size(0)
        sched.step()

        # Validation
        model.eval()
        corr = 0; m = 0
        with torch.no_grad():
            for xb_ts, xb_text, yb in val_loader:
                xb_ts = xb_ts.to(device)
                xb_text = xb_text.to(device)
                yb = yb.to(device)
                p = torch.sigmoid(model(xb_ts, xb_text))
                pred = (p >= 0.5).float()
                corr += (pred.eq(yb)).sum().item()
                m += yb.numel()

        acc = corr / max(1, m)
        print(f"Epoch {ep}/{epochs}: train_loss={tot/max(1,n):.4f} val_acc={acc:.4f}")

        if acc > best_val:
            best_val = acc
            yield {"epoch": ep, "val_acc": acc}

def main(args):
    # Load panel data
    df = pd.read_csv(args.data, parse_dates=["week_start"])

    # Load reviews for text
    print("Loading review texts...")
    if args.reviews_file.endswith('.parquet'):
        reviews_df = pd.read_parquet(args.reviews_file,
                                     columns=['parent_asin', 'title', 'text'])
    else:
        reviews_df = pd.read_csv(args.reviews_file,
                                usecols=['parent_asin', 'title', 'text'])

    # Build vocabulary
    print("Building vocabulary...")
    all_texts = pd.concat([reviews_df['title'], reviews_df['text']]).dropna()
    vocab = SimpleVocab(all_texts, max_vocab=args.vocab_size, min_freq=args.min_freq)

    # Save vocab
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    vocab_path = outdir / "vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab.word2idx, f)
    print(f"Saved vocabulary to {vocab_path}")

    # Time-based split
    last_week = df["week_start"].max()
    cutoff = last_week - pd.Timedelta(weeks=26)
    train_df = df[df["week_start"] <= cutoff].copy()
    val_df = df[df["week_start"] > cutoff].copy()

    # Create datasets
    train_ds = PanelSeqWithTextDataset(train_df, reviews_df, vocab,
                                       seq_len=args.seq_len, text_len=args.text_len)
    val_ds = PanelSeqWithTextDataset(val_df, reviews_df, vocab,
                                     seq_len=args.seq_len, text_len=args.text_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, collate_fn=collate_with_text, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, collate_fn=collate_with_text, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerWithText(
        vocab_size=len(vocab.word2idx),
        embed_dim=args.embed_dim,
        d_model=args.d_model
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    best = None
    for ckpt in train_loop(model, train_loader, val_loader, device,
                          epochs=args.epochs, lr=args.lr):
        best = ckpt
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': len(vocab.word2idx),
            'args': vars(args)
        }, outdir / "model_with_text.pt")

    # Validation predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for xb_ts, xb_text, yb in val_loader:
            p = torch.sigmoid(model(xb_ts.to(device), xb_text.to(device)))
            preds.extend(p.squeeze(1).cpu().numpy().tolist())

    pd.DataFrame({"pred": preds}).to_csv(outdir / "preds_with_text.csv", index=False)

    if best:
        (outdir / "best_with_text.json").write_text(json.dumps(best, indent=2))

    print(f"Saved all artifacts to {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Transformer with text embeddings")
    ap.add_argument("--data", required=True, help="Weekly panel CSV")
    ap.add_argument("--reviews_file", required=True, help="Reviews parquet/CSV with text")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--seq_len", type=int, default=32, help="Time series sequence length")
    ap.add_argument("--text_len", type=int, default=50, help="Max text length")
    ap.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    ap.add_argument("--min_freq", type=int, default=5, help="Min word frequency")
    ap.add_argument("--embed_dim", type=int, default=128, help="Text embedding dimension")
    ap.add_argument("--d_model", type=int, default=128, help="Time series model dimension")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = ap.parse_args()
    main(args)