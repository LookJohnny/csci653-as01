
import argparse, os, math, json
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---- Dataset ----
class PanelSeqDataset(Dataset):
    def __init__(self, df, seq_len=32, min_weeks=12):
        # Expect df columns: parent_asin, week_start, reviews, helpful_sum, verified_ratio, rating_mean, label_top5
        self.seq_len = seq_len
        feats = ["reviews","helpful_sum","verified_ratio","rating_mean"]
        self.samples = []
        for pid, g in df.groupby("parent_asin"):
            g = g.sort_values("week_start").reset_index(drop=True)
            if len(g) < min_weeks:
                continue
            X = torch.tensor(g[feats].values, dtype=torch.float32)
            y = torch.tensor(g["label_top5"].values, dtype=torch.float32)
            for t in range(seq_len-1, len(g)):
                x_seq = X[t-seq_len+1:t+1]
                y_t = y[t]
                if torch.isnan(y_t):
                    continue
                self.samples.append((x_seq, y_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)  # already fixed length
    ys = torch.stack(ys).unsqueeze(1)
    return xs, ys

# ---- Model ----
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features=4, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        z = self.input_proj(x)          # (B, T, d)
        z = self.encoder(z)             # (B, T, d)
        h = z[:, -1, :]                 # last token
        logit = self.cls(h)             # (B, 1)
        return logit

def train_loop(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    best_val = 0.0
    for ep in range(1, epochs+1):
        model.train()
        tot = 0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = bce(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()

        # quick validation accuracy at 0.5 threshold
        model.eval()
        corr = 0; m = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                p = torch.sigmoid(model(xb))
                pred = (p >= 0.5).float()
                corr += (pred.eq(yb)).sum().item()
                m += yb.numel()
        acc = corr / max(1, m)
        print(f"Epoch {ep}: train_loss={tot/max(1,n):.4f} val_acc={acc:.4f}")
        if acc > best_val:
            best_val = acc
            yield {"epoch": ep, "val_acc": acc}

def main(args):
    df = pd.read_csv(args.data, parse_dates=["week_start"])

    # time-based split: last 26 weeks for validation
    last_week = df["week_start"].max()
    cutoff = last_week - pd.Timedelta(weeks=26)
    train_df = df[df["week_start"] <= cutoff].copy()
    val_df   = df[df["week_start"]  > cutoff].copy()

    train_ds = PanelSeqDataset(train_df, seq_len=args.seq_len)
    val_ds   = PanelSeqDataset(val_df, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer().to(device)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    best = None
    for ckpt in train_loop(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr):
        best = ckpt
        torch.save(model.state_dict(), outdir / "model.pt")

    # produce validation predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = torch.sigmoid(model(xb.to(device))).squeeze(1).cpu().numpy()
            preds.extend(p.tolist())
    pd.DataFrame({"pred": preds}).to_csv(outdir / "preds.csv", index=False)

    if best:
        (outdir / "best.json").write_text(json.dumps(best, indent=2))
    print("Saved artifacts to", str(outdir))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    main(args)
