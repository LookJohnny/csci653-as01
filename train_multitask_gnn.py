"""
Advanced Multi-Task Model with Graph Neural Networks

Features:
- Multi-task learning (predict hot-seller + rating + engagement)
- Graph Neural Network for product-category relationships
- Temporal Fusion Transformer components
- Attention mechanisms across modalities
"""
import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ---- Graph Construction ----
class ProductGraph:
    """Build graph connecting products through shared categories and users"""
    def __init__(self, df, reviews_df, max_neighbors=20):
        print("Building product graph...")

        self.product_ids = sorted(df['parent_asin'].unique())
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        self.n_products = len(self.product_ids)

        # Build adjacency based on co-reviews (users who reviewed both)
        self.edges = defaultdict(set)

        print("Finding co-review relationships...")
        user_products = reviews_df.groupby('user_id')['parent_asin'].apply(set).to_dict()

        for user, products in tqdm(user_products.items(), desc="Building edges"):
            products = list(products)
            for i, p1 in enumerate(products):
                for p2 in products[i+1:]:
                    if p1 in self.pid_to_idx and p2 in self.pid_to_idx:
                        self.edges[p1].add(p2)
                        self.edges[p2].add(p1)

        # Limit neighbors
        for pid in self.edges:
            if len(self.edges[pid]) > max_neighbors:
                self.edges[pid] = set(list(self.edges[pid])[:max_neighbors])

        print(f"Graph: {self.n_products} nodes, {sum(len(v) for v in self.edges.values()) // 2} edges")

    def get_neighbors(self, pid):
        """Get neighbor indices for a product"""
        if pid not in self.pid_to_idx:
            return []
        neighbors = self.edges.get(pid, set())
        return [self.pid_to_idx[n] for n in neighbors if n in self.pid_to_idx]

# ---- Multi-Task Dataset ----
class MultiTaskDataset(Dataset):
    def __init__(self, df, reviews_df, tokenizer, graph, seq_len=32,
                 min_weeks=12, text_max_len=128):
        self.seq_len = seq_len
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.graph = graph

        # Features
        self.feature_cols = [
            'reviews', 'rating_mean', 'rating_std',
            'helpful_sum', 'helpful_mean', 'pct_with_helpful',
            'engagement_score', 'avg_text_length', 'pct_long_reviews',
            'pct_with_images', 'image_count', 'sentiment_mean', 'sentiment_std',
            'verified_ratio', 'quality_score', 'pct_5star', 'pct_1star',
            'rev_prev4', 'review_momentum', 'helpful_growth', 'sentiment_trend'
        ]

        # Product texts
        print("Aggregating product texts...")
        self.product_texts = {}
        if reviews_df is not None:
            for pid, group in tqdm(reviews_df.groupby('parent_asin'), desc="Texts"):
                top = group.nlargest(5, 'helpful_vote') if 'helpful_vote' in group.columns else group.head(5)
                texts = []
                for _, row in top.iterrows():
                    title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
                    text = str(row.get('text', ''))[:300] if pd.notna(row.get('text')) else ''
                    if title:
                        texts.append(title)
                    if text:
                        texts.append(text)
                self.product_texts[pid] = ' '.join(texts)[:500]

        # Build samples
        self.samples = []
        print("Building multi-task samples...")

        for pid, g in tqdm(df.groupby('parent_asin'), desc="Samples"):
            g = g.sort_values('week_start').reset_index(drop=True)
            if len(g) < min_weeks:
                continue

            try:
                X = torch.tensor(g[self.feature_cols].values, dtype=torch.float32)
            except KeyError:
                continue

            X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

            # Multiple targets for multi-task learning
            y_hotseller = torch.tensor(g['label_top5'].values, dtype=torch.float32)
            y_rating = torch.tensor(g['rating_mean'].values, dtype=torch.float32) / 5.0  # Normalize to [0,1]
            y_engagement = torch.tensor(g['engagement_score'].values, dtype=torch.float32)
            y_engagement = (y_engagement - y_engagement.mean()) / (y_engagement.std() + 1e-6)  # Standardize

            text = self.product_texts.get(pid, "")
            neighbors = self.graph.get_neighbors(pid)
            pid_idx = self.graph.pid_to_idx.get(pid, 0)

            for t in range(seq_len-1, len(g)):
                x_seq = X[t-seq_len+1:t+1]

                y_hs = y_hotseller[t]
                y_rt = y_rating[t]
                y_eng = y_engagement[t]

                if torch.isnan(y_hs) or torch.isnan(y_rt):
                    continue

                self.samples.append((
                    x_seq, text, pid_idx, neighbors,
                    y_hs, y_rt, y_eng
                ))

        print(f"Created {len(self.samples)} multi-task samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, text, pid_idx, neighbors, y_hs, y_rt, y_eng = self.samples[idx]

        text_encoded = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Pad neighbors
        neighbor_tensor = torch.zeros(20, dtype=torch.long)
        if neighbors:
            neighbor_tensor[:len(neighbors)] = torch.tensor(neighbors[:20], dtype=torch.long)

        return (
            x_seq,
            text_encoded['input_ids'].squeeze(0),
            text_encoded['attention_mask'].squeeze(0),
            torch.tensor(pid_idx, dtype=torch.long),
            neighbor_tensor,
            torch.tensor(len(neighbors), dtype=torch.long),
            y_hs, y_rt, y_eng
        )

def collate_multitask(batch):
    xs, input_ids, attn_masks, pids, neighbors, n_neighbors, y_hs, y_rt, y_eng = zip(*batch)

    xs = pad_sequence(xs, batch_first=True)
    input_ids = torch.stack(input_ids)
    attn_masks = torch.stack(attn_masks)
    pids = torch.stack(pids)
    neighbors = torch.stack(neighbors)
    n_neighbors = torch.stack(n_neighbors)

    y_hs = torch.stack(y_hs).unsqueeze(1)
    y_rt = torch.stack(y_rt).unsqueeze(1)
    y_eng = torch.stack(y_eng).unsqueeze(1)

    return xs, input_ids, attn_masks, pids, neighbors, n_neighbors, y_hs, y_rt, y_eng

# ---- GNN Layer ----
class GraphConvLayer(nn.Module):
    """Graph convolution for product relationships"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_feats, adj_matrix):
        """
        node_feats: (B, N, D)
        adj_matrix: (B, N, N) adjacency matrix
        """
        # Aggregate neighbor features
        aggregated = torch.bmm(adj_matrix, node_feats)  # (B, N, D)

        # Transform
        out = self.linear(aggregated)
        out = self.norm(out)
        out = F.relu(out)

        return out

# ---- Multi-Task Model ----
class MultiTaskGNNModel(nn.Module):
    def __init__(self, bert_model_name='distilbert-base-uncased',
                 n_features=21, n_products=10000, d_model=256,
                 nhead=8, num_layers=4, dropout=0.2, freeze_bert=False):
        super().__init__()

        self.n_products = n_products

        # BERT for text
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Product embeddings for GNN
        self.product_embeds = nn.Embedding(n_products, d_model)

        # GNN layers
        self.gnn1 = GraphConvLayer(d_model, d_model)
        self.gnn2 = GraphConvLayer(d_model, d_model)

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Time series encoder
        self.ts_proj = nn.Linear(n_features, d_model)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=num_layers
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Task-specific heads
        self.hotseller_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        self.rating_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.engagement_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x_ts, input_ids, attn_mask, product_ids, neighbors, n_neighbors):
        batch_size = x_ts.size(0)

        # 1. Text encoding with BERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        text_feat = self.text_proj(bert_out.last_hidden_state[:, 0, :])  # (B, d)

        # 2. Time series encoding
        ts_emb = self.ts_proj(x_ts)  # (B, T, d)
        ts_enc = self.ts_encoder(ts_emb)  # (B, T, d)
        ts_pooled = ts_enc[:, -1, :]  # (B, d)

        # 3. GNN for product relationships
        # Get product and neighbor embeddings
        prod_emb = self.product_embeds(product_ids)  # (B, d)

        # Build adjacency matrix (simple version for batching)
        neighbor_embs = self.product_embeds(neighbors)  # (B, max_neighbors, d)

        # Stack for GNN input
        node_feats = torch.stack([prod_emb] + [neighbor_embs[:, i] for i in range(20)], dim=1)  # (B, 21, d)

        # Simple adjacency (connect center to all neighbors)
        adj = torch.zeros(batch_size, 21, 21, device=x_ts.device)
        adj[:, 0, 1:] = 1.0  # Center to neighbors
        adj[:, 1:, 0] = 1.0  # Neighbors to center

        # Mask based on actual neighbor count
        for i in range(batch_size):
            n = n_neighbors[i].item()
            if n < 20:
                adj[i, 0, n+1:] = 0
                adj[i, n+1:, 0] = 0

        # Normalize adjacency
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / deg

        # Apply GNN layers
        gnn_out = self.gnn1(node_feats, adj)
        gnn_out = self.gnn2(gnn_out, adj)
        graph_feat = gnn_out[:, 0, :]  # Center node (B, d)

        # 4. Cross-attention between text and time series
        text_expanded = text_feat.unsqueeze(1)  # (B, 1, d)
        cross_out, _ = self.cross_attn(text_expanded, ts_enc, ts_enc)
        cross_feat = cross_out.squeeze(1)  # (B, d)

        # 5. Fusion
        combined = torch.cat([text_feat, ts_pooled, graph_feat, cross_feat], dim=1)  # (B, 4*d)
        shared = self.shared_backbone(combined)  # (B, d)

        # 6. Multi-task outputs
        hotseller_logit = self.hotseller_head(shared)
        rating_pred = self.rating_head(shared)
        engagement_pred = self.engagement_head(shared)

        return hotseller_logit, rating_pred, engagement_pred

# ---- Training ----
def train_multitask(model, train_loader, val_loader, device, epochs=20, lr=2e-4):
    # Optimizer
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': lr / 10},
        {'params': other_params, 'lr': lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # Task weights
    w_hs = 1.0  # Hot-seller (main task)
    w_rt = 0.3  # Rating (auxiliary)
    w_eng = 0.2  # Engagement (auxiliary)

    best_val = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0
        n_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            xs, input_ids, attn_mask, pids, neighbors, n_neighbors, y_hs, y_rt, y_eng = batch

            xs = xs.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            pids = pids.to(device)
            neighbors = neighbors.to(device)
            n_neighbors = n_neighbors.to(device)
            y_hs = y_hs.to(device)
            y_rt = y_rt.to(device)
            y_eng = y_eng.to(device)

            optimizer.zero_grad()

            hs_logit, rt_pred, eng_pred = model(xs, input_ids, attn_mask, pids, neighbors, n_neighbors)

            # Multi-task loss
            loss_hs = bce_loss(hs_logit, y_hs)
            loss_rt = mse_loss(rt_pred, y_rt)
            loss_eng = mse_loss(eng_pred, y_eng)

            loss = w_hs * loss_hs + w_rt * loss_rt + w_eng * loss_eng

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_loss += loss.item() * xs.size(0)
            n_samples += xs.size(0)

        scheduler.step()
        avg_loss = tot_loss / max(1, n_samples)

        # Validation
        model.eval()
        corr = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                xs, input_ids, attn_mask, pids, neighbors, n_neighbors, y_hs, y_rt, y_eng = batch

                xs = xs.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                pids = pids.to(device)
                neighbors = neighbors.to(device)
                n_neighbors = n_neighbors.to(device)
                y_hs = y_hs.to(device)

                hs_logit, _, _ = model(xs, input_ids, attn_mask, pids, neighbors, n_neighbors)

                pred = (torch.sigmoid(hs_logit) >= 0.5).float()
                corr += (pred.eq(y_hs)).sum().item()
                total += y_hs.numel()

        acc = corr / max(1, total)

        print(f"\nEpoch {ep}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Accuracy: {acc:.4f}")
        print(f"  Best: {best_val:.4f}")

        if acc > best_val:
            best_val = acc
            yield {"epoch": ep, "val_acc": acc, "train_loss": avg_loss}

# ---- Main ----
def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data, parse_dates=['week_start'])
    reviews_df = pd.read_parquet(args.reviews_file,
                                 columns=['parent_asin', 'user_id', 'title', 'text', 'helpful_vote'])

    print("Building product graph...")
    graph = ProductGraph(df, reviews_df, max_neighbors=20)

    print("Loading tokenizer...")
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
                             pin_memory=True)
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

    # Train
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    best = None
    for ckpt in train_multitask(model, train_loader, val_loader, device,
                               epochs=args.epochs, lr=args.lr):
        best = ckpt
        torch.save({
            'model_state_dict': model.state_dict(),
            'graph': graph,
            'metrics': ckpt
        }, outdir / "model_gnn_multitask.pt")

    if best:
        (outdir / "best_gnn.json").write_text(json.dumps(best, indent=2))

    print(f"Done! Saved to {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reviews_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bert_model", default="distilbert-base-uncased")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--text_max_len", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--freeze_bert", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    main(args)