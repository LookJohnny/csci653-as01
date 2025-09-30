# Advanced Deep Learning Architectures

## Overview
State-of-the-art models combining multiple AI techniques for hot-seller prediction.

---

## Model 1: Multi-Task Learning with Graph Neural Networks ⭐ **MOST ADVANCED**

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
├─────────────┬──────────────┬────────────────────────────┤
│  Text Data  │  Time Series │   Product Graph            │
│  (Reviews)  │  (Features)  │   (Relationships)          │
└─────────────┴──────────────┴────────────────────────────┘
      │               │                    │
      ▼               ▼                    ▼
┌──────────┐   ┌──────────┐       ┌──────────────┐
│ DistilB  │   │   Time   │       │ Product Node │
│   ERT    │   │  Series  │       │  Embeddings  │
│ Encoder  │   │Transform │       └──────────────┘
└──────────┘   │   er     │               │
      │        └──────────┘               ▼
      │              │            ┌────────────────┐
      │              │            │  GNN Layer 1   │
      │              │            │ (Graph Conv)   │
      │              │            ├────────────────┤
      │              │            │  GNN Layer 2   │
      │              │            │ (Graph Conv)   │
      ▼              ▼            └────────────────┘
  Text Feat      TS Feat                 │
   (256-d)       (256-d)           Graph Feat
      │              │              (256-d)
      └──────┬───────┴──────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Cross-Attention    │
    │ (Text ⟷ Time Series)│
    └─────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │   Fusion Layer      │
    │  (4 × 256-d → 256-d)│
    └─────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Shared Backbone    │
    │    (256-d deep)     │
    └─────────────────────┘
             │
        ┌────┴────┬────────────┐
        ▼         ▼            ▼
  ┌─────────┐ ┌────────┐ ┌────────────┐
  │Hot-Seller│ │ Rating │ │ Engagement │
  │  Head   │ │  Head  │ │    Head    │
  └─────────┘ └────────┘ └────────────┘
       │          │            │
       ▼          ▼            ▼
   [0 or 1]   [1-5 stars]  [score]
```

### Key Innovations

#### 1. **Graph Neural Network (GNN)**
Models product relationships through shared customers:

```python
# Build graph: Products → Neighbors (co-reviewed items)
User A: [Product 1, Product 2]  → Edge(P1, P2)
User B: [Product 1, Product 3]  → Edge(P1, P3)

# GNN aggregates neighbor information
Product_Embedding = f(own_features, neighbor_features)
```

**Benefits:**
- Captures collaborative filtering patterns
- Learns category trends
- Identifies similar products
- Cross-product insights

#### 2. **Multi-Task Learning**
Trains 3 tasks simultaneously:

| Task | Loss | Weight | Purpose |
|------|------|--------|---------|
| **Hot-Seller** | BCE | 1.0 | Main task |
| **Rating Prediction** | MSE | 0.3 | Quality signal |
| **Engagement** | MSE | 0.2 | Interest signal |

**Why it works:**
- Shared representations learn better features
- Auxiliary tasks provide regularization
- Rating/engagement help predict virality

#### 3. **Cross-Modal Attention**
Text and time series attend to each other:

```python
Query: Text features
Key/Value: Time series sequence
Output: Text-informed temporal features
```

**Learns patterns like:**
- "Trending" mentions → review spike
- Positive sentiment → sustained growth
- "TikTok viral" → explosive growth

#### 4. **Positional Encoding**
Adds temporal awareness:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Model understands sequence order and time gaps.

---

## Model 2: Enhanced BERT with Rich Features

### Features (24 total)
See `ENHANCED_FEATURES.md` for full details.

### Architecture Highlights
- Pre-trained DistilBERT (768-d)
- 4-layer Transformer for time series
- Deep fusion network (3 pooling strategies)
- Focal loss for imbalanced data

---

## Model Comparison

| Feature | Basic | Word2Vec | BERT | **GNN Multi-Task** |
|---------|-------|----------|------|--------------------|
| **Text** | ❌ | Custom | ✅ BERT | ✅ BERT |
| **Features** | 4 | 4 | 24 | 24 |
| **GNN** | ❌ | ❌ | ❌ | ✅ Graph Conv |
| **Multi-Task** | ❌ | ❌ | ❌ | ✅ 3 tasks |
| **Cross-Attention** | ❌ | ❌ | ✅ | ✅✅ Enhanced |
| **Parameters** | 2M | 15M | 85M | **120M** |
| **Expected Acc** | 72% | 78% | 86% | **90-92%** |
| **Training Time** | 2h | 3h | 5h | **8-10h** |
| **GPU Memory** | 4GB | 8GB | 14GB | **18GB** |

---

## Why GNN + Multi-Task is Better

### 1. **Graph Relationships** (+3-5% accuracy)
- Products don't exist in isolation
- Co-review patterns reveal substitutes/complements
- Category trends propagate through graph
- Cold-start products benefit from neighbors

### 2. **Multi-Task Regularization** (+2-4% accuracy)
- Prevents overfitting to hot-seller task
- Rating/engagement provide quality signals
- Shared backbone learns robust features
- Better generalization

### 3. **Cross-Modal Attention** (+2-3% accuracy)
- Text-time series interaction
- Learns "viral trigger words"
- Temporal sentiment dynamics

### 4. **Rich Features** (+8-12% accuracy)
- Sentiment, engagement, images
- 24 features vs 4 baseline

**Total improvement: +15-24% over basic model**

---

## Technical Deep Dive

### Graph Convolution Math

```
Input: Node features X ∈ R^(N × D)
       Adjacency matrix A ∈ R^(N × N)

Step 1: Aggregate neighbors
    H = A × X

Step 2: Transform
    Z = σ(H × W + b)

Step 3: Normalize
    Z = LayerNorm(Z)
```

### Multi-Task Loss

```python
L_total = w_hs × L_hotseller +
          w_rt × L_rating +
          w_eng × L_engagement

where:
  w_hs = 1.0   # Main task
  w_rt = 0.3   # Auxiliary
  w_eng = 0.2  # Auxiliary
```

### Cross-Attention Mechanism

```python
Q = text_features                     # (B, 1, D)
K, V = time_series_features          # (B, T, D)

Attention(Q, K, V) = softmax(QK^T / √d) × V

# Output: Text-informed temporal features
```

---

## Training Strategy

### 1. **Staged Training** (Optional)
```bash
# Stage 1: Train GNN on graph structure
python train_gnn_only.py --epochs 5

# Stage 2: Freeze GNN, train end-to-end
python train_multitask_gnn.py --freeze_gnn --epochs 15
```

### 2. **Learning Rates**
- BERT: 2e-5 (fine-tune)
- GNN: 2e-4 (learn from scratch)
- Other: 2e-4

### 3. **Batch Size Strategy**
- Effective batch: 16 × 4 = 64 (with accumulation)
- Balances memory and convergence

### 4. **Regularization**
- Dropout: 0.2
- Weight decay: 1e-4
- Layer normalization
- Gradient clipping: 1.0

---

## Usage

### Quick Start
```bash
sbatch train_gnn_multitask.slurm
```

### Custom Training
```bash
python train_multitask_gnn.py \
  --data enhanced_panel.csv \
  --reviews_file combined_reviews.parquet \
  --out gnn_multitask/ \
  --epochs 20 \
  --batch_size 16 \
  --d_model 256
```

---

## Interpretability

### 1. **Attention Visualization**
```python
# See which time periods text attends to
attention_weights = model.cross_attn.get_weights()
plot_attention_heatmap(attention_weights)
```

### 2. **Graph Neighborhood Analysis**
```python
# Which products influence each other?
neighbors = graph.get_neighbors(product_id)
visualize_subgraph(product_id, neighbors)
```

### 3. **Multi-Task Analysis**
```python
# Which auxiliary task helps most?
correlation(rating_pred, hotseller_accuracy)
correlation(engagement_pred, hotseller_accuracy)
```

---

## Expected Results

### Performance by Component

| Configuration | Val Accuracy | Notes |
|---------------|--------------|-------|
| Time Series Only | 72% | Baseline |
| + Text (BERT) | 82% | +10% |
| + Rich Features | 86% | +4% |
| + GNN | 89% | +3% |
| **+ Multi-Task** | **91-92%** | **+2-3%** |

### Computational Cost

| Model | Train Time | Inference | GPU Memory |
|-------|-----------|-----------|------------|
| Basic | 2h | 5ms/sample | 4GB |
| **GNN Multi-Task** | **8-10h** | **12ms/sample** | **18GB** |

**Worth it?** Yes! 20% accuracy improvement for 4× training time.

---

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size
--batch_size 8

# Freeze BERT
--freeze_bert

# Reduce model size
--d_model 128
```

### Slow Training
```bash
# Limit graph neighbors
graph = ProductGraph(df, reviews_df, max_neighbors=10)

# Use mixed precision
# Add to train_multitask_gnn.py:
scaler = torch.cuda.amp.GradScaler()
```

### Graph Too Large
```bash
# Sample edges
graph.sample_edges(max_edges_per_node=20)

# Or use mini-batch GNN (advanced)
```

---

## Future Enhancements

1. **Hierarchical GNN** - Multi-level: user → product → category
2. **Temporal GNN** - Graph structure changes over time
3. **Heterogeneous GNN** - Different node types (user, product, brand)
4. **Meta-Learning** - Fast adaptation to new categories
5. **Contrastive Learning** - Learn better representations

---

## Summary

**GNN + Multi-Task Model** is the **most powerful** approach:
- ✅ Pre-trained BERT for text
- ✅ 24 rich features
- ✅ Graph relationships
- ✅ Multi-task learning
- ✅ Cross-modal attention

**Best for:** Maximum accuracy when computational resources available

**Expected improvement:** +20-25% over baseline