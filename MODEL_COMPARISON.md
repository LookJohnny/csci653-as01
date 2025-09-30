# Model Comparison Guide

## Three Training Options Available

### 1. **Basic Transformer** (train_transformer.py)
**Features:**
- Time series numeric features only (reviews, ratings, etc.)
- Simple and fast
- Good baseline performance

**Pros:**
- Fast training (~1-2 hours)
- Low memory usage
- Easy to understand

**Cons:**
- Ignores text content
- Limited feature representation

**Use when:** You want quick results or limited compute

---

### 2. **Transformer with Custom Word2Vec** (train_transformer_with_text.py)
**Features:**
- Custom vocabulary built from reviews
- Word embeddings trained from scratch
- Combines text + time series

**Pros:**
- Learns domain-specific embeddings
- More interpretable than BERT
- Moderate resource usage

**Cons:**
- Vocabulary limited to your data
- Needs more data to train good embeddings
- Less sophisticated than pre-trained models

**Use when:** You want text features but limited GPU resources

---

### 3. **Transformer with BERT** (train_transformer_bert.py) ⭐ **RECOMMENDED**
**Features:**
- Pre-trained DistilBERT embeddings
- Transfer learning from 100M+ texts
- Cross-attention between text and time series
- State-of-the-art text understanding

**Architecture:**
```
Review Text → DistilBERT → Text Features (768-d)
                                ↓
                          Project to 256-d
                                ↓
Time Series → Transformer → TS Features (256-d)
                                ↓
                    Cross-Attention Layer
                                ↓
                    Fusion Network → Prediction
```

**Pros:**
- **Best accuracy** - BERT understands semantic meaning
- **Transfer learning** - pre-trained on billions of words
- **Captures nuances** - sentiment, context, product descriptions
- **Cross-attention** - learns relationships between text and trends

**Cons:**
- Slower training (4-6 hours)
- Requires GPU with 16GB+ RAM
- Larger model size

**Use when:** You want best performance and have GPU resources

---

## Performance Comparison (Expected)

| Model | Accuracy | Training Time | GPU Memory | Model Size |
|-------|----------|---------------|------------|------------|
| Basic | ~70-75% | 1-2 hrs | 4GB | 50MB |
| Word2Vec | ~75-80% | 2-3 hrs | 8GB | 200MB |
| **BERT** | **~82-88%** | **4-6 hrs** | **12-16GB** | **500MB** |

---

## Key Improvements in BERT Model

### 1. **Better Text Understanding**
- Basic: Ignores "great product, highly recommend"
- Word2Vec: Captures word frequency
- **BERT: Understands sentiment, context, sarcasm**

### 2. **Cross-Attention Mechanism**
Learns patterns like:
- "Viral on TikTok" + sudden sales spike
- "Best seller" tags + sustained growth
- Product descriptions + category trends

### 3. **Training Optimizations**
- Differential learning rates (BERT: 2e-5, other layers: 2e-4)
- Gradient accumulation for larger effective batch size
- OneCycleLR scheduler with warmup
- Gradient clipping for stability

### 4. **Feature Engineering**
- Uses top 5 most helpful reviews per product
- Combines title + review text
- Truncates intelligently (128 tokens)

---

## How to Train

### Option 1: Basic Model
```bash
sbatch train_full_pipeline.slurm
```

### Option 2: Word2Vec Model
```bash
sbatch train_text_model.slurm
```

### Option 3: BERT Model (Recommended)
```bash
sbatch train_bert_model.slurm
```

All models wait for the weekly panel dataset to be ready before training.

---

## Output Files

Each model saves to `/home1/yliu0158/amazon2023/amazon23/training_output/`:

- `transformer_model/` - Basic model
- `transformer_with_text/` - Word2Vec model
- `transformer_bert/` - BERT model

Each directory contains:
- `model*.pt` - Trained weights
- `best*.json` - Best validation metrics
- `preds*.csv` - Validation predictions (optional)

---

## Recommendations

1. **Start with BERT** - If you have GPU, use the BERT model for best results
2. **Use ensemble** - Combine predictions from multiple models
3. **Monitor validation** - Check logs for overfitting
4. **Tune hyperparameters** - Adjust learning rate, batch size based on results

---

## Next Steps After Training

1. **Evaluate models** - Compare validation accuracy
2. **Analyze predictions** - Which products are predicted as hot sellers?
3. **Feature importance** - Which signals matter most?
4. **Ensemble** - Combine with AutoTS forecasts
5. **Deploy** - Use best model for production predictions