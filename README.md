# DarkHorse 🐴

**AI-Driven Dynamic Load Balancing for High-Throughput Distributed Systems**

An advanced demand forecasting and load balancing system for predicting product demand spikes in e-commerce platforms using time series analysis and ensemble machine learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Results & Metrics](#results--metrics)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

DarkHorse is designed to predict demand surges for products in high-throughput distributed systems. By analyzing historical review patterns, it forecasts which products will become "hot sellers" (top 5% growth) in the next 12 weeks, enabling proactive resource allocation and load balancing.

### Key Use Cases
- **E-commerce Inventory Management**: Predict product demand spikes for optimized stocking
- **Dynamic Resource Allocation**: Balance computational loads across distributed systems
- **Hot-Seller Detection**: Identify trending products early for marketing and pricing strategies
- **Supply Chain Optimization**: Proactive planning based on demand forecasts

---

## ✨ Features

- **Multi-Model Ensemble**: Combines Temporal Fusion Transformer (TFT) and AutoTS for robust predictions
- **Scalable Data Pipeline**: Process millions of reviews with configurable sharding
- **Weekly Aggregation**: Converts raw review data to weekly panel format with engineered features
- **Interpretability**: Variable importance analysis and visualization tools
- **Flexible Configuration**: Extensive CLI options for experimentation
- **Production Ready**: Modular design with forecast operations library

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Raw Amazon Reviews                         │
│              (McAuley-Lab Dataset 2023)                       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Data Cleaning & Preprocessing                    │
│  • 33 Product Categories  • Hash-based Sharding               │
│  • Timestamp Normalization  • Category Mapping                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                Weekly Panel Construction                      │
│  • 4-week rolling reviews  • 12-week future growth            │
│  • Verified purchase ratio  • Helpful vote aggregation        │
│  • Rating statistics  • Top-5% label assignment               │
└────────────────────────┬─────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌──────────────────────┐    ┌──────────────────────┐
│  Transformer Model   │    │  Ensemble Forecast   │
│  • Sequence Learning │    │  • TFT Model         │
│  • Binary Classifier │    │  • AutoTS Model      │
│  • Hot-Seller Pred   │    │  • Weighted Blend    │
└──────────────────────┘    └──────────────────────┘
            │                         │
            └────────────┬────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Load Balancing & Allocation                  │
│  • Demand Spike Prediction  • Resource Optimization           │
│  • Per-Series Metrics  • Visualization & Monitoring           │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- 16GB+ RAM (for large datasets)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/darkHorse.git
cd darkHorse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=10.0.0
datasets>=2.14.0
transformers>=4.30.0
pytorch-forecasting>=1.0.0
pytorch-lightning>=2.0.0
autots>=0.6.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🚀 Quick Start

### Interactive Mode

```bash
python main.py
```

### Command-Line Mode

```bash
# Generate ASIN mapping
python main.py --pipeline mapping

# Build weekly dataset
python main.py --pipeline weekly

# Train models
python main.py --pipeline transformer
python main.py --pipeline forecast
```

### Complete Pipeline Example

```bash
# 1. Generate category mapping
python generate_asin_mapping.py --input cleaned_beauty_reviews.csv

# 2. Build weekly panel
python build_weekly_dataset.py \
  --input cleaned_beauty_reviews.csv \
  --out weekly_beauty_panel.csv \
  --top_quantile 0.95

# 3. Train transformer model
python train_transformer.py \
  --data weekly_beauty_panel.csv \
  --out transformer_output \
  --epochs 20 \
  --batch_size 256

# 4. Run forecast ensemble
python forecast_pipeline.py \
  --dataset weekly_beauty_panel.csv \
  --series-col parent_asin \
  --time-col week_start \
  --horizon 12 \
  --outdir forecast_output
```

---

## 🔧 Pipeline Components

### 1. Data Cleaning (`dataCleaning.py`)
Processes raw Amazon review data from the McAuley-Lab 2023 dataset.

**Features:**
- Loads 33 product categories
- Hash-based sharding (64 shards default)
- Timestamp normalization
- Category mapping via `asin2category.json`

**Configuration:**
- `OUT`: Output directory path
- `S`: Number of shards (default: 64)

### 2. Weekly Panel Builder (`build_weekly_dataset.py`)
Aggregates review-level data into weekly time series.

**Arguments:**
```bash
--input         Input CSV with review data
--out           Output CSV path
--top_quantile  Threshold for hot-seller labels (default: 0.95)
--min_reviews   Minimum rolling reviews to keep (default: 1)
```

**Output Columns:**
- `parent_asin`: Product identifier
- `week_start`: Monday of the week
- `reviews`: Review count this week
- `helpful_sum`: Total helpful votes
- `verified_ratio`: Proportion of verified purchases
- `rating_mean`: Average rating
- `rev_prev4`: Rolling 4-week review count
- `rev_next12`: Future 12-week review count
- `growth_score`: Future/past review ratio
- `label_top5`: Binary hot-seller label (1 if top 5%)

### 3. Transformer Model (`train_transformer.py`)
Time series transformer for binary classification.

**Architecture:**
- Input projection: 4 features → d_model
- Transformer encoder: 3 layers, 4 heads
- Output: Binary logit (hot-seller probability)

**Arguments:**
```bash
--data         Input weekly panel CSV
--out          Output directory
--seq_len      Sequence length (default: 32 weeks)
--batch_size   Batch size (default: 256)
--epochs       Training epochs (default: 10)
--lr           Learning rate (default: 1e-3)
```

### 4. Forecast Pipeline (`forecast_pipeline.py`)
Ensemble forecasting with TFT and AutoTS.

**Models:**
- **TFT (Temporal Fusion Transformer)**: Deep learning model with attention mechanisms
- **AutoTS**: Automated time series model selection
- **Ensemble**: Weighted blend optimized per-series

**Arguments:**
```bash
--dataset          Input panel CSV
--series-col       Series identifier column
--time-col         Timestamp column
--horizon          Forecast horizon (default: 12)
--val-size         Validation size (default: 12)
--disable-tft      Skip TFT model
--disable-autots   Skip AutoTS model
--outdir           Output directory
```

**Outputs:**
- `pred_tft_val.csv`: TFT predictions
- `pred_autots_val.csv`: AutoTS predictions
- `pred_blend_val.csv`: Ensemble predictions
- `metrics.json`: Global and per-series metrics
- `plots/`: Visualization outputs

---

## ⚙️ Configuration

### Environment Variables
Create a `.env` file:
```bash
DATA_DIR=/path/to/data
OUTPUT_DIR=/path/to/outputs
CUDA_VISIBLE_DEVICES=0
```

### Model Hyperparameters
Key parameters in `forecast_pipeline.py`:
- `max_encoder_len`: Historical lookback window (default: 64)
- `hidden_size`: Model hidden dimensions (default: 64)
- `attention_heads`: Number of attention heads (default: 4)
- `dropout`: Dropout rate (default: 0.1)

---

## 🤖 Model Details

### Temporal Fusion Transformer (TFT)
- **Architecture**: Multi-head attention with variable selection
- **Features**:
  - Known reals: Time-varying features known in advance
  - Observed reals: Features only known historically
- **Training**: AdamW optimizer with cosine annealing

### AutoTS
- **Method**: Automated model selection from 30+ algorithms
- **Ensemble**: Genetic algorithm for optimal combination
- **Speed**: Optimized for fast inference

### Ensemble Strategy
- **Global Weights**: Average optimal weights across all series
- **Per-Series Weights**: Customized weights per product
- **Metric**: Minimize SMAPE (Symmetric Mean Absolute Percentage Error)

---

## 📊 Results & Metrics

### Evaluation Metrics
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Accuracy**: Binary classification accuracy (for transformer model)

### Output Files
```
forecast_output/
├── pred_tft_val.csv           # TFT predictions
├── pred_autots_val.csv        # AutoTS predictions
├── pred_blend_val.csv         # Ensemble predictions
├── metrics.json               # Performance metrics
├── data_schema.json           # Dataset metadata
├── artifacts_index.json       # Model paths
└── plots/
    ├── forecast_vs_actual.png # Time series comparison
    ├── residual_hist_blend.png
    ├── residual_scatter_blend.png
    ├── smape_violin.png
    └── tft_importance.png     # Feature importance
```

---

## 🧪 Testing

```bash
# Run unit tests (TODO)
pytest tests/

# Test specific module
pytest tests/test_build_weekly.py
```

---

## 📝 Examples

### Predict Top Products for Next Quarter
```python
import pandas as pd

# Load predictions
preds = pd.read_csv("forecast_output/pred_blend_val.csv")

# Get top 10 predicted hot sellers
top_products = preds.groupby("series_id")["yhat_blend"].sum() \
    .sort_values(ascending=False).head(10)

print("Top 10 predicted hot sellers:")
print(top_products)
```

### Analyze Feature Importance
```python
import json

with open("forecast_output/artifacts_index.json") as f:
    artifacts = json.load(f)

# Load TFT model and extract importance
# (See forecast_ops/plots.py for implementation)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **McAuley-Lab** for the Amazon Reviews 2023 dataset
- **PyTorch Forecasting** for the TFT implementation
- **AutoTS** for automated time series modeling

---

## 📧 Contact

For questions or support, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).

---

## 🔮 Roadmap

- [ ] Implement actual load balancing service
- [ ] Add REST API endpoints
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] Real-time inference pipeline
- [ ] A/B testing framework
- [ ] Enhanced visualization dashboard

---

**Made with ❤️ by the DarkHorse Team**