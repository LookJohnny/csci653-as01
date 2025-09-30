#!/bin/bash
#SBATCH --job-name=04_xgboost
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/04_xgboost_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/04_xgboost_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --partition=main

# ============================================================================
# STAGE 4A: XGBOOST MODEL
# ============================================================================
# Gradient boosting on enhanced tabular features
# Expected Accuracy: 88-90%
# Training Time: ~2-3 hours (32 CPUs)
# ============================================================================

set -e
set -u

export OMP_NUM_THREADS=32

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "=========================================="
echo "STAGE 4A: XGBOOST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: 32"
echo "=========================================="

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

pip install -q xgboost scikit-learn

python <<EOF
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("Loading data...")
df = pd.read_csv("${OUT_DIR}/panel_enhanced.csv")

# Prepare features
feature_cols = [c for c in df.columns if c not in ['parent_asin', 'week_start', 'label_top5']]
X = df[feature_cols].fillna(0)
y = df['label_top5']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Features: {len(feature_cols)}")
print(f"Positive class: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")

print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=32,
    random_state=42,
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"XGBoost Test Accuracy: {accuracy*100:.2f}%")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("${OUT_DIR}/models", exist_ok=True)
joblib.dump(model, "${OUT_DIR}/models/xgboost_model.pkl")
print("\n✓ Model saved to ${OUT_DIR}/models/xgboost_model.pkl")
EOF

echo "✓ XGBoost training complete!"