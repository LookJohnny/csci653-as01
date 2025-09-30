"""
Ensemble Prediction System

Combines multiple models:
1. Ultimate GNN Model
2. Enhanced BERT Model
3. AutoTS Time Series Model
4. XGBoost (gradient boosting)
5. LightGBM (fast gradient boosting)

Methods:
- Simple averaging
- Weighted averaging (by validation performance)
- Stacking (meta-learner)
- Voting (for classification)
"""
import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

import torch
from torch import nn
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb
import xgboost as xgb

class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models
    """
    def __init__(self, config_path):
        self.config = json.load(open(config_path))
        self.models = {}
        self.weights = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_deep_model(self, name, model_path, model_class):
        """Load PyTorch model"""
        print(f"Loading {name}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        model = model_class(**checkpoint.get('model_config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.models[name] = model
        self.weights[name] = checkpoint.get('val_acc', 0.85)

        print(f"  ✓ {name} loaded (val_acc: {self.weights[name]:.4f})")

    def load_tree_model(self, name, model_path):
        """Load tree-based model"""
        print(f"Loading {name}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.models[name] = model
        # Assume same val_acc as deep models if not specified
        self.weights[name] = 0.85

        print(f"  ✓ {name} loaded")

    def predict_deep_model(self, model_name, x_ts, input_ids, attn_mask, pids=None, neighbors=None, n_neighbors=None):
        """Get predictions from deep learning model"""
        model = self.models[model_name]

        with torch.no_grad():
            x_ts = x_ts.to(self.device)
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            if pids is not None:
                # GNN model
                pids = pids.to(self.device)
                neighbors = neighbors.to(self.device)
                n_neighbors = n_neighbors.to(self.device)
                logits, _, _ = model(x_ts, input_ids, attn_mask, pids, neighbors, n_neighbors)
            else:
                # Regular model
                logits = model(x_ts, input_ids, attn_mask)

            probs = torch.sigmoid(logits).cpu().numpy()

        return probs.flatten()

    def predict_tree_model(self, model_name, features):
        """Get predictions from tree model"""
        model = self.models[model_name]

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[:, 1]
        else:
            probs = model.predict(features)

        return probs

    def simple_average(self, predictions):
        """Simple average of all predictions"""
        return np.mean(predictions, axis=0)

    def weighted_average(self, predictions, model_names):
        """Weighted average based on validation performance"""
        weights = np.array([self.weights[name] for name in model_names])
        weights = weights / weights.sum()  # Normalize

        weighted_preds = np.average(predictions, axis=0, weights=weights)
        return weighted_preds

    def voting(self, predictions, threshold=0.5):
        """Majority voting (for hard predictions)"""
        binary_preds = (np.array(predictions) > threshold).astype(int)
        votes = np.sum(binary_preds, axis=0)
        return (votes > len(predictions) / 2).astype(int)

    def rank_average(self, predictions):
        """Average of rank-transformed predictions"""
        from scipy.stats import rankdata

        ranks = np.array([rankdata(pred) for pred in predictions])
        avg_ranks = np.mean(ranks, axis=0)

        # Convert back to [0, 1]
        return (avg_ranks - avg_ranks.min()) / (avg_ranks.max() - avg_ranks.min())

class StackingEnsemble:
    """
    Stacking ensemble with meta-learner

    Level 0: Base models (GNN, BERT, XGBoost, etc.)
    Level 1: Meta-learner (LogisticRegression, Neural Network)
    """
    def __init__(self, base_models, meta_learner='logreg'):
        self.base_models = base_models

        if meta_learner == 'logreg':
            self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        elif meta_learner == 'xgb':
            self.meta_model = xgb.XGBClassifier(n_estimators=100, max_depth=3)
        elif meta_learner == 'nn':
            self.meta_model = SimpleMetaNet(len(base_models))
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner}")

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train stacking ensemble

        1. Get base model predictions on validation set
        2. Train meta-learner on these predictions
        """
        print("Training stacking ensemble...")

        # Get base predictions
        base_preds_train = []
        base_preds_val = []

        for name, model in self.base_models.items():
            print(f"  Getting predictions from {name}...")

            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]

            base_preds_train.append(train_pred)
            base_preds_val.append(val_pred)

        # Stack predictions as features
        X_meta_train = np.column_stack(base_preds_train)
        X_meta_val = np.column_stack(base_preds_val)

        # Train meta-learner
        print("  Training meta-learner...")
        self.meta_model.fit(X_meta_train, y_train)

        # Evaluate
        val_pred = self.meta_model.predict_proba(X_meta_val)[:, 1]
        val_acc = accuracy_score(y_val, val_pred > 0.5)
        val_auc = roc_auc_score(y_val, val_pred)

        print(f"  Stacking Ensemble - Val Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        return self

    def predict(self, X_test):
        """Predict using stacking ensemble"""
        base_preds = []

        for name, model in self.base_models.items():
            pred = model.predict_proba(X_test)[:, 1]
            base_preds.append(pred)

        X_meta = np.column_stack(base_preds)
        return self.meta_model.predict_proba(X_meta)[:, 1]

class SimpleMetaNet(nn.Module):
    """Simple neural network meta-learner"""
    def __init__(self, n_models):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_models, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model"""
    print("Training XGBoost...")

    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'early_stopping_rounds': 50
        }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate
    val_pred = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_pred > 0.5)
    val_auc = roc_auc_score(y_val, val_pred)

    print(f"XGBoost - Val Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    return model

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM model"""
    print("Training LightGBM...")

    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'random_state': 42,
            'verbose': -1
        }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # Evaluate
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred > 0.5)
    val_auc = roc_auc_score(y_val, val_pred)

    print(f"LightGBM - Val Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    return model

def create_ensemble_features(df):
    """Create features for tree-based models"""
    features = []
    feature_names = []

    # Time series features (already in df)
    ts_features = [
        'reviews', 'rating_mean', 'rating_std',
        'helpful_sum', 'helpful_mean', 'pct_with_helpful',
        'engagement_score', 'avg_text_length', 'pct_long_reviews',
        'pct_with_images', 'sentiment_mean', 'sentiment_std',
        'verified_ratio', 'quality_score',
        'rev_prev4', 'review_momentum', 'helpful_growth'
    ]

    for feat in ts_features:
        if feat in df.columns:
            features.append(df[feat].values)
            feature_names.append(feat)

    # Interaction features
    if 'sentiment_mean' in df.columns and 'review_momentum' in df.columns:
        features.append((df['sentiment_mean'] * df['review_momentum']).values)
        feature_names.append('sentiment_x_momentum')

    if 'quality_score' in df.columns and 'engagement_score' in df.columns:
        features.append((df['quality_score'] * df['engagement_score']).values)
        feature_names.append('quality_x_engagement')

    # Rolling features
    if 'reviews' in df.columns:
        df_sorted = df.sort_values('week_start')
        features.append(df_sorted.groupby('parent_asin')['reviews'].transform(
            lambda x: x.rolling(8, min_periods=1).mean()
        ).values)
        feature_names.append('reviews_roll8_mean')

    X = np.column_stack(features)
    return X, feature_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Panel dataset CSV')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--train_tree_models', action='store_true', help='Train XGBoost and LightGBM')
    parser.add_argument('--build_ensemble', action='store_true', help='Build ensemble from all models')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data, parse_dates=['week_start'])

    # Split
    last_week = df['week_start'].max()
    cutoff = last_week - pd.Timedelta(weeks=26)
    train_df = df[df['week_start'] <= cutoff].copy()
    val_df = df[df['week_start'] > cutoff].copy()

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    if args.train_tree_models:
        # Create features for tree models
        print("\nCreating features for tree models...")
        X_train, feature_names = create_ensemble_features(train_df)
        X_val, _ = create_ensemble_features(val_df)

        y_train = train_df['label_top5'].values
        y_val = val_df['label_top5'].values

        print(f"Feature matrix: {X_train.shape}")
        print(f"Features: {feature_names}")

        # Train XGBoost
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

        # Save
        xgb_path = out_dir / 'xgboost_model.pkl'
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        print(f"Saved XGBoost to {xgb_path}")

        # Train LightGBM
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

        # Save
        lgb_path = out_dir / 'lightgbm_model.txt'
        lgb_model.save_model(str(lgb_path))
        print(f"Saved LightGBM to {lgb_path}")

        # Feature importance
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        xgb.plot_importance(xgb_model, ax=ax1, max_num_features=20)
        ax1.set_title('XGBoost Feature Importance')

        lgb.plot_importance(lgb_model, ax=ax2, max_num_features=20)
        ax2.set_title('LightGBM Feature Importance')

        plt.tight_layout()
        plt.savefig(out_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot")

    if args.build_ensemble:
        print("\n" + "="*60)
        print("BUILDING ENSEMBLE")
        print("="*60)

        # TODO: Load all trained models and combine predictions
        # This requires all models to be trained first

        print("Ensemble building requires all models to be trained.")
        print("Train models first, then run with --build_ensemble")

    print("\nDone!")

if __name__ == '__main__':
    main()