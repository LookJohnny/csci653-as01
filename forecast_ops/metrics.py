from typing import Dict

import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    return {
        "smape": smape(df["y"].values, df[pred_col].values),
        "mae": mae(df["y"].values, df[pred_col].values),
        "rmse": rmse(df["y"].values, df[pred_col].values),
    }


def compute_per_series_metrics(df: pd.DataFrame, pred_cols: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    per_series: Dict[str, Dict[str, Dict[str, float]]] = {}
    for series_id, group in df.groupby("series_id"):
        series_metrics: Dict[str, Dict[str, float]] = {}
        for name, col in pred_cols.items():
            series_metrics[name] = compute_metrics(group, col)
        per_series[series_id] = series_metrics
    return per_series
