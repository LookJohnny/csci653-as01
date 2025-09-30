from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import compute_metrics, smape


def plot_residual_hist(df: pd.DataFrame, pred_col: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    residual = df["y"] - df[pred_col]
    plt.figure(figsize=(8, 5))
    sns.histplot(residual, kde=True, bins=30)
    plt.title(f"Residual distribution ({pred_col})")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_residual_scatter(df: pd.DataFrame, pred_col: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    residual = df["y"] - df[pred_col]
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["y"], y=residual)
    plt.title(f"Actual vs Residual ({pred_col})")
    plt.xlabel("Actual y")
    plt.ylabel("Residual")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_forecast_curves(df: pd.DataFrame, series_ids: Iterable[str], model_cols: Dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series_ids = list(series_ids)
    n = len(series_ids)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(6 * cols, 4 * rows))
    for idx, series_id in enumerate(series_ids, start=1):
        subset = df[df["series_id"] == series_id].sort_values("time")
        if subset.empty:
            continue
        ax = plt.subplot(rows, cols, idx)
        ax.plot(subset["time"], subset["y"], label="actual", color="black", linewidth=2)
        for label, col in model_cols.items():
            if col in subset.columns:
                ax.plot(subset["time"], subset[col], label=label)
        ax.set_title(f"series_id={series_id}")
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_smape_violin(per_series: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for series_id, models in per_series.items():
        for model_name, metrics in models.items():
            records.append({"series_id": series_id, "model": model_name, "smape": metrics["smape"]})
    data = pd.DataFrame(records)
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=data, x="model", y="smape", cut=0)
    plt.title("Per-series sMAPE distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_tft_importance(importance_df: pd.DataFrame, out_path: Path) -> None:
    if importance_df is None or importance_df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="variable", hue="stage", orient="h")
    plt.title("TFT variable importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
