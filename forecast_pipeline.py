import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from forecast_ops.autots_model import train_autots
from forecast_ops.data_utils import ForecastConfig, build_schema_report, load_and_prepare_panel, write_json
from forecast_ops.ensemble import blend_predictions
from forecast_ops.metrics import compute_metrics, compute_per_series_metrics
from forecast_ops.plots import (
    plot_forecast_curves,
    plot_residual_hist,
    plot_residual_scatter,
    plot_smape_violin,
    plot_tft_importance,
)
from forecast_ops.tft_model import train_tft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFT and AutoTS ensemble for 12-week forecasts")
    parser.add_argument("--dataset", required=True, help="Path to input CSV")
    parser.add_argument("--series-col", required=True, help="Column name for series identifier")
    parser.add_argument("--time-col", required=True, help="Timestamp column name")
    parser.add_argument("--target-col", default=None, help="Optional target column (if absent, counts are aggregated)")
    parser.add_argument("--known-cols", default="", help="Comma-separated known real feature columns")
    parser.add_argument("--obs-cols", default="", help="Comma-separated observed real feature columns")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--frequency", default="W")
    parser.add_argument("--max-encoder-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--disable-tft", action="store_true", help="Skip training the TFT model")
    parser.add_argument("--disable-autots", action="store_true", help="Skip training the AutoTS model")
    parser.add_argument("--outdir", default="out", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    known_cols: List[str] = [c.strip() for c in args.known_cols.split(",") if c.strip()]
    obs_cols: List[str] = [c.strip() for c in args.obs_cols.split(",") if c.strip()]

    config = ForecastConfig(
        horizon=args.horizon,
        val_size=args.val_size,
        frequency=args.frequency,
        max_encoder_len=args.max_encoder_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        num_workers=args.num_workers,
        seed=args.seed,
        enable_tft=not args.disable_tft,
        enable_autots=not args.disable_autots,
    )

    if not (config.enable_tft or config.enable_autots):
        raise ValueError("At least one of TFT or AutoTS must be enabled")

    min_series_len = config.max_encoder_len + config.horizon
    panel, info = load_and_prepare_panel(
        Path(args.dataset),
        series_col=args.series_col,
        time_col=args.time_col,
        target_col=args.target_col,
        freq=config.frequency,
        known_reals=known_cols,
        observed_reals=obs_cols,
        min_length=min_series_len,
    )

    panel_path = outdir / "panel.parquet"
    panel.to_parquet(panel_path, index=False)

    schema_report = build_schema_report(panel)
    schema_report.update(info)
    write_json(schema_report, outdir / "data_schema.json")

    tft_val_df = None
    tft_model_path = None
    importance_df = None
    if config.enable_tft:
        tft_val_df, tft_model_path, importance_df = train_tft(panel, config, known_cols, obs_cols, outdir)
        tft_val_df.to_csv(outdir / "pred_tft_val.csv", index=False)

    autots_val_df = None
    autots_model_path = None
    template_path = None
    if config.enable_autots:
        autots_val_df, autots_model_path, template_path = train_autots(panel, config, outdir)
        autots_val_df.to_csv(outdir / "pred_autots_val.csv", index=False)

    weights = {"global": {}, "per_series": {}}
    if config.enable_tft and config.enable_autots:
        blend_df, weights = blend_predictions(tft_val_df, autots_val_df)
        blend_df.to_csv(outdir / "pred_blend_val.csv", index=False)
    elif config.enable_tft:
        blend_df = tft_val_df.copy()
        blend_df["yhat_blend"] = blend_df["yhat_tft"]
        blend_df["yhat_blend_series"] = blend_df["yhat_tft"]
        weights = {
            "global": {"tft": 1.0},
            "per_series": {sid: {"tft": 1.0} for sid in blend_df["series_id"].unique()},
        }
        blend_df.to_csv(outdir / "pred_blend_val.csv", index=False)
    elif config.enable_autots:
        blend_df = autots_val_df.copy()
        blend_df["yhat_blend"] = blend_df["yhat_autots"]
        blend_df["yhat_blend_series"] = blend_df["yhat_autots"]
        weights = {
            "global": {"autots": 1.0},
            "per_series": {sid: {"autots": 1.0} for sid in blend_df["series_id"].unique()},
        }
        blend_df.to_csv(outdir / "pred_blend_val.csv", index=False)
    else:
        raise RuntimeError("No model outputs were generated")

    global_metrics = {}
    if tft_val_df is not None:
        global_metrics["tft"] = compute_metrics(tft_val_df, "yhat_tft")
    if autots_val_df is not None:
        global_metrics["autots"] = compute_metrics(autots_val_df, "yhat_autots")
    if "yhat_blend" in blend_df:
        global_metrics["blend"] = compute_metrics(blend_df, "yhat_blend")
    if "yhat_blend_series" in blend_df:
        global_metrics["blend_series"] = compute_metrics(blend_df, "yhat_blend_series")

    metric_columns = {}
    if "yhat_tft" in blend_df.columns and tft_val_df is not None:
        metric_columns["tft"] = "yhat_tft"
    if "yhat_autots" in blend_df.columns and autots_val_df is not None:
        metric_columns["autots"] = "yhat_autots"
    if "yhat_blend" in blend_df.columns:
        metric_columns["blend"] = "yhat_blend"
    if "yhat_blend_series" in blend_df.columns:
        metric_columns["blend_series"] = "yhat_blend_series"

    per_series_metrics = (
        compute_per_series_metrics(blend_df, metric_columns) if metric_columns else {}
    )

    metrics_payload = {
        "global": global_metrics,
        "per_series": per_series_metrics,
        "weights": weights,
    }
    write_json(metrics_payload, outdir / "metrics.json")

    # Plots
    top_series = blend_df.groupby("series_id")["y"].sum().sort_values(ascending=False).head(4).index.tolist()
    if "yhat_blend" in blend_df.columns:
        plot_residual_hist(blend_df, "yhat_blend", outdir / "plots" / "residual_hist_blend.png")
        plot_residual_scatter(blend_df, "yhat_blend", outdir / "plots" / "residual_scatter_blend.png")

    model_plot_cols = {}
    if config.enable_tft and "yhat_tft" in blend_df.columns:
        model_plot_cols["TFT"] = "yhat_tft"
    if config.enable_autots and "yhat_autots" in blend_df.columns:
        model_plot_cols["AutoTS"] = "yhat_autots"
    if "yhat_blend" in blend_df.columns:
        model_plot_cols.setdefault("Blend", "yhat_blend")

    if model_plot_cols:
        plot_forecast_curves(
            blend_df,
            top_series,
            model_plot_cols,
            outdir / "plots" / "forecast_vs_actual.png",
        )

    if metric_columns:
        plot_smape_violin(per_series_metrics, outdir / "plots" / "smape_violin.png")
    if importance_df is not None:
        plot_tft_importance(importance_df, outdir / "plots" / "tft_importance.png")

    artifacts = {
        "panel": str(panel_path),
    }
    if tft_model_path is not None:
        artifacts["tft_model"] = str(tft_model_path)
    if autots_model_path is not None:
        artifacts["autots_model"] = str(autots_model_path)
    if template_path is not None:
        artifacts["autots_template"] = str(template_path)
    write_json(artifacts, outdir / "artifacts_index.json")

    print(json.dumps({"status": "completed", "metrics": global_metrics}, indent=2))


if __name__ == "__main__":
    main()
