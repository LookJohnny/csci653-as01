from pathlib import Path
from typing import Tuple

import pandas as pd

from .data_utils import ForecastConfig


def train_autots(
    panel: pd.DataFrame,
    config: ForecastConfig,
    outdir: Path,
) -> Tuple[pd.DataFrame, Path, Path]:
    from autots import AutoTS

    max_idx = panel.groupby("series_id")["time_idx"].max().to_dict()
    panel = panel.copy()
    panel["max_time_idx"] = panel["series_id"].map(max_idx)

    train_df = panel[panel["time_idx"] <= panel["max_time_idx"] - config.val_size].copy()
    if train_df.empty:
        raise RuntimeError("AutoTS training data is empty after split; reduce val_size")

    model = AutoTS(
        forecast_length=config.horizon,
        frequency=config.frequency,
        ensemble="simple",
        max_generations=2,
        num_validations=1,
        validation_method="backwards",
        model_list="fast",
        verbose=0,
    )

    model = model.fit(
        train_df,
        date_col="time",
        value_col="y",
        id_col="series_id",
    )
    forecast = model.predict()
    forecast_df = forecast.forecast
    forecast_df.index.name = "time"

    val_times = (
        panel[panel["time_idx"] > panel["max_time_idx"] - config.val_size]["time"].drop_duplicates().sort_values()
    )
    val_times = list(val_times)
    forecast_df = forecast_df.loc[forecast_df.index.isin(val_times)]
    if forecast_df.empty:
        raise RuntimeError("AutoTS did not return forecast for validation horizon")

    pred_long = (
        forecast_df.reset_index()
        .melt(id_vars="time", var_name="series_id", value_name="yhat_autots")
        .dropna(subset=["yhat_autots"])
    )

    merged = panel.merge(pred_long, on=["series_id", "time"], how="inner")
    merged = merged[merged["time_idx"] > merged["max_time_idx"] - config.val_size].copy()
    merged = merged.drop(columns=["max_time_idx"])

    model_path = outdir / "autots_model.zip"
    template_path = outdir / "autots_template.csv"
    model.export_template(template_path)
    model.save(model_path)

    return merged[["series_id", "time", "y", "yhat_autots"]], model_path, template_path
