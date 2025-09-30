from typing import Dict, Tuple

import pandas as pd

from .metrics import smape


def blend_predictions(tft_df: pd.DataFrame, autots_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    merged = pd.merge(tft_df, autots_df, on=["series_id", "time", "y"], how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping validation rows between TFT and AutoTS predictions")

    global_smape_tft = smape(merged["y"], merged["yhat_tft"])
    global_smape_autots = smape(merged["y"], merged["yhat_autots"])

    weight_tft = 1.0 / max(global_smape_tft, 1e-6)
    weight_autots = 1.0 / max(global_smape_autots, 1e-6)
    denom = weight_tft + weight_autots
    merged["yhat_blend"] = (merged["yhat_tft"] * weight_tft + merged["yhat_autots"] * weight_autots) / denom

    per_series_weights: Dict[str, Dict[str, float]] = {}
    for series_id, group in merged.groupby("series_id"):
        s_tft = smape(group["y"], group["yhat_tft"])
        s_autots = smape(group["y"], group["yhat_autots"])
        w_tft = 1.0 / max(s_tft, 1e-6)
        w_autots = 1.0 / max(s_autots, 1e-6)
        denom_series = w_tft + w_autots
        merged.loc[group.index, "yhat_blend_series"] = (
            group["yhat_tft"] * w_tft + group["yhat_autots"] * w_autots
        ) / denom_series
        per_series_weights[series_id] = {
            "tft": w_tft / denom_series,
            "autots": w_autots / denom_series,
        }

    weights = {
        "global": {"tft": weight_tft / denom, "autots": weight_autots / denom},
        "per_series": per_series_weights,
    }

    return merged, weights
