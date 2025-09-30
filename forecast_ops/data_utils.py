import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastConfig:
    horizon: int = 12
    val_size: int = 12
    frequency: str = "W"
    max_encoder_len: int = 64
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    attention_heads: int = 4
    dropout: float = 0.1
    num_workers: int = 0
    seed: int = 2025
    enable_tft: bool = True
    enable_autots: bool = True


FREQ_TO_OFFSET = {
    "D": pd.Timedelta(days=1),
    "W": pd.Timedelta(weeks=1),
    "M": pd.Timedelta(days=30),  # coarse fallback
}


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=int)
    if series.dtype == bool:
        return series.astype(int)
    if np.issubdtype(series.dtype, np.number):
        return (series.fillna(0) != 0).astype(int)
    return series.astype(str).str.lower().isin({"true", "t", "1", "yes"}).astype(int)


def _regularize_frequency(df: pd.DataFrame, freq: str, numeric_cols: Iterable[str], categorical_cols: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols)
    for series_id, group in df.groupby("series_id"):
        group = group.sort_values("time").set_index("time")
        if group.empty:
            continue
        full_index = pd.date_range(group.index.min(), group.index.max(), freq=freq)
        expanded = group.reindex(full_index)
        expanded["series_id"] = series_id
        frames.append(expanded.reset_index().rename(columns={"index": "time"}))
    if not frames:
        return df.copy()
    result = pd.concat(frames, ignore_index=True)
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)
    for col in categorical_cols:
        if col in result.columns:
            result[col] = result.groupby("series_id")[col].transform(lambda s: s.ffill().bfill())
            result[col] = result[col].fillna("Unknown")
    return result


def load_and_prepare_panel(
    csv_path: Path,
    series_col: str,
    time_col: str,
    target_col: Optional[str],
    freq: str,
    known_reals: Optional[List[str]] = None,
    observed_reals: Optional[List[str]] = None,
    min_length: int = 24,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    known_reals = known_reals or []
    observed_reals = observed_reals or []
    df = pd.read_csv(csv_path, low_memory=False)
    if series_col not in df.columns:
        raise ValueError(f"series_col '{series_col}' not found in {csv_path}")
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in {csv_path}")

    df = df.rename(columns={series_col: "series_id", time_col: "time"})
    df["series_id"] = df["series_id"].astype(str)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "series_id"]).copy()

    derived_cols: List[str] = []
    if target_col and target_col in df.columns:
        df = df.rename(columns={target_col: "y"})
        df = df[["series_id", "time", "y"] + [col for col in known_reals + observed_reals if col in df.columns]]
    else:
        df = df[[col for col in df.columns if col in {"series_id", "time", "verified_purchase", "helpful_vote", "rating", "main_category"}]]
        df["default_target"] = 1.0
        agg_dict = {"y": ("default_target", "sum")}
        if "helpful_vote" in df.columns:
            df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0.0)
            agg_dict["helpful_sum"] = ("helpful_vote", "sum")
            derived_cols.append("helpful_sum")
        if "verified_purchase" in df.columns:
            df["verified_flag"] = _coerce_bool(df["verified_purchase"])
            agg_dict["verified_count"] = ("verified_flag", "sum")
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            agg_dict["rating_mean"] = ("rating", "mean")
            derived_cols.append("rating_mean")
        if "main_category" in df.columns:
            agg_dict["main_category"] = ("main_category", "first")
            derived_cols.append("main_category")

        grouped = (
            df.groupby([
                "series_id",
                pd.Grouper(key="time", freq=freq, label="left", origin="epoch"),
            ], dropna=True)
            .agg(**agg_dict)
            .reset_index()
        )
        grouped = grouped.rename(columns={"time": "time"})
        if "verified_count" in grouped.columns:
            grouped["verified_ratio"] = np.where(grouped["y"] > 0, grouped["verified_count"] / grouped["y"], 0.0)
            grouped = grouped.drop(columns=["verified_count"])
            derived_cols.append("verified_ratio")
        grouped["rating_mean"] = grouped.get("rating_mean", 0.0).fillna(0.0)
        df = grouped
        derived_cols.append("y")

    df = df.sort_values(["series_id", "time"]).reset_index(drop=True)
    df = _regularize_frequency(
        df,
        freq=freq,
        numeric_cols=[col for col in df.columns if col not in {"series_id", "time", "main_category"}],
        categorical_cols=[col for col in df.columns if df[col].dtype == object and col not in {"series_id"}],
    )
    df = df.sort_values(["series_id", "time"]).reset_index(drop=True)
    per_series_counts = df.groupby("series_id").size()
    valid_series = per_series_counts[per_series_counts >= min_length].index
    dropped = len(per_series_counts) - len(valid_series)
    df = df[df["series_id"].isin(valid_series)].copy()

    df["time_period"] = df["time"].dt.to_period(freq)
    df["time_idx"] = df["time_period"].view("int64")
    df["time_idx"] = df["time_idx"] - df["time_idx"].min()

    info = {
        "series_total": int(per_series_counts.size),
        "series_retained": int(len(valid_series)),
        "series_dropped": int(dropped),
    }
    return df.drop(columns=["time_period"]), info


def build_schema_report(panel: pd.DataFrame) -> Dict[str, object]:
    lengths = panel.groupby("series_id").size()
    report = {
        "rows": int(len(panel)),
        "series": int(panel["series_id"].nunique()),
        "time_start": panel["time"].min().isoformat() if not panel.empty else None,
        "time_end": panel["time"].max().isoformat() if not panel.empty else None,
        "lengths": {
            "min": int(lengths.min()) if not lengths.empty else 0,
            "median": float(lengths.median()) if not lengths.empty else 0.0,
            "max": int(lengths.max()) if not lengths.empty else 0,
        },
        "missing_counts": {col: int(panel[col].isna().sum()) for col in panel.columns},
    }
    return report


def write_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
