import argparse
import pandas as pd
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Build weekly panel and hot-seller labels from review data.")
    ap.add_argument("--input", required=True, help="CSV of review-level data (one row per review)")
    ap.add_argument("--out", required=True, help="Output CSV filepath for weekly panel")
    ap.add_argument("--top_quantile", type=float, default=0.95, help="Quantile threshold for label assignment (default=0.95)")
    ap.add_argument("--min_reviews", type=int, default=1, help="Drop rows with rolling prev4 reviews below this")
    return ap.parse_args()


def coerce_bool(series: pd.Series, index) -> pd.Series:
    """Normalize verified_purchase style columns to 0/1 ints."""
    if series is None:
        return pd.Series(0, index=index)
    if series.dtype == bool:
        return series.astype(int)
    if np.issubdtype(series.dtype, np.number):
        return (series.fillna(0) != 0).astype(int)
    return series.astype(str).str.lower().isin({"true", "t", "1", "yes"}).astype(int)


def compute_future_sum(series: pd.Series, window: int) -> pd.Series:
    """Sum over the next `window` steps, excluding the current index."""
    reversed_sum = series.iloc[::-1].rolling(window=window, min_periods=1).sum().iloc[::-1]
    future = (reversed_sum - series).clip(lower=0)
    return future


def build_weekly_panel(df: pd.DataFrame, top_q: float, min_reviews: int) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Input must contain a 'timestamp' column")
    if "parent_asin" not in df.columns:
        raise ValueError("Input must contain a 'parent_asin' column")

    ts = pd.to_datetime(df["timestamp"], unit='ms', errors="coerce", utc=True)
    if ts.isna().all():
        raise ValueError("All timestamps failed to parse; check the input format")
    df["week_start"] = ts.dt.tz_localize(None).dt.to_period("W-MON").dt.start_time

    df["verified_flag"] = coerce_bool(df.get("verified_purchase"), df.index)

    df["helpful_vote"] = pd.to_numeric(df.get("helpful_vote", 0), errors="coerce").fillna(0)
    df["rating"] = pd.to_numeric(df.get("rating", np.nan), errors="coerce")

    agg_dict = {
        "reviews": ("parent_asin", "size"),
        "helpful_sum": ("helpful_vote", "sum"),
        "verified_count": ("verified_flag", "sum"),
        "rating_mean": ("rating", "mean"),
    }

    # Only include main_category if it exists
    if "main_category" in df.columns:
        agg_dict["main_category"] = ("main_category", "first")

    agg = df.groupby(["parent_asin", "week_start"], as_index=False).agg(**agg_dict)

    agg["verified_ratio"] = np.where(agg["reviews"] > 0, agg["verified_count"] / agg["reviews"], 0.0)
    agg["helpful_sum"] = agg["helpful_sum"].fillna(0)
    agg["rating_mean"] = agg["rating_mean"].fillna(0)

    agg = agg.sort_values(["parent_asin", "week_start"]).reset_index(drop=True)

    def add_rolls(group: pd.DataFrame) -> pd.DataFrame:
        reviews = group["reviews"].astype(float)
        group["rev_prev4"] = reviews.rolling(window=4, min_periods=1).sum()
        group["rev_next12"] = compute_future_sum(reviews, window=12)
        group["growth_score"] = group["rev_next12"] / (group["rev_prev4"] + 1.0)
        return group

    agg = agg.groupby("parent_asin", group_keys=False).apply(add_rolls)

    agg = agg[agg["rev_prev4"] >= min_reviews].copy()

    thresh = agg.groupby("week_start")["growth_score"].quantile(top_q)
    agg = agg.merge(thresh.rename("growth_threshold"), left_on="week_start", right_index=True, how="left")
    agg["label_top5"] = (agg["growth_score"] >= agg["growth_threshold"]).astype(int)

    return agg


def main():
    args = parse_args()
    # Support both CSV and parquet input
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    panel = build_weekly_panel(df, top_q=args.top_quantile, min_reviews=args.min_reviews)
    panel.to_csv(args.out, index=False)
    print(f"Saved panel with {len(panel)} rows to {args.out}")


if __name__ == "__main__":
    main()
