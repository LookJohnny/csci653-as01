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


def coerce_bool(series: pd.Series) -> pd.Series:
    # Accept booleans, strings ("true"/"false"), or integers
    if series.dtype == bool:
        return series.astype(int)
    if np.issubdtype(series.dtype, np.number):
        return (series != 0).astype(int)
    return series.astype(str).str.lower().isin({"true", "t", "1", "yes"}).astype(int)


def build_weekly_panel(df: pd.DataFrame, top_q: float, min_reviews: int) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Input must contain a 'timestamp' column")

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if ts.isna().all():
        raise ValueError("All timestamps failed to parse; check the input format")
    # Use naive datetime if original data is naive but keep monotonic weeks
    df["week_start"] = ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz else ts.dt.to_period("W-MON").dt.start_time
    if ts.dt.tz is None:
        df["week_start"] = ts.dt.to_period("W-MON").dt.start_time

    df["verified_flag"] = coerce_bool(df.get("verified_purchase", pd.Series(False, index=df.index)))
    helpful = df.get("helpful_vote")
    if helpful is None:
        helpful = pd.Series(0, index=df.index)
    helpful = pd.to_numeric(helpful, errors="coerce").fillna(0)

    ratings = pd.to_numeric(df.get("rating"), errors="coerce")

    # Aggregate to weekly level
    agg = df.groupby(["parent_asin", "week_start"], as_index=False).agg(
        reviews=("parent_asin", "size"),
        helpful_sum=(lambda s: helpful.loc[s.index].sum()),
        verified_count=(lambda s: df.loc[s.index, "verified_flag"].sum()),
        rating_mean=(lambda s: ratings.loc[s.index].mean()),
        main_category=("main_category", "first"),
    )

    agg["verified_ratio"] = agg.apply(lambda r: r["verified_count"] / r["reviews"] if r["reviews"] else 0.0, axis=1)
    agg["helpful_sum"] = agg["helpful_sum"].fillna(0)
    agg["rating_mean"] = agg["rating_mean"].fillna(0)

    agg = agg.sort_values(["parent_asin", "week_start"]).reset_index(drop=True)

    def add_rolls(group: pd.DataFrame) -> pd.DataFrame:
        reviews = group["reviews"]
        group["rev_prev4"] = reviews.rolling(window=4, min_periods=1).sum()
        # Compute future 12-week sum excluding current week
        shifted = reviews[::-1].rolling(window=12, min_periods=1).sum()[::-1]
        group["rev_next12"] = (shifted - reviews).clip(lower=0)
        group["growth_score"] = group["rev_next12"] / (group["rev_prev4"] + 1.0)
        return group

    agg = agg.groupby("parent_asin", group_keys=False).apply(add_rolls)

    agg = agg[agg["rev_prev4"] >= min_reviews].copy()

    # Assign labels per calendar week
    thresh = agg.groupby("week_start")["growth_score"].quantile(top_q)
    agg = agg.merge(thresh.rename("growth_threshold"), left_on="week_start", right_index=True, how="left")
    agg["label_top5"] = (agg["growth_score"] >= agg["growth_threshold"]).astype(int)

    return agg


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    panel = build_weekly_panel(df, top_q=args.top_quantile, min_reviews=args.min_reviews)
    panel.to_csv(args.out, index=False)
    print(f"Saved panel with {len(panel)} rows to {args.out}")


if __name__ == "__main__":
    main()
