"""
Fast parallel version of weekly dataset builder using Dask
Processes 31.8M rows much faster by using parallel computation
"""
import argparse
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def parse_args():
    ap = argparse.ArgumentParser(description="Build weekly panel (FAST version with Dask)")
    ap.add_argument("--input", required=True, help="Parquet file of review data")
    ap.add_argument("--out", required=True, help="Output CSV filepath")
    ap.add_argument("--top_quantile", type=float, default=0.95)
    ap.add_argument("--min_reviews", type=int, default=1)
    ap.add_argument("--npartitions", type=int, default=64, help="Number of parallel partitions")
    return ap.parse_args()


def build_weekly_panel_fast(input_path: str, top_q: float, min_reviews: int, npartitions: int):
    """
    Fast parallel aggregation using Dask
    """
    print(f"Loading data with {npartitions} partitions...")

    # Read with Dask (lazy loading, parallel)
    ddf = dd.read_parquet(input_path, engine='pyarrow')

    print(f"Total rows: {len(ddf):,}")

    # Convert timestamp to datetime (parallel)
    ddf['timestamp'] = dd.to_datetime(ddf['timestamp'], unit='ms', errors='coerce', utc=True)
    ddf['week_start'] = ddf['timestamp'].dt.tz_localize(None).dt.to_period('W-MON').dt.start_time

    # Prepare columns
    ddf['verified_flag'] = ddf['verified_purchase'].fillna(False).astype(int)
    ddf['helpful_vote'] = dd.to_numeric(ddf['helpful_vote'], errors='coerce').fillna(0)
    ddf['rating'] = dd.to_numeric(ddf['rating'], errors='coerce')

    print("Aggregating by product and week (parallel)...")

    # Parallel aggregation
    with ProgressBar():
        agg = ddf.groupby(['parent_asin', 'week_start']).agg({
            'parent_asin': 'size',  # reviews count
            'helpful_vote': 'sum',
            'verified_flag': 'sum',
            'rating': 'mean'
        }).compute()

    # Rename columns
    agg.columns = ['reviews', 'helpful_sum', 'verified_count', 'rating_mean']
    agg = agg.reset_index()

    print(f"Aggregated to {len(agg):,} product-week pairs")

    # Calculate ratios
    agg['verified_ratio'] = agg['verified_count'] / agg['reviews'].clip(lower=1)
    agg['helpful_sum'] = agg['helpful_sum'].fillna(0)
    agg['rating_mean'] = agg['rating_mean'].fillna(0)

    # Sort for rolling windows
    agg = agg.sort_values(['parent_asin', 'week_start']).reset_index(drop=True)

    print("Computing rolling windows (parallel)...")

    # Use Dask for rolling windows too
    ddf_agg = dd.from_pandas(agg, npartitions=npartitions)

    def add_rolls_fast(group):
        """Vectorized rolling window computation"""
        reviews = group['reviews'].astype(float)

        # Previous 4 weeks
        group['rev_prev4'] = reviews.rolling(window=4, min_periods=1).sum()

        # Next 12 weeks (reverse rolling)
        reversed_sum = reviews.iloc[::-1].rolling(window=12, min_periods=1).sum().iloc[::-1]
        group['rev_next12'] = (reversed_sum - reviews).clip(lower=0)

        # Growth score
        group['growth_score'] = group['rev_next12'] / (group['rev_prev4'] + 1.0)

        return group

    with ProgressBar():
        agg = ddf_agg.groupby('parent_asin', group_keys=False).apply(
            add_rolls_fast,
            meta=agg
        ).compute()

    # Filter by minimum reviews
    print(f"Filtering products with rev_prev4 >= {min_reviews}...")
    agg = agg[agg['rev_prev4'] >= min_reviews].copy()
    print(f"After filter: {len(agg):,} rows")

    # Compute threshold per week
    print("Computing quantile thresholds...")
    thresh = agg.groupby('week_start')['growth_score'].quantile(top_q)
    agg = agg.merge(thresh.rename('growth_threshold'), left_on='week_start', right_index=True, how='left')
    agg['label_top5'] = (agg['growth_score'] >= agg['growth_threshold']).astype(int)

    return agg


def main():
    args = parse_args()

    print("=" * 60)
    print("FAST WEEKLY PANEL BUILDER (Dask Parallel)")
    print("=" * 60)

    panel = build_weekly_panel_fast(
        args.input,
        args.top_quantile,
        args.min_reviews,
        args.npartitions
    )

    print(f"\nSaving {len(panel):,} rows to {args.out}...")
    panel.to_csv(args.out, index=False)
    print("✓ Done!")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total rows: {len(panel):,}")
    print(f"  Unique products: {panel['parent_asin'].nunique():,}")
    print(f"  Date range: {panel['week_start'].min()} to {panel['week_start'].max()}")
    print(f"  Hot-sellers: {panel['label_top5'].sum():,} ({panel['label_top5'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()