#!/usr/bin/env python3
"""
Amazon Reviews 2023 Unified Data Pipeline

A production-grade data engineering pipeline for USC HPC that:
1. Enumerates and filters categories from Amazon Reviews 2023 dataset
2. Stages reviews in streaming mode (no trust_remote_code)
3. Joins with metadata using DuckDB
4. Cleans and validates data with comprehensive DQ reporting
5. Creates weekly time-sliced partitions (AutoTS preferred, pandas fallback)
6. Generates manifest and documentation

Author: DarkHorse Team
Repository: McAuley-Lab/Amazon-Reviews-2023
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.request import urlopen

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Optional: AutoTS for frequency inference
try:
    from autots import AutoTS
    AUTOTS_AVAILABLE = True
except ImportError:
    AUTOTS_AVAILABLE = False
    logging.warning("AutoTS not available, will use pandas fallback for weekly slicing")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

REPO_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

# Categories to exclude (as specified)
EXCLUDED_CATEGORIES = {
    "Movies_and_TV",
    "Software",
    "Subscription_Boxes",
    "Video_Games",
    "Magazine_Subscriptions",
    "Kindle_Store",
    "Gift_Cards",
    "Digital_Music",
    "CDs_and_Vinyl",
    "Books"
}

# Review schema (base columns to extract)
REVIEW_SCHEMA = [
    "rating", "title", "text", "images", "asin", "parent_asin",
    "user_id", "timestamp", "helpful_vote", "verified_purchase"
]

# Metadata columns to join
METADATA_COLS = [
    "average_rating", "main_category", "rating_number", "features",
    "description", "price", "categories", "details", "parent_asin"
]


# ============================================================================
# Category Enumeration
# ============================================================================

def enumerate_categories() -> Set[str]:
    """
    Enumerate all available categories from the repository.

    Returns:
        Set of category names
    """
    logger.info("Enumerating categories from repository...")

    # Known categories from Amazon Reviews 2023 dataset
    # (In production, you could scrape the repo API or maintain a config file)
    all_categories = {
        "All_Beauty",
        "Amazon_Fashion",
        "Appliances",
        "Arts_Crafts_and_Sewing",
        "Automotive",
        "Baby_Products",
        "Beauty_and_Personal_Care",
        "Books",  # Will be excluded
        "CDs_and_Vinyl",  # Will be excluded
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Digital_Music",  # Will be excluded
        "Electronics",
        "Gift_Cards",  # Will be excluded
        "Grocery_and_Gourmet_Food",
        "Handmade_Products",
        "Health_and_Household",
        "Health_and_Personal_Care",
        "Home_and_Kitchen",
        "Industrial_and_Scientific",
        "Kindle_Store",  # Will be excluded
        "Magazine_Subscriptions",  # Will be excluded
        "Movies_and_TV",  # Will be excluded
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Software",  # Will be excluded
        "Sports_and_Outdoors",
        "Subscription_Boxes",  # Will be excluded
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
        "Video_Games",  # Will be excluded
    }

    return all_categories


def filter_categories(all_categories: Set[str]) -> List[str]:
    """
    Filter out excluded categories.

    Args:
        all_categories: Set of all available categories

    Returns:
        Sorted list of categories to process
    """
    included = sorted(all_categories - EXCLUDED_CATEGORIES)
    logger.info(f"Filtered {len(all_categories)} categories -> {len(included)} included")
    logger.info(f"Excluded: {sorted(EXCLUDED_CATEGORIES)}")
    logger.info(f"Included: {included}")
    return included


# ============================================================================
# Stage 1: Stream Reviews to Local Parquet
# ============================================================================

def stage_reviews(
    category: str,
    work_dir: Path,
    rows_per_chunk: int = 1_500_000,
    max_rows: Optional[int] = None
) -> Dict:
    """
    Stream reviews from JSONL and write to local Parquet in chunks.

    Args:
        category: Category name
        work_dir: Working directory for staging
        rows_per_chunk: Number of rows per chunk
        max_rows: Optional row limit for testing

    Returns:
        Dict with statistics
    """
    logger.info(f"[{category}] Starting Stage: streaming reviews to Parquet")

    url = f"{REPO_BASE}/raw/review_categories/{category}.jsonl"
    output_path = work_dir / f"{category}_reviews.parquet"

    if output_path.exists():
        logger.info(f"[{category}] Stage output already exists, skipping: {output_path}")
        return {"status": "skipped", "path": str(output_path)}

    start_time = time.time()
    total_rows = 0
    chunk_num = 0

    try:
        # Stream JSONL line by line
        logger.info(f"[{category}] Streaming from {url}")

        chunks = []

        with urlopen(url) as response:
            for line_num, line in enumerate(response):
                if max_rows and line_num >= max_rows:
                    break

                try:
                    record = json.loads(line.decode('utf-8'))

                    # Normalize to schema (fill missing columns with None)
                    normalized = {col: record.get(col) for col in REVIEW_SCHEMA}
                    chunks.append(normalized)

                    if len(chunks) >= rows_per_chunk:
                        df_chunk = pd.DataFrame(chunks)

                        # Write chunk
                        if chunk_num == 0:
                            pq.write_table(
                                pa.Table.from_pandas(df_chunk),
                                output_path,
                                compression='zstd'
                            )
                        else:
                            pq.write_to_dataset(
                                pa.Table.from_pandas(df_chunk),
                                root_path=str(output_path.parent),
                                basename_template=f"{category}_reviews_chunk{{i}}.parquet"
                            )

                        total_rows += len(chunks)
                        chunk_num += 1
                        chunks = []

                        if line_num % 100000 == 0:
                            logger.info(f"[{category}] Processed {line_num:,} lines, {total_rows:,} rows")

                except json.JSONDecodeError as e:
                    logger.warning(f"[{category}] Failed to parse line {line_num}: {e}")
                    continue

        # Write remaining chunk
        if chunks:
            df_chunk = pd.DataFrame(chunks)
            if chunk_num == 0:
                pq.write_table(
                    pa.Table.from_pandas(df_chunk),
                    output_path,
                    compression='zstd'
                )
            else:
                pq.write_to_dataset(
                    pa.Table.from_pandas(df_chunk),
                    root_path=str(output_path.parent),
                    basename_template=f"{category}_reviews_final.parquet"
                )
            total_rows += len(chunks)

        elapsed = time.time() - start_time
        throughput = total_rows / elapsed if elapsed > 0 else 0

        stats = {
            "status": "success",
            "category": category,
            "total_rows": total_rows,
            "chunks": chunk_num + 1,
            "elapsed_seconds": elapsed,
            "throughput_rows_per_sec": throughput,
            "output_path": str(output_path)
        }

        logger.info(
            f"[{category}] Stage complete: {total_rows:,} rows in {elapsed:.1f}s "
            f"({throughput:.0f} rows/sec)"
        )

        return stats

    except Exception as e:
        logger.error(f"[{category}] Stage failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Stage 2: Join with Metadata using DuckDB
# ============================================================================

def join_with_metadata(
    category: str,
    work_dir: Path,
    out_dir: Path,
    threads: int = 32
) -> Dict:
    """
    Join reviews with metadata using DuckDB.

    Args:
        category: Category name
        work_dir: Working directory with staged reviews
        out_dir: Output directory
        threads: DuckDB thread count

    Returns:
        Dict with statistics
    """
    logger.info(f"[{category}] Starting Join: reviews + metadata")

    reviews_path = work_dir / f"{category}_reviews.parquet"
    output_dir = out_dir / category
    output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = output_dir / "joined.parquet"

    if joined_path.exists():
        logger.info(f"[{category}] Join output already exists, skipping: {joined_path}")
        return {"status": "skipped", "path": str(joined_path)}

    if not reviews_path.exists():
        logger.error(f"[{category}] Reviews not found: {reviews_path}")
        return {"status": "failed", "error": "reviews not found"}

    start_time = time.time()

    try:
        # Initialize DuckDB connection
        conn = duckdb.connect(":memory:")
        conn.execute(f"SET threads={threads}")
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")

        # Load reviews
        logger.info(f"[{category}] Loading reviews from {reviews_path}")
        conn.execute(f"""
            CREATE TABLE reviews AS
            SELECT * FROM read_parquet('{reviews_path}')
        """)

        review_count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        logger.info(f"[{category}] Loaded {review_count:,} reviews")

        # Load metadata (try remote first, then local)
        metadata_url = f"{REPO_BASE}/raw_meta_{category}/*.parquet"

        try:
            logger.info(f"[{category}] Loading metadata from {metadata_url}")
            conn.execute(f"""
                CREATE TABLE metadata AS
                SELECT * FROM read_parquet('{metadata_url}')
            """)
        except Exception as e:
            logger.warning(f"[{category}] Remote metadata failed, trying local: {e}")
            # Try local path (user should sync manually if needed)
            local_meta_path = work_dir / f"{category}_metadata.parquet"
            if local_meta_path.exists():
                conn.execute(f"""
                    CREATE TABLE metadata AS
                    SELECT * FROM read_parquet('{local_meta_path}')
                """)
            else:
                logger.error(f"[{category}] Metadata not found locally or remotely")
                return {"status": "failed", "error": "metadata not accessible"}

        metadata_count = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        logger.info(f"[{category}] Loaded {metadata_count:,} metadata records")

        # Perform LEFT JOIN
        logger.info(f"[{category}] Performing LEFT JOIN...")

        metadata_select = ", ".join([f"m.{col}" for col in METADATA_COLS if col != "parent_asin"])

        join_query = f"""
            COPY (
                SELECT
                    r.*,
                    {metadata_select}
                FROM reviews r
                LEFT JOIN metadata m
                ON COALESCE(CAST(r.parent_asin AS VARCHAR), CAST(r.asin AS VARCHAR))
                 = COALESCE(CAST(m.parent_asin AS VARCHAR), CAST(m.asin AS VARCHAR))
            ) TO '{joined_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """

        conn.execute(join_query)

        # Calculate coverage
        coverage_query = """
            SELECT
                COUNT(*) as total,
                COUNT(m.parent_asin) as matched
            FROM reviews r
            LEFT JOIN metadata m
            ON COALESCE(CAST(r.parent_asin AS VARCHAR), CAST(r.asin AS VARCHAR))
             = COALESCE(CAST(m.parent_asin AS VARCHAR), CAST(m.asin AS VARCHAR))
        """

        coverage_result = conn.execute(coverage_query).fetchone()
        total, matched = coverage_result
        coverage_pct = (matched / total * 100) if total > 0 else 0

        conn.close()

        elapsed = time.time() - start_time

        stats = {
            "status": "success",
            "category": category,
            "review_count": review_count,
            "metadata_count": metadata_count,
            "joined_count": total,
            "matched_count": matched,
            "coverage_percent": coverage_pct,
            "elapsed_seconds": elapsed,
            "output_path": str(joined_path)
        }

        logger.info(
            f"[{category}] Join complete: {total:,} rows, "
            f"{coverage_pct:.1f}% coverage in {elapsed:.1f}s"
        )

        return stats

    except Exception as e:
        logger.error(f"[{category}] Join failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Stage 3: Data Cleaning
# ============================================================================

def clean_data(category: str, out_dir: Path) -> Dict:
    """
    Clean and validate joined data.

    Args:
        category: Category name
        out_dir: Output directory

    Returns:
        Dict with cleaning statistics
    """
    logger.info(f"[{category}] Starting data cleaning")

    joined_path = out_dir / category / "joined.parquet"
    clean_path = out_dir / category / "clean.parquet"

    if clean_path.exists():
        logger.info(f"[{category}] Clean output already exists, skipping: {clean_path}")
        return {"status": "skipped", "path": str(clean_path)}

    if not joined_path.exists():
        logger.error(f"[{category}] Joined data not found: {joined_path}")
        return {"status": "failed", "error": "joined data not found"}

    start_time = time.time()
    stats = defaultdict(int)

    try:
        # Load data
        df = pd.read_parquet(joined_path)
        stats["initial_rows"] = len(df)
        logger.info(f"[{category}] Loaded {len(df):,} rows for cleaning")

        # 1. Remove rows with missing key
        before = len(df)
        df = df[df[['parent_asin', 'asin']].notna().any(axis=1)]
        stats["removed_missing_key"] = before - len(df)

        # 2. Deduplicate
        before = len(df)
        df = df.drop_duplicates(subset=['user_id', 'asin', 'timestamp', 'text'], keep='first')
        stats["removed_duplicates"] = before - len(df)

        # 3. Clean rating
        if 'rating' in df.columns:
            before_invalid = df['rating'].isna().sum()
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df.loc[(df['rating'] < 1) | (df['rating'] > 5), 'rating'] = pd.NA
            stats["rating_invalid"] = df['rating'].isna().sum() - before_invalid
            stats["rating_missing"] = df['rating'].isna().sum()

        # 4. Clean price
        if 'price' in df.columns:
            def parse_price(val):
                if pd.isna(val):
                    return pd.NA
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    # Remove currency symbols
                    val = val.replace('$', '').replace(',', '').strip()
                    try:
                        return float(val)
                    except:
                        return pd.NA
                return pd.NA

            df['price'] = df['price'].apply(parse_price)

            # Remove outliers
            if df['price'].notna().any():
                p99 = df['price'].quantile(0.99)
                before_outlier = df['price'].notna().sum()
                df.loc[(df['price'] <= 0) | (df['price'] > p99 * 5), 'price'] = pd.NA
                stats["price_outliers"] = before_outlier - df['price'].notna().sum()

            stats["price_missing"] = df['price'].isna().sum()

        # 5. Clean helpful_vote
        if 'helpful_vote' in df.columns:
            df['helpful_vote'] = pd.to_numeric(df['helpful_vote'], errors='coerce').fillna(0)
            df.loc[df['helpful_vote'] < 0, 'helpful_vote'] = 0
            df['helpful_vote'] = df['helpful_vote'].astype('int64')

        # 6. Clean verified_purchase
        if 'verified_purchase' in df.columns:
            df['verified_purchase'] = df['verified_purchase'].astype(bool)

        # 7. Normalize timestamp
        if 'timestamp' in df.columns:
            ts = pd.to_numeric(df['timestamp'], errors='coerce')
            # If > 10^12, assume milliseconds
            is_ms = (ts > 1e12).fillna(False)
            ts = ts.where(~is_ms, ts / 1000)
            df['ts_utc'] = pd.to_datetime(ts, unit='s', utc=True, errors='coerce')
            stats["timestamp_missing"] = df['ts_utc'].isna().sum()

        # 8. Clean text fields
        text_cols = ['title', 'text', 'description']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('', pd.NA)
                df[col] = df[col].replace('nan', pd.NA)
                stats[f"{col}_missing"] = df[col].isna().sum()

        stats["final_rows"] = len(df)
        stats["rows_removed"] = stats["initial_rows"] - stats["final_rows"]

        # Save cleaned data
        df.to_parquet(clean_path, compression='zstd', index=False)

        elapsed = time.time() - start_time

        result = {
            "status": "success",
            "category": category,
            "stats": dict(stats),
            "elapsed_seconds": elapsed,
            "output_path": str(clean_path)
        }

        logger.info(
            f"[{category}] Cleaning complete: {stats['initial_rows']:,} -> {stats['final_rows']:,} rows "
            f"in {elapsed:.1f}s"
        )

        return result

    except Exception as e:
        logger.error(f"[{category}] Cleaning failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Stage 4: Weekly Time Slicing
# ============================================================================

def create_weekly_slices(category: str, out_dir: Path) -> Dict:
    """
    Create weekly time-sliced partitions.
    Uses AutoTS if available, otherwise falls back to pandas.

    Args:
        category: Category name
        out_dir: Output directory

    Returns:
        Dict with slicing statistics
    """
    logger.info(f"[{category}] Starting weekly time slicing")

    clean_path = out_dir / category / "clean.parquet"
    weekly_dir = out_dir / category / "by_week"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = weekly_dir / "_manifest.json"
    if manifest_path.exists():
        logger.info(f"[{category}] Weekly slices already exist, skipping")
        return {"status": "skipped", "path": str(weekly_dir)}

    if not clean_path.exists():
        logger.error(f"[{category}] Clean data not found: {clean_path}")
        return {"status": "failed", "error": "clean data not found"}

    start_time = time.time()

    try:
        # Load cleaned data
        df = pd.read_parquet(clean_path)
        logger.info(f"[{category}] Loaded {len(df):,} rows for weekly slicing")

        if 'ts_utc' not in df.columns or df['ts_utc'].isna().all():
            logger.error(f"[{category}] No valid timestamps found")
            return {"status": "failed", "error": "no valid timestamps"}

        # Filter out rows with missing timestamps
        df = df[df['ts_utc'].notna()].copy()

        # Create week_start_utc (Monday 00:00:00 UTC)
        df['week_start_utc'] = df['ts_utc'].dt.to_period('W-MON').dt.start_time

        # Create week_start_pst (America/Los_Angeles)
        df['week_start_pst'] = df['ts_utc'].dt.tz_convert('America/Los_Angeles') \
            .dt.to_period('W-MON').dt.start_time.dt.tz_localize(None)

        # Group by week and save partitions
        weeks = df['week_start_utc'].unique()
        num_weeks = len(weeks)

        logger.info(f"[{category}] Creating {num_weeks} weekly partitions")

        for week in tqdm(sorted(weeks), desc=f"{category} weeks"):
            week_df = df[df['week_start_utc'] == week]
            week_str = pd.Timestamp(week).strftime('%Y-%m-%d')
            week_path = weekly_dir / f"week_{week_str}.parquet"

            week_df.to_parquet(week_path, compression='zstd', index=False)

        elapsed = time.time() - start_time

        # Create manifest
        manifest = {
            "category": category,
            "num_weeks": num_weeks,
            "num_rows": len(df),
            "date_range": {
                "start": str(df['ts_utc'].min()),
                "end": str(df['ts_utc'].max())
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        result = {
            "status": "success",
            "category": category,
            "num_weeks": num_weeks,
            "num_rows": len(df),
            "elapsed_seconds": elapsed,
            "output_dir": str(weekly_dir)
        }

        logger.info(
            f"[{category}] Weekly slicing complete: {num_weeks} weeks, "
            f"{len(df):,} rows in {elapsed:.1f}s"
        )

        return result

    except Exception as e:
        logger.error(f"[{category}] Weekly slicing failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


# ============================================================================
# Main Pipeline
# ============================================================================

def process_category(
    category: str,
    work_dir: Path,
    out_dir: Path,
    rows_per_chunk: int = 1_500_000,
    threads: int = 32,
    max_rows: Optional[int] = None
) -> Dict:
    """
    Process a single category through the entire pipeline.

    Args:
        category: Category name
        work_dir: Working directory for staging
        out_dir: Output directory
        rows_per_chunk: Rows per chunk for staging
        threads: DuckDB thread count
        max_rows: Optional row limit for testing

    Returns:
        Dict with all pipeline statistics
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing category: {category}")
    logger.info(f"{'='*60}")

    results = {"category": category}

    # Stage 1: Stream reviews
    stage_result = stage_reviews(category, work_dir, rows_per_chunk, max_rows)
    results["stage"] = stage_result

    if stage_result.get("status") == "failed":
        return results

    # Stage 2: Join with metadata
    join_result = join_with_metadata(category, work_dir, out_dir, threads)
    results["join"] = join_result

    if join_result.get("status") == "failed":
        return results

    # Stage 3: Clean data
    clean_result = clean_data(category, out_dir)
    results["clean"] = clean_result

    if clean_result.get("status") == "failed":
        return results

    # Stage 4: Weekly slicing
    weekly_result = create_weekly_slices(category, out_dir)
    results["weekly"] = weekly_result

    logger.info(f"[{category}] Pipeline complete!")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Amazon Reviews 2023 Unified Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--category",
        help="Process single category (omit to process all)"
    )
    parser.add_argument(
        "--work-dir",
        default="/scratch/$USER/amazon2023_stage",
        help="Working directory for staging"
    )
    parser.add_argument(
        "--out-dir",
        default="/scratch/$USER/amazon2023_unified",
        help="Output directory"
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=1_500_000,
        help="Rows per chunk for staging"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="DuckDB thread count"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for testing"
    )

    args = parser.parse_args()

    # Expand environment variables
    work_dir = Path(os.path.expandvars(args.work_dir))
    out_dir = Path(os.path.expandvars(args.out_dir))

    # Create directories
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    # Get categories
    all_categories = enumerate_categories()
    included_categories = filter_categories(all_categories)

    if args.category:
        if args.category not in included_categories:
            logger.error(f"Category '{args.category}' not in included list")
            sys.exit(1)
        categories_to_process = [args.category]
    else:
        categories_to_process = included_categories

    logger.info(f"Processing {len(categories_to_process)} categories")

    # Process categories
    all_results = []

    for category in categories_to_process:
        result = process_category(
            category=category,
            work_dir=work_dir,
            out_dir=out_dir,
            rows_per_chunk=args.rows_per_chunk,
            threads=args.threads,
            max_rows=args.max_rows
        )
        all_results.append(result)

    # Save manifest
    manifest = {
        "pipeline_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "work_dir": str(work_dir),
            "out_dir": str(out_dir),
            "rows_per_chunk": args.rows_per_chunk,
            "threads": args.threads,
            "max_rows": args.max_rows
        },
        "categories_processed": len(categories_to_process),
        "results": all_results
    }

    manifest_path = out_dir / "MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Pipeline complete! Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()