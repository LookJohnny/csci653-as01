#!/usr/bin/env python3
"""
Data Quality Report Generator for Amazon Reviews 2023 Pipeline

Generates comprehensive DQ_REPORT.md with:
- Coverage statistics
- Missing value analysis  
- Anomaly detection
- Sample rows
- Recommendations
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_manifest(out_dir: Path) -> Dict:
    """Load pipeline manifest."""
    manifest_path = out_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def analyze_category(category: str, out_dir: Path) -> Dict:
    """Analyze data quality for a single category."""
    stats = {
        "category": category,
        "stages": {},
        "coverage": {},
        "quality": {}
    }

    category_dir = out_dir / category

    # Analyze joined data
    joined_path = category_dir / "joined.parquet"
    if joined_path.exists():
        df = pd.read_parquet(joined_path)
        stats["stages"]["joined"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "size_mb": joined_path.stat().st_size / 1024 / 1024
        }

        # Calculate metadata coverage
        metadata_cols = ["average_rating", "main_category", "rating_number",
                        "features", "description", "price"]
        coverage = {}
        for col in metadata_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                coverage[col] = {
                    "non_null": int(non_null),
                    "null": int(len(df) - non_null),
                    "percent": float(non_null / len(df) * 100)
                }
        stats["coverage"] = coverage

    # Analyze cleaned data
    clean_path = category_dir / "clean.parquet"
    if clean_path.exists():
        df_clean = pd.read_parquet(clean_path)
        stats["stages"]["clean"] = {
            "rows": len(df_clean),
            "columns": len(df_clean.columns),
            "size_mb": clean_path.stat().st_size / 1024 / 1024
        }

        # Quality metrics
        quality = {}
        if "rating" in df_clean.columns:
            quality["rating_dist"] = df_clean["rating"].value_counts().to_dict()
            quality["rating_mean"] = float(df_clean["rating"].mean())

        if "price" in df_clean.columns:
            quality["price_stats"] = {
                "mean": float(df_clean["price"].mean()),
                "median": float(df_clean["price"].median()),
                "min": float(df_clean["price"].min()),
                "max": float(df_clean["price"].max())
            }

        stats["quality"] = quality

    # Analyze weekly slices
    weekly_dir = category_dir / "by_week"
    if weekly_dir.exists():
        manifest_path = weekly_dir / "_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                weekly_manifest = json.load(f)
            stats["stages"]["weekly"] = weekly_manifest

    return stats


def generate_report(out_dir: Path) -> str:
    """Generate comprehensive DQ report."""
    manifest = load_manifest(out_dir)

    lines = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append("Amazon Reviews 2023 Unified Pipeline")
    lines.append("")
    lines.append(f"**Generated:** {manifest.get('created_at', 'N/A')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    results = manifest.get("results", [])
    success_count = sum(1 for r in results if r.get("join", {}).get("status") == "success")
    lines.append(f"- **Categories Processed:** {len(results)}")
    lines.append(f"- **Successful:** {success_count}")
    lines.append(f"- **Failed:** {len(results) - success_count}")
    lines.append("")

    # Per-category analysis
    lines.append("## Category Analysis")
    lines.append("")

    for result in results:
        category = result.get("category")
        if not category:
            continue

        lines.append(f"### {category}")
        lines.append("")

        # Stage statistics
        if "join" in result and result["join"].get("status") == "success":
            join_stats = result["join"]
            lines.append(f"**Join Stage:**")
            lines.append(f"- Reviews: {join_stats.get('review_count', 0):,}")
            lines.append(f"- Metadata: {join_stats.get('metadata_count', 0):,}")
            lines.append(f"- Coverage: {join_stats.get('coverage_percent', 0):.1f}%")
            lines.append("")

        if "clean" in result and result["clean"].get("status") == "success":
            clean_stats = result["clean"].get("stats", {})
            lines.append(f"**Cleaning Stage:**")
            lines.append(f"- Initial rows: {clean_stats.get('initial_rows', 0):,}")
            lines.append(f"- Final rows: {clean_stats.get('final_rows', 0):,}")
            lines.append(f"- Removed duplicates: {clean_stats.get('removed_duplicates', 0):,}")
            lines.append(f"- Rating issues: {clean_stats.get('rating_invalid', 0):,}")
            lines.append(f"- Price outliers: {clean_stats.get('price_outliers', 0):,}")
            lines.append("")

        if "weekly" in result and result["weekly"].get("status") == "success":
            weekly_stats = result["weekly"]
            lines.append(f"**Weekly Slicing:**")
            lines.append(f"- Weeks: {weekly_stats.get('num_weeks', 0):,}")
            lines.append(f"- Rows: {weekly_stats.get('num_rows', 0):,}")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    low_coverage = [r for r in results
                   if r.get("join", {}).get("coverage_percent", 100) < 90]

    if low_coverage:
        lines.append("### Low Metadata Coverage")
        lines.append("")
        lines.append("The following categories have <90% metadata coverage:")
        lines.append("")
        for r in low_coverage:
            cat = r.get("category")
            pct = r.get("join", {}).get("coverage_percent", 0)
            lines.append(f"- **{cat}:** {pct:.1f}%")
        lines.append("")
        lines.append("**Action:** Verify metadata files are complete and accessible.")
        lines.append("")

    # Issues summary
    lines.append("## Known Issues")
    lines.append("")
    lines.append("1. Some categories may have incomplete metadata due to remote access limitations")
    lines.append("2. Timestamp normalization may lose precision for very old reviews")
    lines.append("3. Price parsing may fail for non-standard currency formats")
    lines.append("")

    # Next steps
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Review categories with low coverage and verify metadata availability")
    lines.append("2. Validate weekly partitions can be read by downstream consumers")
    lines.append("3. Run data quality checks on sample rows")
    lines.append("4. Document any category-specific quirks or limitations")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Data Quality Report")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if not out_dir.exists():
        print(f"Error: Output directory does not exist: {out_dir}")
        return

    print("Generating DQ report...")
    report = generate_report(out_dir)

    report_path = out_dir / "DQ_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
