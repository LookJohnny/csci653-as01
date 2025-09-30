"""
Generate asin2category.json mapping from the dataset.
This script extracts parent_asin -> main_category mappings from available data.
"""
import json
import pandas as pd
from pathlib import Path


def generate_from_csv(csv_path: str, output_path: str = "asin2category.json") -> None:
    """
    Generate ASIN to category mapping from a CSV file.

    Args:
        csv_path: Path to input CSV containing parent_asin and main_category columns
        output_path: Path to output JSON file
    """
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if "parent_asin" not in df.columns or "main_category" not in df.columns:
        raise ValueError("CSV must contain 'parent_asin' and 'main_category' columns")

    # Create mapping, taking the first category for each ASIN
    mapping = df.groupby("parent_asin")["main_category"].first().to_dict()

    print(f"Generated {len(mapping)} ASIN -> category mappings")

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved mapping to {output_path}")

    # Print sample
    sample_keys = list(mapping.keys())[:5]
    print("\nSample mappings:")
    for key in sample_keys:
        print(f"  {key}: {mapping[key]}")


def generate_sample_mapping(output_path: str = "asin2category.json") -> None:
    """
    Generate a sample ASIN mapping for testing purposes.
    """
    print("Generating sample ASIN mapping...")

    # Sample mapping based on the cleaned_beauty_reviews.csv structure
    sample_mapping = {
        "B086WK6KMN": "All Beauty",
        "B08665V1RQ": "All Beauty",
        "B0C17MLTNF": "All Beauty",
        "B07H7B4K6P": "All Beauty",
        "B01N32VJJK": "All Beauty",
    }

    with open(output_path, "w") as f:
        json.dump(sample_mapping, f, indent=2)

    print(f"Generated sample mapping with {len(sample_mapping)} entries")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ASIN to category mapping")
    parser.add_argument(
        "--input",
        help="Input CSV file with parent_asin and main_category columns"
    )
    parser.add_argument(
        "--output",
        default="asin2category.json",
        help="Output JSON file path (default: asin2category.json)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate a sample mapping for testing"
    )

    args = parser.parse_args()

    if args.sample:
        generate_sample_mapping(args.output)
    elif args.input:
        generate_from_csv(args.input, args.output)
    else:
        # Try to auto-detect available CSV files
        possible_files = ["cleaned_beauty_reviews.csv", "sample.csv", "weekly_beauty_panel.csv"]
        found = None

        for fname in possible_files:
            if Path(fname).exists():
                found = fname
                break

        if found:
            print(f"Auto-detected {found}")
            generate_from_csv(found, args.output)
        else:
            print("No input file specified. Use --input <file> or --sample")
            parser.print_help()