"""
DarkHorse: AI-Driven Dynamic Load Balancing for High-Throughput Distributed Systems

Main entry point for the demand forecasting and load balancing pipeline.
"""
import argparse
import sys
from pathlib import Path


def show_menu():
    """Display interactive menu for pipeline selection."""
    print("\n" + "="*60)
    print("  DarkHorse - AI-Driven Demand Forecasting & Load Balancing")
    print("="*60)
    print("\nAvailable Pipelines:")
    print("  1. Data Cleaning - Process raw Amazon review data")
    print("  2. Build Weekly Dataset - Aggregate reviews to weekly panel")
    print("  3. Train Transformer - Train time series transformer model")
    print("  4. Forecast Pipeline - Run TFT + AutoTS ensemble forecasting")
    print("  5. Generate ASIN Mapping - Create category mapping file")
    print("  0. Exit")
    print("="*60)


def run_data_cleaning():
    """Run the data cleaning pipeline."""
    print("\n[Data Cleaning Pipeline]")
    print("This script processes raw Amazon review data from the 2023 dataset.")
    print("Note: Requires datasets library and substantial storage/memory.")
    print("\nTo run manually:")
    print("  python dataCleaning.py")
    print("\nConfiguration: Edit dataCleaning.py to set output path and shard count")


def run_build_weekly():
    """Run the weekly dataset builder."""
    print("\n[Build Weekly Dataset]")
    print("Aggregates review-level data into weekly panel format.")
    print("\nUsage:")
    print("  python build_weekly_dataset.py --input <reviews.csv> --out <output.csv>")
    print("\nOptions:")
    print("  --input         Input CSV with review data")
    print("  --out           Output CSV path for weekly panel")
    print("  --top_quantile  Quantile threshold for hot-seller labels (default: 0.95)")
    print("  --min_reviews   Minimum rolling reviews to keep (default: 1)")
    print("\nExample:")
    print("  python build_weekly_dataset.py \\")
    print("    --input cleaned_beauty_reviews.csv \\")
    print("    --out weekly_beauty_panel.csv \\")
    print("    --top_quantile 0.95")


def run_train_transformer():
    """Run transformer model training."""
    print("\n[Train Transformer Model]")
    print("Trains a time series transformer for demand prediction.")
    print("\nUsage:")
    print("  python train_transformer.py --data <weekly_panel.csv> --out <output_dir>")
    print("\nOptions:")
    print("  --data         Input weekly panel CSV")
    print("  --out          Output directory for model artifacts")
    print("  --seq_len      Sequence length for transformer (default: 32)")
    print("  --batch_size   Batch size (default: 256)")
    print("  --epochs       Training epochs (default: 10)")
    print("  --lr           Learning rate (default: 1e-3)")
    print("\nExample:")
    print("  python train_transformer.py \\")
    print("    --data weekly_beauty_panel.csv \\")
    print("    --out transformer_output \\")
    print("    --epochs 20")


def run_forecast_pipeline():
    """Run the full forecast pipeline."""
    print("\n[Forecast Pipeline (TFT + AutoTS Ensemble)]")
    print("Runs the complete forecasting pipeline with ensemble models.")
    print("\nUsage:")
    print("  python forecast_pipeline.py --dataset <panel.csv> \\")
    print("    --series-col parent_asin --time-col week_start")
    print("\nKey Options:")
    print("  --dataset          Input panel CSV")
    print("  --series-col       Column for series identifiers")
    print("  --time-col         Timestamp column")
    print("  --target-col       Target variable (if not aggregated)")
    print("  --horizon          Forecast horizon (default: 12)")
    print("  --val-size         Validation size in time steps (default: 12)")
    print("  --disable-tft      Skip TFT model")
    print("  --disable-autots   Skip AutoTS model")
    print("  --outdir           Output directory (default: out)")
    print("\nExample:")
    print("  python forecast_pipeline.py \\")
    print("    --dataset weekly_beauty_panel.csv \\")
    print("    --series-col parent_asin \\")
    print("    --time-col week_start \\")
    print("    --horizon 12 \\")
    print("    --outdir forecast_output")


def run_generate_mapping():
    """Run ASIN mapping generator."""
    print("\n[Generate ASIN Mapping]")
    print("Creates asin2category.json mapping file.")
    print("\nUsage:")
    print("  python generate_asin_mapping.py --input <reviews.csv>")
    print("\nOptions:")
    print("  --input   Input CSV with parent_asin and main_category")
    print("  --output  Output JSON path (default: asin2category.json)")
    print("  --sample  Generate sample mapping for testing")
    print("\nExample:")
    print("  python generate_asin_mapping.py --input cleaned_beauty_reviews.csv")


def main():
    """Main entry point with CLI or interactive menu."""
    parser = argparse.ArgumentParser(
        description="DarkHorse - AI-Driven Demand Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        choices=["clean", "weekly", "transformer", "forecast", "mapping"],
        help="Pipeline to run (clean, weekly, transformer, forecast, mapping)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive menu mode"
    )

    args, remaining = parser.parse_known_args()

    if args.pipeline:
        # Direct pipeline execution
        if args.pipeline == "clean":
            run_data_cleaning()
        elif args.pipeline == "weekly":
            run_build_weekly()
        elif args.pipeline == "transformer":
            run_train_transformer()
        elif args.pipeline == "forecast":
            run_forecast_pipeline()
        elif args.pipeline == "mapping":
            run_generate_mapping()
    else:
        # Interactive menu mode
        while True:
            show_menu()
            choice = input("\nSelect pipeline (0-5): ").strip()

            if choice == "0":
                print("\nExiting DarkHorse. Goodbye!")
                sys.exit(0)
            elif choice == "1":
                run_data_cleaning()
            elif choice == "2":
                run_build_weekly()
            elif choice == "3":
                run_train_transformer()
            elif choice == "4":
                run_forecast_pipeline()
            elif choice == "5":
                run_generate_mapping()
            else:
                print("\n[Error] Invalid choice. Please select 0-5.")

            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()