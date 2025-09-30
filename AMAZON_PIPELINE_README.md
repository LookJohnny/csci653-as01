# Amazon Reviews 2023 Unified Pipeline

Production-grade data engineering pipeline for processing Amazon Reviews 2023 dataset on USC HPC with Slurm.

## Overview

This pipeline:
1. ✅ **Enumerates & Filters** - Automatically identifies and excludes specified categories
2. 📥 **Stages Reviews** - Streams JSONL reviews to local Parquet (memory-efficient)
3. 🔗 **Joins Metadata** - Left joins with metadata using DuckDB (high-performance)
4. 🧹 **Cleans Data** - Comprehensive data cleaning and validation
5. 📅 **Weekly Slicing** - Creates weekly time-partitioned data (AutoTS + pandas fallback)
6. 📊 **Reports Quality** - Generates manifest and DQ reports

### Key Features

- **No `trust_remote_code`** - Direct file access for security
- **Memory Efficient** - Streaming processing with chunking
- **HPC Optimized** - Slurm job arrays for parallel processing
- **Resumable** - Skip completed stages automatically
- **Comprehensive DQ** - Detailed data quality reports

---

## Quick Start

### 1. Validation (Recommended First Step)

Test with small dataset before full run:

```bash
# Run validation on 2 categories with 500k rows each
./validate_pipeline.sh
```

This will:
- Process `All_Beauty` and `Electronics` with 500k rows
- Generate all outputs (joined, clean, weekly)
- Create DQ report
- Verify file integrity

**Review the output before proceeding to full run!**

### 2. Full Pipeline (USC HPC with Slurm)

```bash
# Submit job array for all 23 categories
sbatch run_unify.slurm

# Monitor jobs
squeue -u $USER

# Check specific category log
tail -f /scratch/$USER/amazon2023_unified/logs/<job_id>_<array_id>.out
```

### 3. Generate DQ Report (After Completion)

```bash
python generate_dq_report.py --out-dir /scratch/$USER/amazon2023_unified
```

---

## Installation

### Prerequisites

- Python 3.8+
- USC HPC access with Slurm
- Conda environment

### Setup Environment

```bash
# Create conda environment
conda create -n darkhorse python=3.10
conda activate darkhorse

# Install dependencies
pip install pandas pyarrow duckdb tqdm autots python-dateutil pytz
```

### Dependencies

```
pandas>=2.0.0
pyarrow>=10.0.0
duckdb>=0.9.0
tqdm>=4.60.0
autots>=0.6.7  # Optional, pandas fallback available
python-dateutil>=2.8.0
pytz>=2023.3
```

---

## Configuration

### Environment Variables

```bash
# Override defaults
export WORK_DIR=/scratch/$USER/custom_stage
export OUT_DIR=/scratch/$USER/custom_output
export ROWS_PER_CHUNK=2000000
export THREADS=64
export MAX_ROWS=1000000  # For testing only
```

### Configuration File

See `pipeline_config.yaml` for all available options:

- Excluded categories
- Schema definitions
- Cleaning rules
- Slurm parameters
- Output formats

---

## Pipeline Stages

### Stage 1: Stream Reviews

**Input:** `https://huggingface.co/.../raw/review_categories/{category}.jsonl`  
**Output:** `{WORK_DIR}/{category}_reviews.parquet`

Streams JSONL reviews line-by-line and writes chunked Parquet files.

**Schema:**
- rating, title, text, images
- asin, parent_asin, user_id
- timestamp, helpful_vote, verified_purchase

### Stage 2: Join Metadata

**Input:** Reviews + Metadata Parquet  
**Output:** `{OUT_DIR}/{category}/joined.parquet`

Uses DuckDB to LEFT JOIN reviews with metadata on:
```sql
COALESCE(parent_asin, asin)
```

**Metadata Columns:**
- average_rating, main_category, rating_number
- features, description, price
- categories, details

### Stage 3: Data Cleaning

**Input:** `joined.parquet`  
**Output:** `{OUT_DIR}/{category}/clean.parquet`

**Cleaning Rules:**
1. **Deduplication** - Remove exact duplicates by (user_id, asin, timestamp, text)
2. **Rating** - Clip to [1, 5], set outliers to NaN
3. **Price** - Parse currency, remove outliers (price > p99 × 5)
4. **Helpful Votes** - Convert to int64, negative → 0
5. **Verified Purchase** - Convert to boolean
6. **Timestamp** - Normalize ms/s to UTC datetime
7. **Text Fields** - Empty strings → NaN
8. **Missing Keys** - Drop rows without asin/parent_asin

### Stage 4: Weekly Slicing

**Input:** `clean.parquet`  
**Output:** `{OUT_DIR}/{category}/by_week/week_YYYY-MM-DD.parquet`

Creates weekly partitions (Monday 00:00:00):
- `week_start_utc` - UTC week start
- `week_start_pst` - America/Los_Angeles week start

**Method:**
1. Try AutoTS frequency inference (if available)
2. Fallback to pandas `dt.to_period('W-MON')`

---

## Output Structure

```
/scratch/$USER/amazon2023_unified/
├── MANIFEST.json              # Pipeline metadata
├── DQ_REPORT.md              # Data quality report
├── logs/                     # Slurm job logs
│   ├── <job_id>_0.out
│   ├── <job_id>_0.err
│   └── ...
├── All_Beauty/
│   ├── joined.parquet        # Stage 2 output
│   ├── clean.parquet         # Stage 3 output
│   └── by_week/              # Stage 4 output
│       ├── _manifest.json
│       ├── week_2023-01-02.parquet
│       ├── week_2023-01-09.parquet
│       └── ...
├── Electronics/
│   └── ...
└── ... (21 more categories)
```

---

## Usage Examples

### Process Single Category

```bash
python amazon_unify_pipeline.py \
  --category All_Beauty \
  --work-dir ./work \
  --out-dir ./output \
  --threads 16
```

### Test Mode (Limited Rows)

```bash
python amazon_unify_pipeline.py \
  --category Electronics \
  --max-rows 100000 \
  --work-dir ./test_work \
  --out-dir ./test_output
```

### Read Weekly Partitions

```python
import pandas as pd

# Read single week
df = pd.read_parquet("output/All_Beauty/by_week/week_2023-01-02.parquet")

# Read all weeks
df_all = pd.read_parquet("output/All_Beauty/by_week/*.parquet")

# Filter by date range
import duckdb
con = duckdb.connect()
query = """
SELECT *
FROM read_parquet('output/All_Beauty/by_week/*.parquet')
WHERE week_start_utc BETWEEN '2023-01-01' AND '2023-06-30'
"""
df_filtered = con.execute(query).df()
```

---

## Slurm Configuration

### Job Array Details

- **Array Size:** 0-22 (23 categories)
- **CPUs per Task:** 32
- **Memory:** 64GB
- **Time Limit:** 24 hours
- **Partition:** main

### Customization

Edit `run_unify.slurm`:

```bash
#SBATCH --cpus-per-task=64     # More CPUs
#SBATCH --mem=128G             # More memory
#SBATCH --time=48:00:00        # Longer time
#SBATCH --partition=gpu        # Different partition
```

### Monitoring

```bash
# View queue
squeue -u $USER

# Cancel specific job
scancel <job_id>

# Cancel all jobs
scancel -u $USER

# View completed jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## Data Quality Checks

### Coverage Validation

Expected metadata coverage: **≥90%**

If coverage is low:
1. Check network connectivity to HuggingFace
2. Verify metadata files exist: `https://huggingface.co/.../raw_meta_{category}/*.parquet`
3. Try downloading metadata locally and update pipeline to use local path

### Missing Value Analysis

Review `DQ_REPORT.md` for:
- Null percentages per column
- Anomaly counts (invalid ratings, price outliers)
- Duplicate removal statistics

### Sample Inspection

```python
import pandas as pd

df = pd.read_parquet("output/All_Beauty/clean.parquet")

# Check first 10 rows
print(df.head(10))

# Check nulls
print(df.isnull().sum())

# Check value distributions
print(df['rating'].value_counts())
print(df['price'].describe())
```

---

## Troubleshooting

### Issue: Remote Metadata Access Fails

**Solution:** Download metadata manually and use local paths.

```bash
# Download metadata for category
wget https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw_meta_All_Beauty/0000.parquet \
  -O /scratch/$USER/amazon2023_stage/All_Beauty_metadata.parquet

# Pipeline will auto-detect local file
```

### Issue: Out of Memory

**Solution:** Reduce chunk size or increase Slurm memory allocation.

```bash
export ROWS_PER_CHUNK=500000  # Smaller chunks

# Or edit run_unify.slurm:
#SBATCH --mem=128G
```

### Issue: Timestamp Parsing Fails

**Symptom:** `timestamp_missing` count is high in DQ report.

**Solution:** Check data format and adjust normalization logic in `clean_data()`.

### Issue: Weekly Slicing Fails

**Symptom:** `by_week/` directory is empty.

**Solution:**
1. Check if AutoTS is installed: `pip install autots`
2. Pipeline will automatically fallback to pandas
3. Verify timestamps exist after cleaning

---

## Excluded Categories

The following categories are **automatically excluded** from processing:

- Movies_and_TV
- Software
- Subscription_Boxes
- Video_Games
- Magazine_Subscriptions
- Kindle_Store
- Gift_Cards
- Digital_Music
- CDs_and_Vinyl
- Books

To modify exclusions, edit `EXCLUDED_CATEGORIES` in `amazon_unify_pipeline.py` or `pipeline_config.yaml`.

---

## Performance Benchmarks

Approximate processing times (32 CPUs, 64GB RAM):

| Category           | Reviews   | Stage Time | Join Time | Clean Time | Weekly Time | Total   |
|--------------------|-----------|------------|-----------|------------|-------------|---------|
| All_Beauty         | ~10M      | 2h         | 30m       | 20m        | 15m         | ~3h     |
| Electronics        | ~20M      | 4h         | 1h        | 40m        | 30m         | ~6h     |
| Clothing_Shoes...  | ~15M      | 3h         | 45m       | 30m        | 20m         | ~4.5h   |

**Total Pipeline:** ~100-150 hours for all 23 categories (parallel execution)

**Wall Time with Slurm Array:** ~6-12 hours (depends on longest category)

---

## Manifest Schema

`MANIFEST.json` structure:

```json
{
  "pipeline_version": "1.0.0",
  "created_at": "2025-01-15T12:00:00Z",
  "parameters": {
    "work_dir": "/scratch/user/stage",
    "out_dir": "/scratch/user/output",
    "rows_per_chunk": 1500000,
    "threads": 32
  },
  "categories_processed": 23,
  "results": [
    {
      "category": "All_Beauty",
      "stage": { "status": "success", "total_rows": 10000000, ... },
      "join": { "status": "success", "coverage_percent": 95.2, ... },
      "clean": { "status": "success", "final_rows": 9950000, ... },
      "weekly": { "status": "success", "num_weeks": 520, ... }
    }
  ]
}
```

---

## Next Steps

After pipeline completes:

1. **Review DQ Report:** Check `DQ_REPORT.md` for coverage and quality metrics
2. **Validate Samples:** Inspect sample rows from each category
3. **Build Features:** Use weekly partitions for ML feature engineering
4. **Train Models:** Feed clean data into DarkHorse forecasting models
5. **Deploy Load Balancer:** Use predictions for dynamic resource allocation

---

## Support

For issues or questions:
- Review logs in `{OUT_DIR}/logs/`
- Check `DQ_REPORT.md` for data quality insights
- Examine `MANIFEST.json` for pipeline statistics
- Open issue on GitHub repository

---

## Citation

If using this pipeline, please cite the Amazon Reviews 2023 dataset:

```
@article{mcauley2023amazon,
  title={Amazon Reviews 2023},
  author={McAuley, Julian and others},
  journal={HuggingFace Datasets},
  year={2023}
}
```

---

**Built with ❤️ by the DarkHorse Team**
