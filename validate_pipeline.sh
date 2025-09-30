#!/bin/bash
# Validation script for Amazon Reviews 2023 Pipeline
# Tests with small dataset before full run

set -e

echo "========================================="
echo "Amazon Reviews Pipeline Validation"
echo "========================================="
echo ""

# Configuration
WORK_DIR="${WORK_DIR:-./test_work}"
OUT_DIR="${OUT_DIR:-./test_output}"
MAX_ROWS=500000
TEST_CATEGORIES=("All_Beauty" "Electronics")

# Clean previous test runs
if [ -d "${WORK_DIR}" ]; then
    echo "Cleaning previous test work directory..."
    rm -rf "${WORK_DIR}"
fi

if [ -d "${OUT_DIR}" ]; then
    echo "Cleaning previous test output directory..."
    rm -rf "${OUT_DIR}"
fi

mkdir -p "${WORK_DIR}" "${OUT_DIR}"

echo "Test configuration:"
echo "  Work dir: ${WORK_DIR}"
echo "  Output dir: ${OUT_DIR}"
echo "  Max rows: ${MAX_ROWS}"
echo "  Test categories: ${TEST_CATEGORIES[@]}"
echo ""

# Test each category
for CATEGORY in "${TEST_CATEGORIES[@]}"; do
    echo "========================================="
    echo "Testing category: ${CATEGORY}"
    echo "========================================="
    echo ""

    python amazon_unify_pipeline.py \
        --category "${CATEGORY}" \
        --work-dir "${WORK_DIR}" \
        --out-dir "${OUT_DIR}" \
        --max-rows ${MAX_ROWS} \
        --threads 4

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ ${CATEGORY} completed successfully"
        echo ""
    else
        echo ""
        echo "✗ ${CATEGORY} FAILED"
        echo ""
        exit 1
    fi
done

# Generate DQ report
echo "========================================="
echo "Generating Data Quality Report"
echo "========================================="
echo ""

python generate_dq_report.py --out-dir "${OUT_DIR}"

if [ -f "${OUT_DIR}/DQ_REPORT.md" ]; then
    echo ""
    echo "✓ DQ Report generated successfully"
    echo ""
    echo "Contents:"
    head -n 30 "${OUT_DIR}/DQ_REPORT.md"
else
    echo "✗ DQ Report generation failed"
    exit 1
fi

# Verify outputs
echo ""
echo "========================================="
echo "Verifying Outputs"
echo "========================================="
echo ""

for CATEGORY in "${TEST_CATEGORIES[@]}"; do
    echo "Checking ${CATEGORY}..."

    JOINED="${OUT_DIR}/${CATEGORY}/joined.parquet"
    CLEAN="${OUT_DIR}/${CATEGORY}/clean.parquet"
    WEEKLY="${OUT_DIR}/${CATEGORY}/by_week"

    if [ -f "${JOINED}" ]; then
        SIZE=$(du -h "${JOINED}" | cut -f1)
        echo "  ✓ joined.parquet (${SIZE})"
    else
        echo "  ✗ joined.parquet MISSING"
    fi

    if [ -f "${CLEAN}" ]; then
        SIZE=$(du -h "${CLEAN}" | cut -f1)
        echo "  ✓ clean.parquet (${SIZE})"
    else
        echo "  ✗ clean.parquet MISSING"
    fi

    if [ -d "${WEEKLY}" ]; then
        COUNT=$(find "${WEEKLY}" -name "*.parquet" | wc -l)
        echo "  ✓ by_week/ (${COUNT} weekly partitions)"
    else
        echo "  ✗ by_week/ MISSING"
    fi

    echo ""
done

# Check manifest
if [ -f "${OUT_DIR}/MANIFEST.json" ]; then
    echo "✓ MANIFEST.json exists"
    echo ""
    echo "Manifest summary:"
    python -c "import json; m=json.load(open('${OUT_DIR}/MANIFEST.json')); print(f'  Categories: {m[\"categories_processed\"]}'); print(f'  Created: {m[\"created_at\"]}')"
else
    echo "✗ MANIFEST.json MISSING"
fi

echo ""
echo "========================================="
echo "Validation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Review DQ_REPORT.md: ${OUT_DIR}/DQ_REPORT.md"
echo "  2. Check sample data in ${OUT_DIR}/"
echo "  3. If validation passes, run full pipeline:"
echo "     sbatch run_unify.slurm"
echo ""
