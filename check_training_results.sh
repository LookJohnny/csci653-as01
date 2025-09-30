#!/bin/bash
# Script to check training results after job completes

echo "=========================================="
echo "DarkHorse Training Results Checker"
echo "=========================================="
echo ""

# Check if job is still running
echo "1. Checking job status..."
RUNNING=$(squeue -u $USER | grep train_da | wc -l)
if [ $RUNNING -gt 0 ]; then
    echo "   ⏳ Job is still RUNNING:"
    squeue -u $USER | grep train_da
    echo ""
    echo "   Check progress with:"
    echo "   tail -f /home1/yliu0158/amazon2023/amazon23/logs/train_3012925.out"
else
    echo "   ✓ No training jobs currently running"
fi
echo ""

# Check last job status
echo "2. Last training job status:"
sacct -u $USER -n train_darkhorse --format=JobID,JobName,State,Elapsed,MaxRSS -X | tail -5
echo ""

# Check output files
echo "3. Generated files:"
echo ""
echo "   Combined dataset:"
ls -lh /home1/yliu0158/amazon2023/amazon23/combined_reviews.csv 2>/dev/null || echo "   ✗ Not found"
echo ""

echo "   Training output directory:"
if [ -d "/home1/yliu0158/amazon2023/amazon23/training_output" ]; then
    ls -lh /home1/yliu0158/amazon2023/amazon23/training_output/
else
    echo "   ✗ Directory not found"
fi
echo ""

# Check for weekly panel
echo "   Weekly panel dataset:"
ls -lh /home1/yliu0158/amazon2023/amazon23/training_output/weekly_panel.csv 2>/dev/null || echo "   ✗ Not found yet"
echo ""

# Check for trained models
echo "   Transformer model:"
if [ -d "/home1/yliu0158/amazon2023/amazon23/training_output/transformer_model" ]; then
    echo "   ✓ Found:"
    ls -lh /home1/yliu0158/amazon2023/amazon23/training_output/transformer_model/
else
    echo "   ✗ Not trained yet"
fi
echo ""

# Check for forecast output
echo "   Forecast results:"
if [ -d "/home1/yliu0158/amazon2023/amazon23/training_output/forecast_output" ]; then
    echo "   ✓ Found:"
    ls -lh /home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/
else
    echo "   ✗ Not generated yet"
fi
echo ""

# Show recent log output
echo "4. Recent training log (last 30 lines):"
echo "=========================================="
tail -30 /home1/yliu0158/amazon2023/amazon23/logs/train_*.out 2>/dev/null | tail -30
echo ""

# Show errors if any
echo "5. Recent errors (if any):"
echo "=========================================="
ERRORS=$(tail -20 /home1/yliu0158/amazon2023/amazon23/logs/train_*.err 2>/dev/null | grep -v "MODULEPATH" | grep -v "^$")
if [ -z "$ERRORS" ]; then
    echo "   ✓ No errors found"
else
    echo "$ERRORS"
fi
echo ""

echo "=========================================="
echo "Summary Commands:"
echo "=========================================="
echo ""
echo "# Monitor live progress:"
echo "tail -f /home1/yliu0158/amazon2023/amazon23/logs/train_3012925.out"
echo ""
echo "# Check job status:"
echo "squeue -u \$USER"
echo ""
echo "# View all output files:"
echo "ls -lhR /home1/yliu0158/amazon2023/amazon23/training_output/"
echo ""
echo "# Check forecast metrics:"
echo "cat /home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/metrics.json"
echo ""
echo "# View predictions:"
echo "head -20 /home1/yliu0158/amazon2023/amazon23/training_output/forecast_output/pred_blend_val.csv"
echo ""