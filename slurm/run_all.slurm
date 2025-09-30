#!/bin/bash
#SBATCH --job-name=master_pipeline
#SBATCH --output=/home1/yliu0158/amazon2023/amazon23/logs/master_pipeline_%A.out
#SBATCH --error=/home1/yliu0158/amazon2023/amazon23/logs/master_pipeline_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=main

# ============================================================================
# MASTER TRAINING PIPELINE - Orchestrates all model training jobs
# ============================================================================
# This script submits jobs in sequence with dependencies to ensure
# efficient resource usage and proper data flow between stages.
# ============================================================================

set -e
set -u

SLURM_DIR="/home1/yliu0158/amazon2023/csci653-as01/slurm"
OUT_DIR="/home1/yliu0158/amazon2023/amazon23"

echo "========================================"
echo "MASTER TRAINING PIPELINE"
echo "========================================"
echo "Started at: $(date)"
echo "Output directory: ${OUT_DIR}"
echo ""

# ============================================================================
# STAGE 1: Data Preparation (Required for all models)
# ============================================================================
echo "STAGE 1: Submitting data preparation job..."
JOB1=$(sbatch --parsable "${SLURM_DIR}/01_data_preparation/prepare_data.slurm")
echo "  Job ID: ${JOB1}"
echo ""

# ============================================================================
# STAGE 2: Baseline Models (Run in parallel after Stage 1)
# ============================================================================
echo "STAGE 2: Submitting baseline model training..."
JOB2A=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/02_baseline_models/train_transformer.slurm")
JOB2B=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/02_baseline_models/train_autots.slurm")
echo "  Transformer Job ID: ${JOB2A}"
echo "  AutoTS Job ID: ${JOB2B}"
echo ""

# ============================================================================
# STAGE 3: Advanced Deep Learning Models (Run in parallel after Stage 1)
# ============================================================================
echo "STAGE 3: Submitting advanced model training..."
JOB3A=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/03_advanced_models/train_bert_enhanced.slurm")
JOB3B=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/03_advanced_models/train_transformer_bert.slurm")
JOB3C=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/03_advanced_models/train_gnn_multitask.slurm")
JOB3D=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/03_advanced_models/train_ultimate.slurm")
echo "  BERT Enhanced Job ID: ${JOB3A}"
echo "  Transformer+BERT Job ID: ${JOB3B}"
echo "  GNN Multi-task Job ID: ${JOB3C}"
echo "  Ultimate Model Job ID: ${JOB3D}"
echo ""

# ============================================================================
# STAGE 4: Boosting Models (Run in parallel after Stage 1)
# ============================================================================
echo "STAGE 4: Submitting boosting model training..."
JOB4A=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/04_boosting_models/train_xgboost.slurm")
JOB4B=$(sbatch --parsable --dependency=afterok:${JOB1} "${SLURM_DIR}/04_boosting_models/train_lightgbm.slurm")
echo "  XGBoost Job ID: ${JOB4A}"
echo "  LightGBM Job ID: ${JOB4B}"
echo ""

# ============================================================================
# STAGE 5: Ensemble (Wait for all models to complete)
# ============================================================================
echo "STAGE 5: Submitting ensemble training..."
DEPS="${JOB2A}:${JOB2B}:${JOB3A}:${JOB3B}:${JOB3C}:${JOB3D}:${JOB4A}:${JOB4B}"
JOB5=$(sbatch --parsable --dependency=afterok:${DEPS} "${SLURM_DIR}/05_ensemble/train_ensemble.slurm")
echo "  Ensemble Job ID: ${JOB5}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================"
echo "PIPELINE SUBMITTED SUCCESSFULLY"
echo "========================================"
echo ""
echo "Job Dependency Chain:"
echo "  Stage 1 (Data): ${JOB1}"
echo "  Stage 2 (Baseline): ${JOB2A}, ${JOB2B}"
echo "  Stage 3 (Advanced): ${JOB3A}, ${JOB3B}, ${JOB3C}, ${JOB3D}"
echo "  Stage 4 (Boosting): ${JOB4A}, ${JOB4B}"
echo "  Stage 5 (Ensemble): ${JOB5}"
echo ""
echo "Monitor with: squeue -u yliu0158"
echo "Cancel all with: scancel -u yliu0158"
echo ""
echo "Submitted at: $(date)"