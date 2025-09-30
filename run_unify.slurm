#!/bin/bash
#SBATCH --job-name=amazon_unify
#SBATCH --output=/home1/%u/amazon2023/amazon23/logs/%A_%a.out
#SBATCH --error=/home1/%u/amazon2023/amazon23/logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=main
#SBATCH --array=0-22  # 23 categories

set -e
set -u
set -o pipefail

WORK_DIR="${WORK_DIR:-/home1/$USER/amazon2023/amazon2023_stage}"
OUT_DIR="${OUT_DIR:-/home1/$USER/amazon2023/amazon23}"
LOG_DIR="${OUT_DIR}/logs"

ROWS_PER_CHUNK="${ROWS_PER_CHUNK:-1500000}"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
MAX_ROWS="${MAX_ROWS:-}"

SCRIPT_DIR="/home1/yliu0158/amazon2023/csci653-as01"
VENV_DIR="${SCRIPT_DIR}/venv"
PIPELINE_SCRIPT="${SCRIPT_DIR}/amazon_unify_pipeline.py"

CATEGORIES=(
    "All_Beauty"
    "Amazon_Fashion"
    "Appliances"
    "Arts_Crafts_and_Sewing"
    "Automotive"
    "Baby_Products"
    "Beauty_and_Personal_Care"
    "Cell_Phones_and_Accessories"
    "Clothing_Shoes_and_Jewelry"
    "Electronics"
    "Grocery_and_Gourmet_Food"
    "Handmade_Products"
    "Health_and_Household"
    "Health_and_Personal_Care"
    "Home_and_Kitchen"
    "Industrial_and_Scientific"
    "Musical_Instruments"
    "Office_Products"
    "Patio_Lawn_and_Garden"
    "Pet_Supplies"
    "Sports_and_Outdoors"
    "Tools_and_Home_Improvement"
    "Toys_and_Games"
)

CATEGORY="${CATEGORIES[${SLURM_ARRAY_TASK_ID}]}"

echo "Processing category: ${CATEGORY}"

mkdir -p "${WORK_DIR}" "${OUT_DIR}" "${LOG_DIR}"

module purge
module load gcc/12.3.0 python/3.10.16

source "${VENV_DIR}/bin/activate"

CMD="python ${PIPELINE_SCRIPT} --category ${CATEGORY} --work-dir ${WORK_DIR} --out-dir ${OUT_DIR} --rows-per-chunk ${ROWS_PER_CHUNK} --threads ${THREADS}"

if [ -n "${MAX_ROWS}" ]; then
    CMD="${CMD} --max-rows ${MAX_ROWS}"
fi

MAX_RETRIES=3
RETRY_COUNT=0

while [ ${RETRY_COUNT} -lt ${MAX_RETRIES} ]; do
    if ${CMD}; then
        echo "SUCCESS: ${CATEGORY}"
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        [ ${RETRY_COUNT} -lt ${MAX_RETRIES} ] && sleep $((RETRY_COUNT * 60))
    fi
done

echo "FAILED: ${CATEGORY}"
exit 1
