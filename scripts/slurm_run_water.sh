#!/usr/bin/env bash
#SBATCH --job-name=moea_ssl_water
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-44%15

set -euo pipefail

# ----------------- ENVIRONMENT SETUP -----------------
# Specify your Python executable or use the active Python environment:
PY="${PYTHON_EXEC:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT="${PROJECT_DIR}/run_experiments.py"

# ---------------- EXPERIMENT PARAMETERS ----------------
DATASET="water"
POP_EA=36
GENERATIONS=50
PCX=0.85
PMUT=0.35
ALPHA=0.4

# ----------------- PARAMETER MAPPING -----------------
# 3 fractions x 15 runs = 45 tasks (array 0-44)
LFS=(0.01 0.05 0.10)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

N_LF=${#LFS[@]}
N_SEED=${#SEEDS[@]}

TASK=${SLURM_ARRAY_TASK_ID}
LF_IDX=$(( TASK / N_SEED ))
SEED_IDX=$(( TASK % N_SEED ))

LF=${LFS[$LF_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# ----------------- PATH MANAGEMENT -------------------
OUT_DIR="${PROJECT_DIR}/results_slurm/${DATASET}/csv"
mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/slurm_logs" "${PROJECT_DIR}/history"
OUT_CSV="${OUT_DIR}/result_${DATASET}_lf${LF}_s${SEED}.csv"

cd "${PROJECT_DIR}"

echo "======================================================="
echo "SLURM Job: ${SLURM_JOB_ID} | Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname) | Date: $(date)"
echo "Dataset: ${DATASET} | Labeled Fraction: ${LF} | Seed: ${SEED}"
echo "Config: popEA=${POP_EA}, generations=${GENERATIONS}, pcx=${PCX}, pmut=${PMUT}, alpha=${ALPHA}"
echo "Output: ${OUT_CSV}"
echo "======================================================="

exec "$PY" "$SCRIPT" \
    --dataset "$DATASET" \
    --labeled_fracs "$LF" \
    --seeds "$SEED" \
    --popEA "$POP_EA" \
    --generations "$GENERATIONS" \
    --pcx "$PCX" \
    --pmut "$PMUT" \
    --alpha "$ALPHA" \
    --out_csv "$OUT_CSV"
