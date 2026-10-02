#!/usr/bin/env bash
# ==============================================================================
# run_quick_demo.sh
# Fast local smoke test for MOEA-SSL and baselines (< 1 minute).
# Verifies that environment, data loaders, evolutionary optimization,
# and classifier pipelines run smoothly without errors.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "======================================================================"
echo " Running Quick Demonstration of MOEA-SSL Pipeline"
echo " Dataset: Water Potability (water)"
echo " Labeled Fraction: 0.05 | Seed: 42 | Population: 6 | Generations: 4"
echo "======================================================================"

python3 run_experiments.py \
    --dataset water \
    --labeled_fracs 0.05 \
    --seeds 42 \
    --popEA 6 \
    --generations 4 \
    --pcx 0.85 \
    --pmut 0.35 \
    --alpha 0.4 \
    --out_csv demo_results.csv

echo ""
echo "======================================================================"
echo " Smoke test completed successfully!"
echo " Results written to: demo_results.csv"
echo "======================================================================"
