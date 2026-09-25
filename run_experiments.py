#!/usr/bin/env python3
"""
run_experiments.py
==================
Main experimental execution runner for:
"Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification"
by Francisco José Sedeño and Jamal Toutouh (ITIS Software, University of Malaga).

Orchestrates independent runs across:
- MOEA-SSL (Bi-objective NSGA-II: RobustMacroF1 and FeatureRatio)
- EA-SSL (Mono-objective scalar evolutionary search baseline)
- Lightweight SSL baselines (Self-Training, Heuristic Co-Training, Label Spreading)
- Supervised reference classifiers (Logistic Regression, Linear SVM, Random Forest, HistGradientBoosting)

Usage Example:
--------------
# Fast local test (smoke run):
python run_experiments.py --dataset water --runs 1 --popEA 6 --generations 5 --out_csv test_run.csv

# Standard paper settings on a single dataset:
python run_experiments.py --dataset grid --runs 15 --popEA 36 --generations 50 --pcx 0.85 --pmut 0.35
"""

import os
import argparse
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd

import ea_ssl
try:
    import moea_ssl
except ImportError:
    import moea2f_ssl as moea_ssl
import baseline_supervised


def main():
    ap = argparse.ArgumentParser(
        description="Run benchmark experiments for MOEA-SSL, EA-SSL, and baseline classifiers."
    )
    ap.add_argument("--dataset", type=str, required=True, choices=["building", "grid", "water"],
                    help="Dataset name: 'building' (Building Occupancy), 'grid' (Grid Stability), or 'water' (Water Potability)")
    ap.add_argument("--labeled_fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10],
                    help="Fraction(s) of labeled training data (default: 0.01 0.05 0.10)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Specific random seed(s) to execute. If specified, overrides --runs.")
    ap.add_argument("--runs", type=int, default=5,
                    help="Number of independent runs (seeds 0 to runs-1). Default is 5.")
    ap.add_argument("--popEA", type=int, default=36,
                    help="Population size N (default: 36, as in paper)")
    ap.add_argument("--generations", type=int, default=50,
                    help="Number of generations G (default: 50, as in paper)")
    ap.add_argument("--pcx", type=float, default=0.85,
                    help="Crossover probability (default: 0.85, as in paper Section 5.1)")
    ap.add_argument("--pmut", type=float, default=0.35,
                    help="Mutation probability (default: 0.35, as in paper Section 5.1)")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Robustness weight alpha for RobustMacroF1 (default: 0.4, as in paper Eq. 2)")
    ap.add_argument("--max_u_fit", type=int, default=8000,
                    help="Max unlabeled samples during evolutionary fitness evaluation")
    ap.add_argument("--max_u_final", type=int, default=20000,
                    help="Max unlabeled samples during final model training")
    ap.add_argument("--probe_size", type=int, default=120,
                    help="Size of probe validation split")
    ap.add_argument("--label_spreading_max_total", type=int, default=4000,
                    help="Max dataset size for Label Spreading graph construction")
    ap.add_argument("--skip_supervised", action="store_true", default=False,
                    help="Skip training supervised baselines")
    ap.add_argument("--skip_ea", action="store_true", default=False,
                    help="Skip running EA-SSL (run only MOEA-SSL and baselines)")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Output CSV path. Defaults to 'results_{dataset}.csv'")

    # EA-SSL specific fitness weights
    ap.add_argument("--lam_std", type=float, default=0.20)
    ap.add_argument("--lam_bias", type=float, default=0.70)
    ap.add_argument("--lam_added", type=float, default=0.0005)

    args = ap.parse_args()

    # Determine seeds
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.runs))

    if args.out_csv is None:
        args.out_csv = f"results_{args.dataset}.csv"

    # Load dataset
    dataset_name, X_tr, y_tr, X_te, y_te, columns = ea_ssl.load_dataset(args.dataset)
    cfg = vars(args)
    results = []

    print("=" * 70)
    print(f"Benchmark Experiment: {dataset_name}")
    print(f"Total Features (d): {len(columns)}")
    print(f"Train Pool: {X_tr.shape[0]} samples, Held-out Test Pool: {X_te.shape[0]} samples")
    print(f"Labeled Fractions: {args.labeled_fracs}")
    print(f"Seeds: {seeds}")
    print(f"Evolutionary Config: pop={args.popEA}, gen={args.generations}, pcx={args.pcx}, pmut={args.pmut}, alpha={args.alpha}")
    print(f"Target Output CSV: {args.out_csv}")
    print("=" * 70 + "\n")

    baseline_methods = {"Self-training", "Heuristic co-training", "Label Spreading", "Label Spreading (skipped)"}

    for lf in args.labeled_fracs:
        for seed in seeds:
            print(f"\n>>> Running: Labeled Fraction = {lf} ({int(lf*100)}%), Seed = {seed} <<<")

            # 1. Run EA-SSL (or extract baselines if EA skipped)
            ea_rows = []
            if not args.skip_ea:
                print("-> Running EA-SSL & Lightweight SSL Baselines...")
                t0 = time.time()
                ea_rows = ea_ssl.run_setting(
                    dataset_name=dataset_name,
                    X_tr=X_tr, y_tr=y_tr,
                    X_te=X_te, y_te=y_te,
                    columns=columns,
                    labeled_frac=lf,
                    seed=seed,
                    cfg=cfg
                )
                print(f"   EA-SSL completed in {time.time() - t0:.1f}s.")

            # 2. Run MOEA-SSL
            print("-> Running MOEA-SSL (NSGA-II Bi-Objective)...")
            t0 = time.time()
            moea_rows = moea_ssl.run_setting(
                dataset_name=dataset_name,
                X_tr=X_tr, y_tr=y_tr,
                X_te=X_te, y_te=y_te,
                columns=columns,
                labeled_frac=lf,
                seed=seed,
                cfg=cfg
            )
            print(f"   MOEA-SSL completed in {time.time() - t0:.1f}s.")

            # Filter out baseline runs from MOEA-SSL results to avoid duplication if EA already ran them
            if not args.skip_ea:
                filtered_moea_rows = [
                    row for row in moea_rows
                    if row.get("method") not in baseline_methods
                ]
            else:
                filtered_moea_rows = moea_rows

            # 3. Run Supervised Reference Baselines
            sup_rows = []
            if not args.skip_supervised:
                print("-> Running Supervised Reference Baselines...")
                t0 = time.time()
                sup_rows = baseline_supervised.run_setting(
                    dataset_name=dataset_name,
                    X_tr=X_tr, y_tr=y_tr,
                    X_te=X_te, y_te=y_te,
                    columns=columns,
                    labeled_frac=lf,
                    seed=seed,
                    cfg=cfg
                )
                print(f"   Supervised baselines completed in {time.time() - t0:.1f}s.")

            # Combine all records
            combined = ea_rows + filtered_moea_rows + sup_rows
            results.extend(combined)

            # Save progress incrementally to avoid data loss
            df_out = pd.DataFrame(results)
            out_dir = os.path.dirname(args.out_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df_out.to_csv(args.out_csv, index=False)
            print(f"   Progress saved: {len(df_out)} rows in {args.out_csv}")

    print("\n" + "=" * 70)
    print(f"Experiments finished! Consolidated {len(results)} rows to: {args.out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
