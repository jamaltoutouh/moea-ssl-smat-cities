#!/usr/bin/env python3
"""
merge_results.py
================
Consolidates individual SLURM array run CSV files into a single consolidated CSV.
Computes and displays an aggregated performance summary table.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge per-run SLURM CSV files into a consolidated file.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the individual per-task CSV files (e.g. results_slurm/water/csv)")
    parser.add_argument("--pattern", type=str, default="*.csv",
                        help="Glob pattern for CSV files (default: '*.csv')")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output path for consolidated CSV (e.g. results/results_water.csv)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"No CSV files found in '{args.input_dir}' matching '{args.pattern}'")
        return

    print(f"Found {len(files)} result files in '{args.input_dir}'. Merging...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {f}: {e}")

    if not dfs:
        print("Error: No non-empty dataframes found.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Successfully saved consolidated {len(merged)} rows to: {args.out_csv}")


if __name__ == "__main__":
    main()
