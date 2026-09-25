#!/usr/bin/env python3
"""
baseline_supervised.py
Supervised Learning Baselines for Smart Cities Benchmark Datasets.
Trains standard supervised models strictly on the labeled data partition (X_L, y_L)
matching the SSL data splitting protocol (same splits and fractions as ea_ssl.py).

Models evaluated:
1. Regularized Logistic Regression
2. Linear SVM
3. Random Forest
4. Histogram-based Gradient Boosting

Outputs results to a CSV file matching the schema of ea_ssl.py, cc_ssl.py, and moea_ssl.py.
"""

import os
import argparse
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from src.dataset_loaders import load_dataset


# -----------------------------
# Basic metrics
# -----------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


# -----------------------------
# Preprocessing Pipeline Builder
# -----------------------------
def build_preprocessor_for_columns(X: pd.DataFrame, cols: List[str]) -> ColumnTransformer:
    X_sub = X[cols]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X_sub[c])]
    cat_cols = [c for c in cols if c not in num_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# -----------------------------
# Data Splitting Function (Identical to ea_ssl.py)
# -----------------------------
def make_local_splits(
    X_train_full: pd.DataFrame, y_train_full: np.ndarray,
    X_test_full: pd.DataFrame, y_test_full: np.ndarray,
    labeled_frac: float,
    seed: int,
    val_frac_of_train: float = 0.20,
    probe_size: int = 120,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_train_full = np.asarray(y_train_full)

    # 1. Split train pool into pool and validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_train, random_state=seed)
    pool_idx, val_idx = next(sss.split(X_train_full, y_train_full))
    X_pool = X_train_full.iloc[pool_idx].reset_index(drop=True)
    y_pool = y_train_full[pool_idx]
    X_V = X_train_full.iloc[val_idx].reset_index(drop=True)
    y_V = y_train_full[val_idx]

    n_pool = X_pool.shape[0]
    n_classes = int(np.unique(y_train_full).size)
    min_labeled = 2 * n_classes
    n_L = max(min_labeled, int(round(n_pool * labeled_frac)))

    min_unlabeled = 30
    if n_pool < (min_labeled + min_unlabeled + 1):
        min_unlabeled = max(10, n_pool // 5)
    n_L = min(n_L, max(min_labeled, n_pool - min_unlabeled))

    # Stratified labeled selection
    classes = np.unique(y_pool)
    min_per_class = 2
    min_needed = int(classes.size * min_per_class)
    if n_L < min_needed:
        n_L = min_needed

    labeled_parts = []
    all_idx = np.arange(n_pool)

    for c in classes:
        idx_c = np.where(y_pool == c)[0]
        if idx_c.size == 0:
            continue
        take_c = min(min_per_class, idx_c.size)
        labeled_parts.append(rng.choice(idx_c, size=take_c, replace=False))

    if labeled_parts:
        L_idx = np.unique(np.concatenate(labeled_parts))
    else:
        L_idx = np.array([], dtype=int)

    remaining = np.setdiff1d(all_idx, L_idx, assume_unique=False)
    need = int(n_L - L_idx.size)
    if need > 0 and remaining.size > 0:
        if need > remaining.size:
            need = remaining.size
        extra = rng.choice(remaining, size=need, replace=False)
        L_idx = np.concatenate([L_idx, extra])

    rng.shuffle(L_idx)
    rest = np.setdiff1d(all_idx, L_idx, assume_unique=False)
    rng.shuffle(rest)

    remaining = rest.size
    n_P = min(probe_size, max(0, remaining - min_unlabeled))
    P_idx = rest[:n_P] if n_P > 0 else np.array([], dtype=int)
    U_idx = rest[n_P:]

    X_L = X_pool.iloc[L_idx].reset_index(drop=True)
    y_L = y_pool[L_idx]
    X_P = X_pool.iloc[P_idx].reset_index(drop=True) if n_P > 0 else None
    y_P = y_pool[P_idx] if n_P > 0 else None
    X_U = X_pool.iloc[U_idx].reset_index(drop=True)

    return dict(
        X_L=X_L, y_L=y_L,
        X_U=X_U,
        X_V=X_V, y_V=y_V,
        X_P=X_P, y_P=y_P,
        X_T=X_test_full, y_T=y_test_full
    )


# -----------------------------
# Supervised Models Factory
# -----------------------------
def get_supervised_models(seed: int, n_samples_labeled: int) -> List[Tuple[str, Any]]:
    """
    Returns the four supervised baseline models with consistent hyper-parameters.
    """
    # For HistGradientBoosting, adapt min_samples_leaf if labeled sample size is small
    msl = min(20, max(1, n_samples_labeled // 4))

    return [
        (
            "Regularized Logistic Regression",
            LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
        ),
        (
            "Linear SVM",
            LinearSVC(C=1.0, penalty="l2", max_iter=3000, random_state=seed)
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, random_state=seed)
        ),
        (
            "Histogram-based Gradient Boosting",
            HistGradientBoostingClassifier(random_state=seed, min_samples_leaf=msl)
        ),
    ]


# -----------------------------
# Run Single Configuration Setting
# -----------------------------
def run_setting(
    dataset_name: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    columns: List[str],
    labeled_frac: float,
    seed: int,
    cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Executes all supervised models for a given dataset, labeled fraction, and random seed.
    Models are trained STRICTLY on X_L, y_L.
    """
    probe_size = cfg.get("probe_size", 120)
    splits = make_local_splits(X_tr, y_tr, X_te, y_te, labeled_frac=labeled_frac, seed=seed, probe_size=probe_size)

    X_L = splits["X_L"]
    y_L = splits["y_L"]
    X_T = splits["X_T"]
    y_T = splits["y_T"]

    nL = int(X_L.shape[0])
    nU = int(splits["X_U"].shape[0])
    nP = int(0 if splits["X_P"] is None else splits["X_P"].shape[0])
    nV = int(splits["X_V"].shape[0])
    nT = int(X_T.shape[0])

    models = get_supervised_models(seed=seed, n_samples_labeled=nL)
    rows = []

    for model_name, clf in models:
        pre = build_preprocessor_for_columns(X_L, columns)
        pipe = Pipeline([
            ("prep", pre),
            ("clf", clf)
        ])

        t_start = time.time()
        try:
            pipe.fit(X_L[columns], y_L)
            y_pred = pipe.predict(X_T[columns])
            run_time = time.time() - t_start

            f1_val = macro_f1(y_T, y_pred)
            acc_val = acc(y_T, y_pred)
            err_msg = None
        except Exception as e:
            run_time = time.time() - t_start
            f1_val = np.nan
            acc_val = np.nan
            err_msg = str(e)

        row = dict(
            dataset=dataset_name,
            labeled_frac=float(labeled_frac),
            seed=int(seed),
            method=model_name,
            macroF1_test=float(f1_val),
            acc_test=float(acc_val),
            run_time_sec=float(round(run_time, 4)),
            n_features=int(len(columns)),
            n_classes=int(np.unique(y_te).size),
            n_labeled=nL,
            n_unlabeled=nU,
            n_probe=nP,
            n_val=nV,
            n_test=nT,
        )
        if err_msg is not None:
            row["error"] = err_msg

        rows.append(row)

    return rows


# -----------------------------
# Main Executable
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run supervised baselines on smart cities benchmark datasets.")
    ap.add_argument("--dataset", type=str, required=True, choices=["building", "room", "water", "grid"],
                    help="Dataset name: building, room, water, or grid")
    ap.add_argument("--labeled_fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10],
                    help="Fraction of labeled training data")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Specific seeds to run. Overrides --runs if provided.")
    ap.add_argument("--runs", type=int, default=5,
                    help="Number of independent runs (seeds 0 to runs-1). Default is 5.")
    ap.add_argument("--probe_size", type=int, default=120,
                    help="Probe set size for split consistency with SSL runners.")
    ap.add_argument("--out_csv", type=str, default="results_baseline_supervised.csv",
                    help="Output CSV path for results.")

    args = ap.parse_args()

    # Determine seeds to run
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.runs))

    # Load dataset
    dataset_name, X_tr, y_tr, X_te, y_te, columns = load_dataset(args.dataset)

    cfg = vars(args)
    results = []

    print("==========================================")
    print(f"Supervised Baselines Benchmark: {dataset_name}")
    print(f"Features: {len(columns)}")
    print(f"Train Pool Size: {X_tr.shape[0]}, Test Pool Size: {X_te.shape[0]}")
    print(f"Seeds: {seeds}")
    print(f"Labeled Fractions: {args.labeled_fracs}")
    print("==========================================\n")

    for lf in args.labeled_fracs:
        for seed in seeds:
            print(f"--- Running: Labeled Frac={lf}, Seed={seed} ---")
            t0 = time.time()
            rows = run_setting(
                dataset_name=dataset_name,
                X_tr=X_tr, y_tr=y_tr,
                X_te=X_te, y_te=y_te,
                columns=columns,
                labeled_frac=lf,
                seed=seed,
                cfg=cfg
            )
            elapsed = time.time() - t0
            print(f"Setting completed in {elapsed:.2f}s.")
            results.extend(rows)

            for row in rows:
                m_name = row["method"]
                f1 = row["macroF1_test"]
                acc_val = row["acc_test"]
                rt = row["run_time_sec"]
                print(f"  {m_name:<35} -> Test F1: {f1:.4f}, Test Acc: {acc_val:.4f} (Time: {rt:.4f}s)")
            print()

    # Convert results to DataFrame and save
    df_out = pd.DataFrame(results)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    print("==========================================")
    print(f"All supervised benchmarks completed! Results saved to: {args.out_csv}")
    print("==========================================")

    # Aggregated performance summary table
    summary_cols = ["labeled_frac", "method", "macroF1_test", "acc_test", "run_time_sec"]
    metric_cols = ["macroF1_test", "acc_test", "run_time_sec"]
    if all(c in df_out.columns for c in summary_cols):
        grouped = df_out.groupby(["labeled_frac", "method"])[metric_cols].agg(["mean", "std"])
        print("\nAggregated Performance Summary (Mean +/- Std Dev):")
        print(grouped.to_string())


if __name__ == "__main__":
    main()
