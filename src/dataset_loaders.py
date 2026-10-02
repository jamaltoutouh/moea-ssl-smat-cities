"""
Dataset Loaders for Smart City Semi-Supervised Learning Benchmarks.

This module provides data loading and preprocessing routines for the three
Smart City classification datasets evaluated in:
"Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification"
by Francisco José Sedeño and Jamal Toutouh (ITIS Software, University of Malaga).

Datasets:
1. Building Occupancy Detection (Candanedo, 2016) [7 features, binary]
2. Electric Grid Stability (Arzamasov, 2018) [12 features, binary]
3. Water Potability (Kadiwal, 2021) [9 features, binary]
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def get_datasets_base_dir() -> str:
    """
    Locates the 'datasets' directory reliably across different execution contexts
    without relying on machine-specific absolute paths.
    """
    # 1. Environment variable override
    env_dir = os.environ.get("SSL_DATASETS_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir

    # 2. Relative to this file's location (src/../datasets)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidate1 = os.path.normpath(os.path.join(this_dir, "..", "datasets"))
    if os.path.exists(candidate1):
        return candidate1

    # 3. Relative to current working directory
    candidate2 = os.path.abspath("datasets")
    if os.path.exists(candidate2):
        return candidate2

    # 4. Fallback search upwards
    parent = this_dir
    for _ in range(3):
        cand = os.path.join(parent, "datasets")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(parent)

    raise FileNotFoundError(
        "Could not find 'datasets' directory. Please set SSL_DATASETS_DIR environment variable "
        "or ensure the datasets folder is located at the project root."
    )


def load_and_preprocess_building_occupancy() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    """
    Loads and preprocesses the Building Occupancy Detection dataset (Candanedo, 2016).
    Uses the repository training partition (datatraining.txt) and first held-out
    test partition (datatest.txt), yielding 10,808 observations and 7 processed predictors.

    Returns:
        X_tr (pd.DataFrame): Training pool features (8,143 samples)
        y_tr (np.ndarray): Training labels (0 = Unoccupied, 1 = Occupied)
        X_te (pd.DataFrame): Held-out test partition features (2,665 samples)
        y_te (np.ndarray): Held-out test labels
        columns (List[str]): List of predictor names (7 features)
    """
    base_dir = get_datasets_base_dir()
    train_path = os.path.join(base_dir, "building_occupancy_detection", "datatraining.txt")
    test_path = os.path.join(base_dir, "building_occupancy_detection", "datatest.txt")

    def process_file(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df['Hour'] = df['date'].dt.hour
        df['IsWeekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        df = df.drop(columns=['date'])

        # Drop index column if present
        valid_cols = {'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy', 'Hour', 'IsWeekend'}
        if df.columns[0] not in valid_cols:
            df = df.drop(columns=[df.columns[0]])

        X = df.drop(columns=['Occupancy'])
        y = df['Occupancy'].to_numpy(dtype=int)
        return X, y

    X_tr, y_tr = process_file(train_path)
    X_te, y_te = process_file(test_path)
    return X_tr, y_tr, X_te, y_te, list(X_tr.columns)


def load_and_preprocess_electric_grid_stability(
    test_size: float = 0.30, seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    """
    Loads and preprocesses the Decentralized Electrical Grid Stability dataset (Arzamasov, 2018).
    10,000 observations with 12 predictors (reaction delays tau, power flows p, price elasticities g).
    The continuous root 'stab' is excluded to prevent label leakage; categorical 'stabf' is binary target.
    Stratified 70/30 train/test split.

    Returns:
        X_tr (pd.DataFrame): Training development partition (7,000 samples)
        y_tr (np.ndarray): Binary stability labels (1 = Stable, 0 = Unstable)
        X_te (pd.DataFrame): Held-out test partition (3,000 samples)
        y_te (np.ndarray): Held-out test labels
        columns (List[str]): List of predictor names (12 features)
    """
    base_dir = get_datasets_base_dir()
    csv_path = os.path.join(base_dir, "electric_grid_stability", "Data_for_UCI_named.csv")
    df = pd.read_csv(csv_path)

    # Exclude continuous eigenvalue 'stab' to avoid target leakage; keep 'stabf' as binary classification
    X = df.drop(columns=['stab', 'stabf'])
    y = (df['stabf'] == 'stable').astype(int).to_numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(sss.split(X, y))

    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = y[tr_idx], y[te_idx]
    return X_tr, y_tr, X_te, y_te, list(X_tr.columns)


def load_and_preprocess_water_potability(
    test_size: float = 0.30, seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    """
    Loads and preprocesses the Water Potability dataset (Kadiwal, 2021).
    3,276 observations with 9 physicochemical water quality attributes.
    Stratified 70/30 train/test split. Missing values are imputed using
    median values fitted strictly on training data during the classifier pipeline.

    Returns:
        X_tr (pd.DataFrame): Training development partition (2,293 samples)
        y_tr (np.ndarray): Potability labels (1 = Potable, 0 = Not Potable)
        X_te (pd.DataFrame): Held-out test partition (983 samples)
        y_te (np.ndarray): Held-out test labels
        columns (List[str]): List of predictor names (9 features)
    """
    base_dir = get_datasets_base_dir()
    csv_path = os.path.join(base_dir, "water_patability", "water_potability.csv")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['Potability'])
    y = df['Potability'].to_numpy(dtype=int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(sss.split(X, y))

    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = y[tr_idx], y[te_idx]
    return X_tr, y_tr, X_te, y_te, list(X_tr.columns)


def load_and_preprocess_room_occupancy() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    """
    Loads and preprocesses the Room Occupancy (Multisensor) Estimation dataset (optional benchmark).
    Returns: X_tr, y_tr, X_te, y_te, columns
    """
    base_dir = get_datasets_base_dir()
    csv_path = os.path.join(base_dir, "room_occupancy", "Occupancy_Estimation.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Room Occupancy dataset file not found at '{csv_path}'.")
    df = pd.read_csv(csv_path)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Hour'] = df['Datetime'].dt.hour
    df['IsWeekend'] = df['Datetime'].dt.dayofweek.isin([5, 6]).astype(int)

    df = df.drop(columns=['Date', 'Time', 'Datetime'])
    X = df.drop(columns=['Room_Occupancy_Count'])
    y = df['Room_Occupancy_Count'].to_numpy(dtype=int)

    split_idx = int(len(df) * 0.7)
    X_tr, X_te = X.iloc[:split_idx].reset_index(drop=True), X.iloc[split_idx:].reset_index(drop=True)
    y_tr, y_te = y[:split_idx], y[split_idx:]
    return X_tr, y_tr, X_te, y_te, list(X_tr.columns)


def load_dataset(dataset_name: str) -> Tuple[str, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    """
    Convenience dispatcher to load any of the Smart City datasets evaluated in the paper:
    - 'building': Building Occupancy Detection (Candanedo, 2016)
    - 'grid': Electrical Grid Stability (Arzamasov, 2018)
    - 'water': Water Potability (Kadiwal, 2021)

    Returns:
        (canonical_name, X_tr, y_tr, X_te, y_te, columns)
    """
    norm = dataset_name.lower().replace("-", "_").replace(" ", "_").strip("_")

    if norm in ["building", "building_occupancy", "building_occupancy_detection"]:
        X_tr, y_tr, X_te, y_te, cols = load_and_preprocess_building_occupancy()
        return "Building Occupancy", X_tr, y_tr, X_te, y_te, cols
    elif norm in ["grid", "electric_grid", "electric_grid_stability", "electrical_grid"]:
        X_tr, y_tr, X_te, y_te, cols = load_and_preprocess_electric_grid_stability()
        return "Grid Stability", X_tr, y_tr, X_te, y_te, cols
    elif norm in ["water", "water_potability", "water_patability", "water_probability"]:
        X_tr, y_tr, X_te, y_te, cols = load_and_preprocess_water_potability()
        return "Water Potability", X_tr, y_tr, X_te, y_te, cols
    else:
        valid = ["building", "grid", "water"]
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets in paper are: {valid}")
