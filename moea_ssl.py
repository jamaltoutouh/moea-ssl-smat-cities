#!/usr/bin/env python3
"""
MOEA-SSL: Multi-Objective Evolutionary Algorithm using NSGA-II
Optimizes three objectives:
1. Minimize -RobustMacroF1 (Maximize RobustMacroF1 = mean(val_F1) - 0.4 * std(val_F1))
2. Minimize ProbeDropPlus (max(0, probe_F1_before - probe_F1_after))
3. Minimize FeatureRatio (|m1 U m2| / d)
"""

import os
import argparse
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelSpreading

warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------
# Basic metrics and helper functions
# -----------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))

def schedule_tau(t: int, T: int, tau_start: float, tau_end: float) -> float:
    if T <= 1:
        return float(tau_end)
    frac = t / (T - 1)
    return float(tau_start + (tau_end - tau_start) * frac)

def ensure_min_features(mask: np.ndarray, min_features: int, rng: np.random.Generator) -> np.ndarray:
    if int(mask.sum()) >= int(min_features):
        return mask
    d = mask.size
    idx = rng.choice(d, size=min_features, replace=False)
    out = np.zeros(d, dtype=bool)
    out[idx] = True
    return out

def safe_unique_count(y: np.ndarray) -> int:
    try:
        return int(np.unique(y).size)
    except Exception:
        return 0

# -----------------------------
# Preprocessing and Model Pipeline
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

def make_lr_pipeline(
    X_ref: pd.DataFrame,
    cols: List[str],
    calibrate: bool,
    y_for_calib: Optional[np.ndarray]
):
    pre = build_preprocessor_for_columns(X_ref, cols)
    lr_kwargs = dict(solver="lbfgs", max_iter=400)

    base = Pipeline([
        ("prep", pre),
        ("clf", LogisticRegression(**lr_kwargs))
    ])

    if not calibrate or y_for_calib is None:
        return base

    classes, counts = np.unique(y_for_calib, return_counts=True)
    if classes.size < 2:
        return base
    min_count = int(counts.min())
    n = int(y_for_calib.size)
    if n < 40 or min_count < 10:
        return base
    cv = min(3, min_count)
    if cv < 2:
        return base

    return CalibratedClassifierCV(base, method="sigmoid", cv=cv)

# -----------------------------
# Genomes Definition
# -----------------------------
@dataclass
class ViewGenome:
    mask1: np.ndarray
    mask2: np.ndarray

    @staticmethod
    def random(d: int, rng: np.random.Generator, frac: float = 0.5, min_features: int = 3) -> "ViewGenome":
        k = max(min_features, int(round(d * frac)))
        idx1 = rng.choice(d, size=k, replace=False)
        idx2 = rng.choice(d, size=k, replace=False)
        m1 = np.zeros(d, dtype=bool); m1[idx1] = True
        m2 = np.zeros(d, dtype=bool); m2[idx2] = True
        m1 = ensure_min_features(m1, min_features, rng)
        m2 = ensure_min_features(m2, min_features, rng)
        return ViewGenome(m1, m2)

    def clone(self) -> "ViewGenome":
        return ViewGenome(self.mask1.copy(), self.mask2.copy())

@dataclass
class PolicyGenome:
    calibrate: bool
    tau_start: float
    tau_end: float
    max_iters: int
    max_add_total: int
    max_add_per_class: int
    disagreement_veto: bool
    class_balance: bool
    veto_min_other_proba: float

    @staticmethod
    def random(rng: np.random.Generator) -> "PolicyGenome":
        tau_start = float(rng.uniform(0.85, 0.99))
        tau_end = float(rng.uniform(0.60, min(0.95, tau_start)))
        max_iters = int(rng.integers(3, 10))
        max_add_total = int(rng.integers(20, 400))
        max_add_per_class = int(rng.integers(10, 150))
        calibrate = bool(rng.integers(0, 2))
        disagreement_veto = bool(rng.integers(0, 2))
        class_balance = bool(rng.integers(0, 2))
        veto_min_other_proba = float(rng.uniform(0.45, 0.70))
        return PolicyGenome(
            calibrate=calibrate,
            tau_start=tau_start,
            tau_end=tau_end,
            max_iters=max_iters,
            max_add_total=max_add_total,
            max_add_per_class=max_add_per_class,
            disagreement_veto=disagreement_veto,
            class_balance=class_balance,
            veto_min_other_proba=veto_min_other_proba
        )

    def clone(self) -> "PolicyGenome":
        return PolicyGenome(**self.__dict__)

@dataclass
class JointGenome:
    view: ViewGenome
    policy: PolicyGenome

    @staticmethod
    def random(d: int, rng: np.random.Generator) -> "JointGenome":
        return JointGenome(
            view=ViewGenome.random(d, rng, frac=0.5, min_features=3),
            policy=PolicyGenome.random(rng),
        )

    def clone(self) -> "JointGenome":
        return JointGenome(view=self.view.clone(), policy=self.policy.clone())

# -----------------------------
# Crossover and Mutation operators
# -----------------------------
def crossover_view(a: ViewGenome, b: ViewGenome, rng: np.random.Generator) -> ViewGenome:
    child = a.clone()
    for name in ["mask1", "mask2"]:
        ma = getattr(a, name)
        mb = getattr(b, name)
        take_b = rng.random(ma.size) < 0.5
        mc = ma.copy()
        mc[take_b] = mb[take_b]
        setattr(child, name, mc)
    return child

def mutate_view(v: ViewGenome, rng: np.random.Generator, p_flip: float = 0.08, min_features: int = 3) -> ViewGenome:
    out = v.clone()
    for name in ["mask1", "mask2"]:
        m = getattr(out, name)
        flips = rng.random(m.size) < p_flip
        m = np.logical_xor(m, flips)
        m = ensure_min_features(m, min_features, rng)
        setattr(out, name, m)
    return out

def crossover_policy(a: PolicyGenome, b: PolicyGenome, rng: np.random.Generator) -> PolicyGenome:
    c = a.clone()
    alpha = float(rng.uniform(0.2, 0.8))
    c.tau_start = float(np.clip(alpha * a.tau_start + (1 - alpha) * b.tau_start, 0.75, 0.995))
    c.tau_end = float(np.clip(alpha * a.tau_end + (1 - alpha) * b.tau_end, 0.50, min(0.98, c.tau_start)))
    c.max_iters = int(np.clip(int(round(alpha * a.max_iters + (1 - alpha) * b.max_iters)), 2, 14))
    c.max_add_total = int(np.clip(int(round(alpha * a.max_add_total + (1 - alpha) * b.max_add_total)), 10, 2500))
    c.max_add_per_class = int(np.clip(int(round(alpha * a.max_add_per_class + (1 - alpha) * b.max_add_per_class)), 5, c.max_add_total))
    c.veto_min_other_proba = float(np.clip(alpha * a.veto_min_other_proba + (1 - alpha) * b.veto_min_other_proba, 0.40, 0.80))
    for name in ["calibrate", "disagreement_veto", "class_balance"]:
        setattr(c, name, bool(a.__dict__[name] if rng.random() < 0.5 else b.__dict__[name]))
    return c

def mutate_policy(p: PolicyGenome, rng: np.random.Generator) -> PolicyGenome:
    out = p.clone()
    out.tau_start = float(np.clip(out.tau_start + rng.normal(0, 0.03), 0.75, 0.995))
    out.tau_end = float(np.clip(out.tau_end + rng.normal(0, 0.06), 0.50, min(0.98, out.tau_start)))
    out.max_iters = int(np.clip(out.max_iters + int(rng.integers(-1, 2)), 2, 14))
    out.max_add_total = int(np.clip(out.max_add_total + int(rng.integers(-40, 41)), 10, 2500))
    out.max_add_per_class = int(np.clip(out.max_add_per_class + int(rng.integers(-20, 21)), 5, out.max_add_total))
    out.veto_min_other_proba = float(np.clip(out.veto_min_other_proba + rng.normal(0, 0.05), 0.40, 0.80))
    if rng.random() < 0.12:
        out.calibrate = not out.calibrate
    if rng.random() < 0.12:
        out.disagreement_veto = not out.disagreement_veto
    if rng.random() < 0.12:
        out.class_balance = not out.class_balance
    return out

def crossover_joint(a: JointGenome, b: JointGenome, rng: np.random.Generator) -> JointGenome:
    return JointGenome(
        view=crossover_view(a.view, b.view, rng),
        policy=crossover_policy(a.policy, b.policy, rng),
    )

def mutate_joint(g: JointGenome, rng: np.random.Generator) -> JointGenome:
    return JointGenome(
        view=mutate_view(g.view, rng, p_flip=0.08, min_features=3),
        policy=mutate_policy(g.policy, rng),
    )

# -----------------------------
# Pseudo-label selection and Co-training
# -----------------------------
def select_pseudolabels(
    proba: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
    max_add_total: int,
    max_add_per_class: int,
    class_balance: bool,
) -> np.ndarray:
    conf = np.max(proba, axis=1)
    idx = np.where(conf >= tau)[0]
    if idx.size == 0:
        return idx
    idx = idx[np.argsort(conf[idx])[::-1]]

    if not class_balance:
        return idx[:max_add_total]

    selected = []
    counts: Dict[int, int] = {}
    for i in idx:
        c = int(y_pred[i])
        if counts.get(c, 0) >= max_add_per_class:
            continue
        selected.append(i)
        counts[c] = counts.get(c, 0) + 1
        if len(selected) >= max_add_total:
            break
    return np.array(selected, dtype=int)

def cotraining_ccssl(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_eval: pd.DataFrame, y_eval: np.ndarray,
    X_probe: Optional[pd.DataFrame], y_probe: Optional[np.ndarray],
    columns: List[str],
    view: ViewGenome,
    policy: PolicyGenome,
    rng: np.random.Generator,
    max_u_fit: int = 8000,
) -> Tuple[float, float, float, int]:
    d = len(columns)
    cols1 = [columns[i] for i in range(d) if view.mask1[i]]
    cols2 = [columns[i] for i in range(d) if view.mask2[i]]

    if len(cols1) == 0 or len(cols2) == 0:
        return 0.0, 0.0, 1.0, 0

    if X_U.shape[0] > max_u_fit:
        take = rng.choice(X_U.shape[0], size=max_u_fit, replace=False)
        X_U_fit = X_U.iloc[take].reset_index(drop=True)
    else:
        X_U_fit = X_U.reset_index(drop=True)

    model1 = make_lr_pipeline(X_L, cols1, calibrate=policy.calibrate, y_for_calib=y_L)
    model2 = make_lr_pipeline(X_L, cols2, calibrate=policy.calibrate, y_for_calib=y_L)
    try:
        model1.fit(X_L[cols1], y_L)
        model2.fit(X_L[cols2], y_L)
    except Exception:
        return 0.0, 0.0, 1.0, 0

    # Probe baseline for drop computation
    f1_before = None
    if X_probe is not None and y_probe is not None and X_probe.shape[0] > 0:
        try:
            p1 = model1.predict_proba(X_probe[cols1])
            p2 = model2.predict_proba(X_probe[cols2])
            yb = np.argmax((p1 + p2) / 2.0, axis=1)
            f1_before = macro_f1(y_probe, yb)
        except Exception:
            f1_before = None

    U_idx = np.arange(X_U_fit.shape[0])
    total_added = 0

    for it in range(policy.max_iters):
        if U_idx.size == 0:
            break
        tau = schedule_tau(it, policy.max_iters, policy.tau_start, policy.tau_end)

        U1 = X_U_fit.iloc[U_idx][cols1]
        U2 = X_U_fit.iloc[U_idx][cols2]
        proba1 = model1.predict_proba(U1)
        proba2 = model2.predict_proba(U2)
        yhat1 = np.argmax(proba1, axis=1)
        yhat2 = np.argmax(proba2, axis=1)

        sel1_local = select_pseudolabels(
            proba1, yhat1, tau,
            policy.max_add_total, policy.max_add_per_class, policy.class_balance
        )
        sel2_local = select_pseudolabels(
            proba2, yhat2, tau,
            policy.max_add_total, policy.max_add_per_class, policy.class_balance
        )
        if sel1_local.size == 0 and sel2_local.size == 0:
            break

        if policy.disagreement_veto:
            def veto(sel_local: np.ndarray, yhat_src: np.ndarray, proba_other: np.ndarray) -> np.ndarray:
                if sel_local.size == 0:
                    return sel_local
                keep = []
                for j in sel_local:
                    lab = int(yhat_src[j])
                    if float(proba_other[j, lab]) >= policy.veto_min_other_proba:
                        keep.append(j)
                return np.array(keep, dtype=int)

            sel1_local = veto(sel1_local, yhat1, proba2)
            sel2_local = veto(sel2_local, yhat2, proba1)

        sel1 = U_idx[sel1_local]
        sel2 = U_idx[sel2_local]
        if sel1.size == 0 and sel2.size == 0:
            break

        y_add_for_2 = model1.predict(X_U_fit.iloc[sel1][cols1]) if sel1.size > 0 else np.array([], dtype=int)
        y_add_for_1 = model2.predict(X_U_fit.iloc[sel2][cols2]) if sel2.size > 0 else np.array([], dtype=int)

        X_L1 = pd.concat([X_L[cols1], X_U_fit.iloc[sel2][cols1]], axis=0, ignore_index=True)
        y_L1 = np.concatenate([y_L, y_add_for_1]) if sel2.size > 0 else y_L.copy()

        X_L2 = pd.concat([X_L[cols2], X_U_fit.iloc[sel1][cols2]], axis=0, ignore_index=True)
        y_L2 = np.concatenate([y_L, y_add_for_2]) if sel1.size > 0 else y_L.copy()

        if safe_unique_count(y_L1) < 2 or safe_unique_count(y_L2) < 2:
            break

        model1 = make_lr_pipeline(X_L, cols1, calibrate=policy.calibrate, y_for_calib=y_L1)
        model2 = make_lr_pipeline(X_L, cols2, calibrate=policy.calibrate, y_for_calib=y_L2)
        try:
            model1.fit(X_L1, y_L1)
            model2.fit(X_L2, y_L2)
        except Exception:
            break

        used = np.unique(np.concatenate([sel1, sel2]))
        total_added += int(used.size)
        keep = ~np.isin(U_idx, used)
        U_idx = U_idx[keep]
        if used.size < 5:
            break

    try:
        p1 = model1.predict_proba(X_eval[cols1])
        p2 = model2.predict_proba(X_eval[cols2])
        ye = np.argmax((p1 + p2) / 2.0, axis=1)
        f1_eval = macro_f1(y_eval, ye)
        acc_eval = acc(y_eval, ye)
    except Exception:
        f1_eval, acc_eval = 0.0, 0.0

    probe_drop = 0.0
    if X_probe is not None and y_probe is not None and f1_before is not None and X_probe.shape[0] > 0:
        try:
            p1 = model1.predict_proba(X_probe[cols1])
            p2 = model2.predict_proba(X_probe[cols2])
            yp = np.argmax((p1 + p2) / 2.0, axis=1)
            f1_after = macro_f1(y_probe, yp)
            probe_drop = max(0.0, float(f1_before - f1_after))
        except Exception:
            probe_drop = 0.0

    return float(f1_eval), float(acc_eval), float(probe_drop), int(total_added)

# -----------------------------
# Multiobjective Evaluation Function
# -----------------------------
def evaluate_multiobjective(
    ind: JointGenome,
    splits: Dict[str, Any],
    columns: List[str],
    eval_seeds: List[int],
    max_u_fit: int,
    alpha: float = 0.4
) -> Tuple[float, float, float]:
    """
    Evaluates three objectives:
    f1 = -RobustMacroF1 = -(mean(val_F1) - alpha * std(val_F1))
    f2 = mean(probe_drop) (where probe_drop is max(0, f1_before - f1_after))
    f3 = FeatureRatio = |m1 U m2| / d
    """
    f1_vals = []
    probe_drops = []
    
    for s in eval_seeds:
        rng = np.random.default_rng(s)
        f1, _acc, probe_drop, _added = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_V"], splits["y_V"],
            splits["X_P"], splits["y_P"],
            columns, ind.view, ind.policy, rng,
            max_u_fit=max_u_fit
        )
        f1_vals.append(f1)
        probe_drops.append(probe_drop)
        
    f1_vals = np.array(f1_vals, dtype=float)
    probe_drops = np.array(probe_drops, dtype=float)
    
    mu = float(f1_vals.mean())
    sigma = float(f1_vals.std())
    robust_f1 = mu - alpha * sigma
    
    f1 = -robust_f1
    f2 = float(probe_drops.mean())
    
    # Feature Ratio
    union_mask = np.logical_or(ind.view.mask1, ind.view.mask2)
    f3 = float(union_mask.sum()) / float(len(columns))
    
    return f1, f2, f3

# -----------------------------
# NSGA-II Algorithms
# -----------------------------
def fast_non_dominated_sort(objectives: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """
    Standard NSGA-II fast non-dominated sorting.
    objectives: shape (N, M) where we want to minimize all M objectives.
    Returns:
        fronts: list of lists containing indices of individuals in each front
        ranks: 1D array of rank for each individual (0-indexed rank, where 0 is Pareto front)
    """
    N = objectives.shape[0]
    domination_sets = [[] for _ in range(N)]
    domination_counters = np.zeros(N, dtype=int)
    ranks = np.zeros(N, dtype=int)
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            # Check if p dominates q
            # p dominates q if p is no worse than q in all, and strictly better in at least one
            less_equal = (objectives[p] <= objectives[q]).all()
            less_than = (objectives[p] < objectives[q]).any()
            
            if less_equal and less_than:
                domination_sets[p].append(q)
            elif (objectives[q] <= objectives[p]).all() and (objectives[q] < objectives[p]).any():
                domination_counters[p] += 1

        if domination_counters[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in domination_sets[p]:
                domination_counters[q] -= 1
                if domination_counters[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts, ranks

def calculate_crowding_distance(objectives: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
    """
    Standard NSGA-II crowding distance calculation.
    """
    N = objectives.shape[0]
    distances = np.zeros(N, dtype=float)

    for front in fronts:
        if len(front) == 0:
            continue
        if len(front) <= 2:
            distances[front] = np.inf
            continue

        for m in range(objectives.shape[1]):
            front_objs = objectives[front, m]
            sorted_idx = np.argsort(front_objs)
            
            distances[front[sorted_idx[0]]] = np.inf
            distances[front[sorted_idx[-1]]] = np.inf
            
            obj_min = front_objs[sorted_idx[0]]
            obj_max = front_objs[sorted_idx[-1]]
            scale = obj_max - obj_min
            if scale == 0:
                scale = 1.0
                
            for i in range(1, len(front) - 1):
                idx_prev = front[sorted_idx[i-1]]
                idx_next = front[sorted_idx[i+1]]
                idx_curr = front[sorted_idx[i]]
                if distances[idx_curr] != np.inf:
                    distances[idx_curr] += (objectives[idx_next, m] - objectives[idx_prev, m]) / scale
                    
    return distances

def binary_tournament_selection(
    pop: List[JointGenome],
    ranks: np.ndarray,
    crowding_distances: np.ndarray,
    rng: np.random.Generator
) -> JointGenome:
    N = len(pop)
    i, j = rng.choice(N, size=2, replace=False)
    # Crowded comparison selection
    if ranks[i] < ranks[j]:
        return pop[i]
    elif ranks[j] < ranks[i]:
        return pop[j]
    else:
        if crowding_distances[i] > crowding_distances[j]:
            return pop[i]
        else:
            return pop[j]

# -----------------------------
# Main MOEA Loop (NSGA-II)
# -----------------------------
def evolve_nsga2_ssl(
    splits: Dict[str, Any],
    columns: List[str],
    pop_size: int,
    generations: int,
    seed: int,
    eval_seeds: List[int],
    max_u_fit: int,
    alpha: float = 0.4
) -> Tuple[List[JointGenome], np.ndarray]:
    rng = np.random.default_rng(seed)
    d = len(columns)
    
    # Initialize P_0
    pop = [JointGenome.random(d, rng) for _ in range(pop_size)]
    
    # Evaluate P_0
    objs = np.zeros((pop_size, 3), dtype=float)
    for i, ind in enumerate(pop):
        objs[i] = evaluate_multiobjective(ind, splits, columns, eval_seeds, max_u_fit, alpha)
        
    fronts, ranks = fast_non_dominated_sort(objs)
    crowding_distances = calculate_crowding_distance(objs, fronts)
    
    for gen in range(int(generations)):
        # Generate offspring Q_t of size N
        offspring = []
        while len(offspring) < pop_size:
            p1 = binary_tournament_selection(pop, ranks, crowding_distances, rng)
            p2 = binary_tournament_selection(pop, ranks, crowding_distances, rng)
            child = mutate_joint(crossover_joint(p1, p2, rng), rng)
            offspring.append(child)
            
        # Evaluate Q_t
        offspring_objs = np.zeros((pop_size, 3), dtype=float)
        for i, ind in enumerate(offspring):
            offspring_objs[i] = evaluate_multiobjective(ind, splits, columns, eval_seeds, max_u_fit, alpha)
            
        # Combine R_t = P_t U Q_t (size 2N)
        combined_pop = pop + offspring
        combined_objs = np.vstack([objs, offspring_objs])
        
        # Non-dominated sort on R_t
        c_fronts, c_ranks = fast_non_dominated_sort(combined_objs)
        c_distances = calculate_crowding_distance(combined_objs, c_fronts)
        
        # Build P_t+1
        new_pop_indices = []
        front_idx = 0
        
        while len(new_pop_indices) + len(c_fronts[front_idx]) <= pop_size:
            new_pop_indices.extend(c_fronts[front_idx])
            front_idx += 1
            if front_idx >= len(c_fronts):
                break
                
        # If we need more individuals to fill the population of size N
        if len(new_pop_indices) < pop_size and front_idx < len(c_fronts):
            remaining_slots = pop_size - len(new_pop_indices)
            last_front = c_fronts[front_idx]
            
            # Sort the last front by crowding distance descending
            last_front_cd = c_distances[last_front]
            sorted_last_front = [last_front[i] for i in np.argsort(-last_front_cd)]
            
            new_pop_indices.extend(sorted_last_front[:remaining_slots])
            
        # Update pop and objs
        pop = [combined_pop[i] for i in new_pop_indices]
        objs = combined_objs[new_pop_indices]
        
        # Re-sort and re-distance the new population
        fronts, ranks = fast_non_dominated_sort(objs)
        crowding_distances = calculate_crowding_distance(objs, fronts)
        
        print(f"Gen {gen+1}/{generations} complete. Pareto Front Size: {len(fronts[0])}")
        
    # Return Pareto-optimal individuals (front 0)
    pareto_indices = fronts[0]
    pareto_pop = [pop[i] for i in pareto_indices]
    pareto_objs = objs[pareto_indices]
    return pareto_pop, pareto_objs

# -----------------------------
# Dataset loaders
# -----------------------------
from src.dataset_loaders import (
    load_and_preprocess_building_occupancy,
    load_and_preprocess_room_occupancy,
    load_and_preprocess_water_potability,
    load_and_preprocess_electric_grid_stability,
    load_dataset
)

# -----------------------------
# Local data splitting function
# -----------------------------
def make_local_splits(
    X_train_full: pd.DataFrame, y_train_full: np.ndarray,
    X_test_full: pd.DataFrame, y_test_full: np.ndarray,
    labeled_frac: float,
    seed: int,
    val_frac_of_train: float = 0.20,
    probe_size: int = 120,
) -> Dict[str, Any]:
    from sklearn.model_selection import StratifiedShuffleSplit
    rng = np.random.default_rng(seed)
    y_train_full = np.asarray(y_train_full)

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
# Baselines
# -----------------------------
def self_training_baseline(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_T: pd.DataFrame, y_T: np.ndarray,
    columns: List[str],
    calibrate: bool,
    tau_start: float = 0.97,
    tau_end: float = 0.80,
    max_iters: int = 8,
    max_add_total: int = 250,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    model = make_lr_pipeline(X_L, columns, calibrate=calibrate, y_for_calib=y_L)
    model.fit(X_L[columns], y_L)

    U_idx = np.arange(X_U.shape[0])
    for it in range(max_iters):
        if U_idx.size == 0:
            break
        tau = schedule_tau(it, max_iters, tau_start, tau_end)
        proba = model.predict_proba(X_U.iloc[U_idx][columns])
        yhat = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
        sel_local = np.where(conf >= tau)[0]
        if sel_local.size == 0:
            break
        sel_local = sel_local[np.argsort(conf[sel_local])[::-1]][:max_add_total]
        sel = U_idx[sel_local]

        X_L = pd.concat([X_L, X_U.iloc[sel]], axis=0, ignore_index=True)
        y_L = np.concatenate([y_L, yhat[sel_local]])
        if safe_unique_count(y_L) < 2:
            break
        model = make_lr_pipeline(X_L, columns, calibrate=calibrate, y_for_calib=y_L)
        model.fit(X_L[columns], y_L)

        keep = ~np.isin(U_idx, sel)
        U_idx = U_idx[keep]

    y_pred = np.argmax(model.predict_proba(X_T[columns]), axis=1)
    return macro_f1(y_T, y_pred), acc(y_T, y_pred)

def heuristic_cotraining_baseline(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_T: pd.DataFrame, y_T: np.ndarray,
    columns: List[str],
    rng: np.random.Generator
) -> Tuple[float, float]:
    d = len(columns)
    perm = rng.permutation(d)
    half = max(1, d // 2)
    m1 = np.zeros(d, dtype=bool); m1[perm[:half]] = True
    m2 = np.zeros(d, dtype=bool); m2[perm[half:]] = True
    if int(m2.sum()) == 0:
        m2[perm[:half]] = True

    view = ViewGenome(m1, m2)
    policy = PolicyGenome(
        calibrate=True, tau_start=0.97, tau_end=0.80, max_iters=7,
        max_add_total=250, max_add_per_class=120,
        disagreement_veto=True, class_balance=True,
        veto_min_other_proba=0.5
    )
    f1, _acc, _drop, _added = cotraining_ccssl(
        X_L, y_L, X_U, X_T, y_T, None, None, columns, view, policy, rng
    )
    return f1, _acc

def label_spreading_baseline(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_T: pd.DataFrame, y_T: np.ndarray,
    columns: List[str],
    max_total: int,
    rng: np.random.Generator
) -> Optional[Tuple[float, float]]:
    X_all = pd.concat([X_L, X_U], axis=0, ignore_index=True)
    if X_all.shape[0] > max_total:
        return None

    pre = build_preprocessor_for_columns(X_all, columns)
    X_mat = pre.fit_transform(X_all[columns])

    d = X_mat.shape[1]
    gamma = 1.0 / (2.0 * d)
    ls = LabelSpreading(kernel="rbf", gamma=gamma, max_iter=30)

    y_all = np.concatenate([y_L, -np.ones(X_U.shape[0], dtype=int)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls.fit(X_mat, y_all)
    X_T_mat = pre.transform(X_T[columns])
    y_pred = ls.predict(X_T_mat)
    return macro_f1(y_T, y_pred), acc(y_T, y_pred)

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
    
    splits = make_local_splits(X_tr, y_tr, X_te, y_te, labeled_frac=labeled_frac, seed=seed, probe_size=cfg["probe_size"])
    nL = int(splits["X_L"].shape[0])
    nU = int(splits["X_U"].shape[0])
    nP = int(0 if splits["X_P"] is None else splits["X_P"].shape[0])
    nV = int(splits["X_V"].shape[0])
    nT = int(splits["X_T"].shape[0])

    eval_seeds = [seed + 101, seed + 103, seed + 107]

    # Run NSGA-II Multi-Objective Optimization
    t_start_moea = time.time()
    pareto_pop, pareto_objs = evolve_nsga2_ssl(
        splits=splits,
        columns=columns,
        pop_size=cfg["popEA"],
        generations=cfg["generations"],
        seed=seed,
        eval_seeds=eval_seeds,
        max_u_fit=cfg["max_u_fit"],
        alpha=cfg["alpha"]
    )
    moea_time = time.time() - t_start_moea

    rows = []
    def add_row(method: str, f1: float, acc_val: float, extra: Dict[str, Any], run_time_sec: float = 0.0):
        base = dict(
            dataset=dataset_name,
            labeled_frac=float(labeled_frac),
            seed=int(seed),
            method=method,
            macroF1_test=float(f1),
            acc_test=float(acc_val),
            run_time_sec=float(round(run_time_sec, 4)),
        )
        base.update(extra)
        rows.append(base)

    # Evaluate all Pareto-optimal trade-offs on test set
    for i, (ind, obj) in enumerate(zip(pareto_pop, pareto_objs)):
        f1_test, acc_test, probe_drop_test, added_test = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_T"], splits["y_T"],
            splits["X_P"], splits["y_P"],
            columns, ind.view, ind.policy,
            rng=np.random.default_rng(seed + 999),
            max_u_fit=cfg["max_u_final"]
        )
        
        # objectives: obj[0] = -RobustF1, obj[1] = ProbeDropPlus, obj[2] = FeatureRatio
        add_row(f"MOEA-SSL Pareto {i+1}", f1_test, acc_test, dict(
            val_robust_f1=float(-obj[0]),
            val_probe_drop=float(obj[1]),
            val_feature_ratio=float(obj[2]),
            probe_drop_test=float(probe_drop_test),
            pseudo_added_test=int(added_test),
            policy_calibrate=bool(ind.policy.calibrate),
            policy_tau_start=float(ind.policy.tau_start),
            policy_tau_end=float(ind.policy.tau_end),
            policy_max_iters=int(ind.policy.max_iters),
            policy_disagree=bool(ind.policy.disagreement_veto),
            policy_balance=bool(ind.policy.class_balance),
            policy_veto_min_other=float(ind.policy.veto_min_other_proba),
            view_overlap=float(np.mean(np.logical_and(ind.view.mask1, ind.view.mask2))),
            view_size1=int(ind.view.mask1.sum()),
            view_size2=int(ind.view.mask2.sum()),
            n_features=int(len(columns)),
            n_classes=int(np.unique(y_te).size),
            n_labeled=nL,
            n_unlabeled=nU,
            n_probe=nP,
            n_val=nV,
            n_test=nT,
        ), run_time_sec=moea_time)

    # Evaluate baselines
    t_start_st = time.time()
    st_f1, st_acc = self_training_baseline(
        splits["X_L"].copy(), splits["y_L"].copy(),
        splits["X_U"].copy(),
        splits["X_T"], splits["y_T"],
        columns, calibrate=True,
        rng=np.random.default_rng(seed + 555)
    )
    st_time = time.time() - t_start_st

    t_start_hct = time.time()
    hct_f1, hct_acc = heuristic_cotraining_baseline(
        splits["X_L"], splits["y_L"],
        splits["X_U"],
        splits["X_T"], splits["y_T"],
        columns,
        rng=np.random.default_rng(seed + 777)
    )
    hct_time = time.time() - t_start_hct

    t_start_ls = time.time()
    ls_out = label_spreading_baseline(
        splits["X_L"], splits["y_L"],
        splits["X_U"],
        splits["X_T"], splits["y_T"],
        columns,
        max_total=cfg["label_spreading_max_total"],
        rng=np.random.default_rng(seed + 888)
    )
    ls_time = time.time() - t_start_ls

    add_row("Self-training", st_f1, st_acc, {}, run_time_sec=st_time)
    add_row("Heuristic co-training", hct_f1, hct_acc, {}, run_time_sec=hct_time)
    if ls_out is not None:
        add_row("Label Spreading", ls_out[0], ls_out[1], {}, run_time_sec=ls_time)
    else:
        add_row("Label Spreading (skipped)", np.nan, np.nan, {"reason": "too_large"}, run_time_sec=ls_time)

    return rows

# -----------------------------
# Main Executable
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["building", "room", "water", "grid"], help="Dataset name: building, room, water, or grid")
    ap.add_argument("--labeled_fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--popEA", type=int, default=12, help="NSGA-II population size N")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--teams_per_individual", type=int, default=2)
    ap.add_argument("--max_u_fit", type=int, default=8000)
    ap.add_argument("--max_u_final", type=int, default=20000)
    ap.add_argument("--probe_size", type=int, default=120)
    ap.add_argument("--label_spreading_max_total", type=int, default=4000)
    ap.add_argument("--out_csv", type=str, default="results_moeassl_local.csv")
    ap.add_argument("--alpha", type=float, default=0.4, help="Robustness penalty multiplier for Objective 1")
    
    args = ap.parse_args()

    # Load selected dataset
    dataset_name, X_tr, y_tr, X_te, y_te, columns = load_dataset(args.dataset)

    cfg = vars(args)
    results = []

    print(f"Loaded dataset: {dataset_name} (Features: {len(columns)}, Train Pool: {X_tr.shape[0]}, Test Pool: {X_te.shape[0]})")
    print(f"Running MOEA-SSL (NSGA-II) over labeled fractions {args.labeled_fracs} and seeds {args.seeds}...")

    for lf in args.labeled_fracs:
        for seed in args.seeds:
            print(f"\n--- Running: Frac={lf}, Seed={seed} ---")
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
            print(f"Completed in {time.time() - t0:.1f}s.")
            results.extend(rows)

            print(f"Pareto solutions found: {len(rows) - 3}")
            for row in rows:
                if "Pareto" in row["method"]:
                    print(f"  {row['method']:<20} -> Val F1: {row['val_robust_f1']:.4f}, Val Drop: {row['val_probe_drop']:.4f}, Val FR: {row['val_feature_ratio']:.4f} | Test F1: {row['macroF1_test']:.4f}")
                else:
                    print(f"  {row['method']:<20} -> Test F1: {row['macroF1_test']:.4f}")

    # Save to CSV
    df_out = pd.DataFrame(results)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"\n==========================================")
    print(f"All MOEA benchmarks completed! Results saved to: {args.out_csv}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
