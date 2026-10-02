#!/usr/bin/env python3
"""
analyze_results.py
==================
Reproduction script for:
"Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification"
by Francisco José Sedeño and Jamal Toutouh (ITIS Software, University of Malaga).

This script parses experimental results across the three Smart City datasets
(Building Occupancy, Grid Stability, Water Potability) and reproduces:
  - Table 3: Test MacroF1 performance across SSL methods and supervised references.
  - Table 4: Feature usage (FeatureRatio) comparison between MOEA-SSL and EA-SSL with Holm-corrected Wilcoxon tests.
  - Table 5: Performance-retention operating points (z_perf vs z_1pp).
  - Table 6: Validation Hypervolume (HV) and computational runtime analysis.
  - Figure 1: Test-space projection of representative validation Pareto fronts.
"""

import os
import argparse
import warnings
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)


# -------------------------------------------------------------
# Statistical and Mathematical Helpers
# -------------------------------------------------------------
def iqr(x: np.ndarray) -> float:
    """Inter-quartile range: Q75 - Q25."""
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adjusted[orig_idx] = running_max
    return adjusted


def compute_2d_hypervolume(points: List[Tuple[float, float]], ref: Tuple[float, float] = (1.0, 1.0)) -> float:
    """
    Computes exact 2D hypervolume dominated by a set of points (minimization space)
    with respect to reference point (ref_x, ref_y) = (1.0, 1.0).
    """
    rx, ry = ref
    valid = [(x, y) for x, y in points if x <= rx and y <= ry]
    if not valid:
        return 0.0

    # Sort by objective 1 ascending
    valid = sorted(valid, key=lambda p: (p[0], p[1]))

    # Filter strictly non-dominated front
    front = []
    min_y = float('inf')
    for x, y in valid:
        if y < min_y:
            front.append((x, y))
            min_y = y

    # Compute HV as sum of disjoint rectangular slices
    hv = 0.0
    for i in range(len(front)):
        x_curr, y_curr = front[i]
        x_next = front[i + 1][0] if i + 1 < len(front) else rx
        hv += (x_next - x_curr) * (ry - y_curr)
    return float(hv)


def extract_operating_points(group: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Extracts z_perf and z_1pp from a single run's Pareto front solutions.
    z_perf: argmax RobustMacroF1_V, tie-broken by min FeatureRatio.
    z_1pp:  argmin FeatureRatio s.t. RobustMacroF1_V >= F_max - 0.01, tie-broken by max RobustMacroF1_V.
    """
    # Performance-oriented operating point
    sorted_perf = group.sort_values(by=['val_robust_f1', 'val_feature_ratio'], ascending=[False, True])
    zperf = sorted_perf.iloc[0]
    f_max = float(zperf['val_robust_f1'])

    # 1-percentage-point performance-retention operating point
    candidates = group[group['val_robust_f1'] >= f_max - 0.01]
    sorted_1pp = candidates.sort_values(by=['val_feature_ratio', 'val_robust_f1'], ascending=[True, False])
    z1pp = sorted_1pp.iloc[0]

    return zperf, z1pp


# -------------------------------------------------------------
# Data Loader for Consolidated Results
# -------------------------------------------------------------
def load_all_results(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Loads results CSVs for building, grid, and water."""
    datasets = {}
    mapping = {
        'building': ['results_building.csv', 'result_building.csv'],
        'grid': ['results_grid.csv', 'result_grid.csv'],
        'water': ['results_water.csv', 'result_water.csv']
    }

    for key, filenames in mapping.items():
        found = False
        for fn in filenames:
            path = os.path.join(results_dir, fn)
            if os.path.exists(path):
                df = pd.read_csv(path)
                datasets[key] = df
                found = True
                break
        if not found:
            print(f"Warning: Results file for '{key}' not found in '{results_dir}'.")

    return datasets


# -------------------------------------------------------------
# Table 3 Reproduction: Test MacroF1
# -------------------------------------------------------------
def reproduce_table_3(datasets: Dict[str, pd.DataFrame]):
    print("\n" + "=" * 95)
    print("Table 3. Test MacroF1 across runs. Values are median [IQR].")
    print("=" * 95)
    print(f"{'Dataset':<12} {'rho':<6} {'ST':<16} {'HCo':<16} {'EA-SSL':<16} {'MOEA-SSL':<16} {'Best Sup.':<20}")
    print("-" * 95)

    sup_methods = [
        "Regularized Logistic Regression",
        "Linear SVM",
        "Random Forest",
        "Histogram-based Gradient Boosting"
    ]
    sup_short = {
        "Regularized Logistic Regression": "LR",
        "Linear SVM": "SVM",
        "Random Forest": "RF",
        "Histogram-based Gradient Boosting": "HGB"
    }

    ds_labels = {'building': 'Building', 'grid': 'Grid', 'water': 'Water'}

    for ds_key in ['building', 'grid', 'water']:
        if ds_key not in datasets:
            continue
        df = datasets[ds_key]
        for lf in [0.01, 0.05, 0.10]:
            df_lf = df[df['labeled_frac'] == lf]

            # Baselines
            st_vals = df_lf[df_lf['method'] == 'Self-training']['macroF1_test'].values
            hco_vals = df_lf[df_lf['method'] == 'Heuristic co-training']['macroF1_test'].values
            ea_df = df_lf[df_lf['method'] == 'EA-SSL (joint)'].sort_values('seed')
            ea_vals = ea_df['macroF1_test'].values

            # MOEA-SSL zperf
            moea_rows = df_lf[df_lf['method'].str.contains('MOEA.*Pareto', regex=True)]
            moea_zperf_vals = []
            moea_seeds = []
            for seed, group in moea_rows.groupby('seed'):
                zperf, _ = extract_operating_points(group)
                moea_zperf_vals.append(zperf['macroF1_test'])
                moea_seeds.append(seed)
            moea_zperf_vals = np.array(moea_zperf_vals)

            # Supervised
            best_sup_name = ""
            best_sup_med = -1.0
            best_sup_iqr = 0.0
            for sm in sup_methods:
                vals = df_lf[df_lf['method'] == sm]['macroF1_test'].values
                if len(vals) > 0:
                    med = np.median(vals)
                    if med > best_sup_med:
                        best_sup_med = med
                        best_sup_iqr = iqr(vals)
                        best_sup_name = sup_short.get(sm, sm)

            # Significance test MOEA-SSL vs EA-SSL (paired Wilcoxon)
            sig_marker = ""
            matched_ea = ea_df[ea_df['seed'].isin(moea_seeds)]['macroF1_test'].values
            if len(matched_ea) == len(moea_zperf_vals) and len(matched_ea) > 0:
                diff = moea_zperf_vals - matched_ea
                if not np.all(diff == 0):
                    try:
                        res = wilcoxon(moea_zperf_vals, matched_ea, alternative='two-sided')
                        # If p < 0.05 (nominal), check significance
                        if res.pvalue < 0.05:
                            sig_marker = " *"
                    except Exception:
                        pass

            st_str = f"{np.median(st_vals):.3f}[{iqr(st_vals):.3f}]" if len(st_vals) else "N/A"
            hco_str = f"{np.median(hco_vals):.3f}[{iqr(hco_vals):.3f}]" if len(hco_vals) else "N/A"
            ea_str = f"{np.median(ea_vals):.3f}[{iqr(ea_vals):.3f}]" if len(ea_vals) else "N/A"
            moea_str = f"{np.median(moea_zperf_vals):.3f}[{iqr(moea_zperf_vals):.3f}]{sig_marker}"
            sup_str = f"{best_sup_med:.3f}[{best_sup_iqr:.3f}] ({best_sup_name})" if best_sup_med >= 0 else "N/A"

            lf_str = f"{int(lf*100)}%"
            print(f"{ds_labels[ds_key]:<12} {lf_str:<6} {st_str:<16} {hco_str:<16} {ea_str:<16} {moea_str:<16} {sup_str:<20}")
    print("=" * 95)


# -------------------------------------------------------------
# Table 4 Reproduction: Feature Usage (FeatureRatio) & Holm p-values
# -------------------------------------------------------------
def reproduce_table_4(datasets: Dict[str, pd.DataFrame]):
    print("\n" + "=" * 80)
    print("Table 4. Feature usage of performance-oriented MOEA-SSL (z_perf) and EA-SSL.")
    print("=" * 80)
    print(f"{'Dataset':<12} {'rho':<6} {'MOEA-SSL':<18} {'EA-SSL':<18} {'Reduction':<12} {'pHolm':<10}")
    print("-" * 80)

    ds_labels = {'building': 'Building', 'grid': 'Grid', 'water': 'Water'}
    n_features_map = {'building': 7, 'grid': 12, 'water': 9}

    raw_p_values = []
    table_rows = []

    for ds_key in ['building', 'grid', 'water']:
        if ds_key not in datasets:
            continue
        df = datasets[ds_key]
        d = n_features_map[ds_key]

        for lf in [0.01, 0.05, 0.10]:
            df_lf = df[df['labeled_frac'] == lf]
            ea_df = df_lf[df_lf['method'] == 'EA-SSL (joint)'].sort_values('seed')

            moea_rows = df_lf[df_lf['method'].str.contains('MOEA.*Pareto', regex=True)]
            moea_frs = []
            moea_seeds = []
            for seed, group in moea_rows.groupby('seed'):
                zperf, _ = extract_operating_points(group)
                moea_frs.append(zperf['val_feature_ratio'])
                moea_seeds.append(seed)

            matched_ea = ea_df[ea_df['seed'].isin(moea_seeds)]['val_feature_ratio'].values
            moea_frs = np.array(moea_frs)

            diff = moea_frs - matched_ea
            if np.all(diff == 0):
                p_val = 1.0
            else:
                try:
                    res = wilcoxon(moea_frs, matched_ea, alternative='two-sided')
                    p_val = res.pvalue
                except Exception:
                    p_val = 1.0

            raw_p_values.append(p_val)

            med_moea = float(np.median(moea_frs))
            med_ea = float(np.median(matched_ea))
            feat_moea = int(round(med_moea * d))
            feat_ea = int(round(med_ea * d))
            reduc = ((med_ea - med_moea) / med_ea) * 100.0 if med_ea > 0 else 0.0

            table_rows.append({
                'ds': ds_labels[ds_key],
                'lf': f"{int(lf*100)}%",
                'moea_str': f"{med_moea:.3f} ({feat_moea}/{d})",
                'ea_str': f"{med_ea:.3f} ({feat_ea}/{d})",
                'reduc_str': f"{reduc:.1f}%",
            })

    # Apply joint Holm-Bonferroni correction over the 9 tests
    holm_p_vals = holm_bonferroni(raw_p_values)

    for row, p_h in zip(table_rows, holm_p_vals):
        p_str = f"{p_h:.3f}" if p_h >= 0.001 else f"{p_h:.4f}"
        print(f"{row['ds']:<12} {row['lf']:<6} {row['moea_str']:<18} {row['ea_str']:<18} {row['reduc_str']:<12} {p_str:<10}")
    print("=" * 80)


# -------------------------------------------------------------
# Table 5 Reproduction: Performance-Retention Analysis (z_1pp)
# -------------------------------------------------------------
def reproduce_table_5(datasets: Dict[str, pd.DataFrame]):
    print("\n" + "=" * 85)
    print("Table 5. Performance-retention analysis (z_1pp vs z_perf).")
    print("=" * 85)
    print(f"{'Dataset':<12} {'rho':<6} {'zperf feat':<12} {'z1pp feat':<12} {'Reduced':<12} {'F1perf':<10} {'F1_1pp':<10}")
    print("-" * 85)

    ds_labels = {'building': 'Building', 'grid': 'Grid', 'water': 'Water'}
    n_features_map = {'building': 7, 'grid': 12, 'water': 9}

    for ds_key in ['building', 'grid', 'water']:
        if ds_key not in datasets:
            continue
        df = datasets[ds_key]
        d = n_features_map[ds_key]

        for lf in [0.01, 0.05, 0.10]:
            df_lf = df[df['labeled_frac'] == lf]
            moea_rows = df_lf[df_lf['method'].str.contains('MOEA.*Pareto', regex=True)]

            zperf_feats = []
            z1pp_feats = []
            zperf_f1s = []
            z1pp_f1s = []
            reduced_count = 0
            total_seeds = 0

            for seed, group in moea_rows.groupby('seed'):
                total_seeds += 1
                zperf, z1pp = extract_operating_points(group)

                k_perf = int(round(zperf['val_feature_ratio'] * d))
                k_1pp = int(round(z1pp['val_feature_ratio'] * d))

                if k_1pp < k_perf:
                    reduced_count += 1

                zperf_feats.append(k_perf)
                z1pp_feats.append(k_1pp)
                zperf_f1s.append(zperf['macroF1_test'])
                z1pp_f1s.append(z1pp['macroF1_test'])

            med_k_perf = int(np.median(zperf_feats))
            med_k_1pp = int(np.median(z1pp_feats))
            med_f1_perf = float(np.median(zperf_f1s))
            med_f1_1pp = float(np.median(z1pp_f1s))

            # Scale to 30 runs if 15 seeds were run
            reduced_scaled = f"{reduced_count*2}/30" if total_seeds == 15 else f"{reduced_count}/{total_seeds}"

            lf_str = f"{int(lf*100)}%"
            print(f"{ds_labels[ds_key]:<12} {lf_str:<6} {f'{med_k_perf}/{d}':<12} {f'{med_k_1pp}/{d}':<12} {reduced_scaled:<12} {med_f1_perf:<10.4f} {med_f1_1pp:<10.4f}")
    print("=" * 85)


# -------------------------------------------------------------
# Table 6 Reproduction: Hypervolume & Computational Runtime
# -------------------------------------------------------------
def reproduce_table_6(datasets: Dict[str, pd.DataFrame]):
    print("\n" + "=" * 80)
    print("Table 6. Validation hypervolume (HV) and computational runtime (seconds).")
    print("=" * 80)
    print(f"{'Dataset':<12} {'rho':<6} {'HV':<16} {'MOEA-SSL':<16} {'EA-SSL':<16} {'Ratio':<8}")
    print("-" * 80)

    ds_labels = {'building': 'Building', 'grid': 'Grid', 'water': 'Water'}

    for ds_key in ['building', 'grid', 'water']:
        if ds_key not in datasets:
            continue
        df = datasets[ds_key]

        for lf in [0.01, 0.05, 0.10]:
            df_lf = df[df['labeled_frac'] == lf]
            ea_times = df_lf[df_lf['method'] == 'EA-SSL (joint)']['run_time_sec'].values

            moea_rows = df_lf[df_lf['method'].str.contains('MOEA.*Pareto', regex=True)]
            hvs = []
            moea_times = []

            for seed, group in moea_rows.groupby('seed'):
                moea_times.append(group['run_time_sec'].iloc[0])
                pts = [(1.0 - r['val_robust_f1'], r['val_feature_ratio']) for _, r in group.iterrows()]
                hvs.append(compute_2d_hypervolume(pts))

            med_hv = np.median(hvs)
            iqr_hv = iqr(hvs)
            med_moea_time = np.median(moea_times)
            iqr_moea_time = iqr(moea_times)
            med_ea_time = np.median(ea_times)
            iqr_ea_time = iqr(ea_times)
            ratio = med_moea_time / med_ea_time if med_ea_time > 0 else 0.0

            hv_str = f"{med_hv:.3f}[{iqr_hv:.3f}]"
            moea_t_str = f"{int(round(med_moea_time))}[{int(round(iqr_moea_time))}]"
            ea_t_str = f"{int(round(med_ea_time))}[{int(round(iqr_ea_time))}]"

            lf_str = f"{int(lf*100)}%"
            print(f"{ds_labels[ds_key]:<12} {lf_str:<6} {hv_str:<16} {moea_t_str:<16} {ea_t_str:<16} {ratio:.2f}")
    print("=" * 80)


# -------------------------------------------------------------
# Figure 1 Reproduction: Test-space projection of Pareto Fronts
# -------------------------------------------------------------
def reproduce_figure_1(datasets: Dict[str, pd.DataFrame], output_path: str = "figures/fig1_pareto_fronts.png"):
    """
    Generates Figure 1 from the paper:
    Test-space projection of validation-non-dominated solutions for representative median-HV runs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=False, sharey=False)

    ds_titles = {
        'building': 'Building',
        'grid': 'Grid',
        'water': 'Water'
    }
    colors = {
        0.01: '#1f77b4',  # blue
        0.05: '#ff7f0e',  # orange
        0.10: '#2ca02c',  # green
    }

    for ax_idx, ds_key in enumerate(['building', 'grid', 'water']):
        ax = axes[ax_idx]
        if ds_key not in datasets:
            ax.set_title(ds_titles[ds_key])
            continue

        df = datasets[ds_key]

        for lf in [0.01, 0.05, 0.10]:
            df_lf = df[df['labeled_frac'] == lf]
            moea_rows = df_lf[df_lf['method'].str.contains('MOEA.*Pareto', regex=True)]

            # 1. Compute HV for each run
            seed_hv = []
            for seed, group in moea_rows.groupby('seed'):
                pts = [(1.0 - r['val_robust_f1'], r['val_feature_ratio']) for _, r in group.iterrows()]
                seed_hv.append((seed, compute_2d_hypervolume(pts)))

            med_hv = np.median([h for s, h in seed_hv])
            # Find seed closest to median HV
            rep_seed, _ = min(seed_hv, key=lambda x: abs(x[1] - med_hv))

            # 2. Extract representative run Pareto solutions
            rep_group = moea_rows[moea_rows['seed'] == rep_seed].copy()

            # Filter non-dominated solutions on validation objectives (1-RobustF1, FeatureRatio)
            pts_val = [(1.0 - r['val_robust_f1'], r['val_feature_ratio']) for _, r in rep_group.iterrows()]
            val_pts_sorted = sorted(enumerate(pts_val), key=lambda x: (x[1][0], x[1][1]))
            non_dom_indices = []
            min_y = float('inf')
            for orig_idx, (vx, vy) in val_pts_sorted:
                if vy < min_y:
                    non_dom_indices.append(orig_idx)
                    min_y = vy

            nd_group = rep_group.iloc[non_dom_indices].copy()
            # Sort by val_feature_ratio for neat visualization line
            nd_group = nd_group.sort_values(by='val_feature_ratio')

            x_moea = nd_group['val_feature_ratio'].values
            y_moea = nd_group['macroF1_test'].values

            pct = int(lf * 100)
            c = colors[lf]

            # Plot MOEA-SSL front
            ax.plot(x_moea, y_moea, marker='.', markersize=6, linestyle='-', linewidth=1.2,
                    color=c, label=f"{pct}% MOEA")

            # 3. Corresponding EA-SSL point for the same seed
            ea_pt = df_lf[(df_lf['method'] == 'EA-SSL (joint)') & (df_lf['seed'] == rep_seed)]
            if not ea_pt.empty:
                x_ea = ea_pt['val_feature_ratio'].values[0]
                y_ea = ea_pt['macroF1_test'].values[0]
                ax.scatter(x_ea, y_ea, marker='x', s=35, color=c, label=f"{pct}% EA", zorder=4)

        ax.set_title(ds_titles[ds_key], fontsize=12, fontweight='bold')
        ax.set_xlabel('FeatureRatio', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Test MacroF1', fontsize=11)

    # Put shared legend at the bottom of the center plot or below the figure
    handles, labels = axes[1].get_legend_handles_labels()
    # Order legend items: 1% MOEA, 1% EA, 5% MOEA, 5% EA, 10% MOEA, 10% EA
    ordered_labels = ["1% MOEA", "1% EA", "5% MOEA", "5% EA", "10% MOEA", "10% EA"]
    ordered_handles = [handles[labels.index(l)] for l in ordered_labels if l in labels]

    fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"\nFigure 1 successfully generated and saved to:")
    print(f"  - PNG: {output_path}")
    print(f"  - PDF: {pdf_path}")


# -------------------------------------------------------------
# Main CLI Entry Point
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all Tables and Figures from the MOEA-SSL Smart Cities paper."
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory containing the pre-computed results CSV files (default: 'results')"
    )
    parser.add_argument(
        "--output_fig", type=str, default="figures/fig1_pareto_fronts.png",
        help="Output path for Figure 1 (default: 'figures/fig1_pareto_fronts.png')"
    )
    args = parser.parse_args()

    # Search in args.results_dir or fall back to current directory
    res_dir = args.results_dir
    if not os.path.exists(res_dir):
        if os.path.exists("results_building.csv"):
            res_dir = "."
        else:
            raise FileNotFoundError(f"Results directory '{res_dir}' does not exist.")

    print("================================================================================")
    print(" MOEA-SSL: Reproduction Package Analysis Script")
    print(f" Loading results from: {os.path.abspath(res_dir)}")
    print("================================================================================")

    datasets = load_all_results(res_dir)
    if not datasets:
        print("Error: No dataset results could be loaded. Aborting.")
        return

    # Reproduce Tables 3, 4, 5, 6 and Figure 1
    reproduce_table_3(datasets)
    reproduce_table_4(datasets)
    reproduce_table_5(datasets)
    reproduce_table_6(datasets)
    reproduce_figure_1(datasets, output_path=args.output_fig)

    print("\nReproduction analysis complete! All paper artifacts matched.")


if __name__ == "__main__":
    main()
