# Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-scikit--learn-orange.svg)](https://scikit-learn.org/)

This repository is the official reproduction package for the research paper:

> **Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification**  
> Francisco José Sedeño and Jamal Toutouh  
> *ITIS Software, University of Malaga, Malaga, Spain*  
> Email: `{sedenoguerrero, jamal}@uma.es`

---

## 📌 Abstract & Overview

Smart-city applications often combine scarce labeled observations with abundant unlabeled data. Semi-supervised learning (SSL) exploits both sources to enhance predictive accuracy under limited supervision. However, in IoT and urban sensing systems, predictive quality is only one concern: every sensor input requires measurement, communication, preprocessing, and energy resources.

**MOEA-SSL** is a multi-objective evolutionary approach for lightweight semi-supervised tabular classification. It reformulates the evolutionary optimization layer as a **bi-objective problem** and leverages **NSGA-II** to jointly optimize:
1. **Predictive Quality ($f_1$)**: Minimizing the negative robust validation Macro F1 score across $R=10$ resamples:
   $$f_1(z) = -\text{RobustMacroF1}(z) = -(\mu_F(z) - \alpha \sigma_F(z)), \quad \alpha = 0.4$$
2. **Measurement Complexity ($f_2$)**: Minimizing the fraction of distinct original variables required across both views:
   $$f_2(z) = \text{FeatureRatio}(z) = \frac{|S^{(1)} \cup S^{(2)}|}{d}$$

### Key Operating Points Evolved by MOEA-SSL:
- **Performance-Oriented Operating Point ($z_{\text{perf}}$)**:
  $$z_{\text{perf}} = \arg\max_{z \in P_V} \text{RobustMacroF1}_V(z)$$
  *(Ties broken by lower FeatureRatio)*
- **One-Point Performance-Retention Operating Point ($z_{1\text{pp}}$)**:
  $$z_{1\text{pp}} = \arg\min_{z \in P_V} \text{FeatureRatio}(z) \quad \text{s.t.} \quad \text{RobustMacroF1}_V(z) \ge F_V^{\max} - 0.01$$
  Exchanges up to 1 percentage point of validation performance for substantially smaller sensor sets.

---

## 📂 Repository Structure

```text
reproduction_package/
├── README.md                   <- Main documentation and reproduction guide
├── requirements.txt            <- Python dependencies for pip
├── environment.yml             <- Conda environment specification
├── LICENSE                     <- MIT License
├── .gitignore                  <- Git ignore patterns
│
├── analyze_results.py          <- Analysis & reproduction script: generates Tables 3-6 and Figure 1
├── run_experiments.py          <- Unified benchmark runner (MOEA-SSL, EA-SSL, Supervised Baselines)
├── run_independent_runs.py     <- CLI wrapper for multi-seed independent benchmark runs
├── moea2f_ssl.py               <- MOEA-SSL implementation (NSGA-II Bi-Objective)
├── ea_ssl.py                   <- EA-SSL implementation (Single-objective scalar evolutionary search)
├── baseline_supervised.py      <- Supervised learning baselines (LR, SVM, RF, HGB)
│
├── src/                        <- Shared source modules
│   ├── __init__.py
│   └── dataset_loaders.py      <- Portable loaders for Building, Grid, and Water datasets
│
├── datasets/                   <- Smart City benchmark datasets
│   ├── README.md               <- Detailed dataset documentation and citations
│   ├── building_occupancy_detection/  (10,808 samples, 7 features)
│   ├── electric_grid_stability/       (10,000 samples, 12 features)
│   └── water_patability/              (3,276 samples, 9 features)
│
├── figures/                    <- Publication figures
│   ├── fig1_pareto_fronts.png  <- Figure 1: Test-space projection of Pareto fronts (PNG)
│   └── fig1_pareto_fronts.pdf  <- Figure 1: Test-space projection of Pareto fronts (PDF)
│
└── scripts/                    <- Cluster and execution utility scripts
    ├── README.md               <- HPC and SLURM execution instructions
    ├── run_quick_demo.sh       <- Fast local smoke test (~10 seconds)
    ├── slurm_run_building.sh   <- SLURM array job for Building Occupancy (45 tasks)
    ├── slurm_run_grid.sh       <- SLURM array job for Grid Stability (45 tasks)
    ├── slurm_run_water.sh      <- SLURM array job for Water Potability (45 tasks)
    └── merge_results.py        <- Consolidates SLURM array task CSVs
```

---

## 🛠️ Environment Setup

### Option 1: Conda (Recommended)
```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate moea-ssl
```

### Option 2: Pip / Virtualenv
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Running Experiments

### 1. Fast Local Smoke Test (~10 seconds)
To verify that the code, evolutionary engine, and classifier pipelines execute cleanly on your local machine:
```bash
bash scripts/run_quick_demo.sh
```

### 2. Running via Unified Benchmark Runner
You can benchmark any dataset across custom labeled fractions ($\rho \in \{0.01, 0.05, 0.10\}$) and random seeds:
```bash
# Run on Water Potability across 5 seeds:
python3 run_experiments.py --dataset water --runs 5 --popEA 36 --generations 50 --out_csv results_water.csv

# Run on Electrical Grid Stability:
python3 run_experiments.py --dataset grid --runs 5 --popEA 36 --generations 50 --out_csv results_grid.csv

# Run on Building Occupancy Detection:
python3 run_experiments.py --dataset building --runs 5 --popEA 36 --generations 50 --out_csv results_building.csv
```

### 3. Running Individual Algorithms
Each algorithm can also be executed standalone:
```bash
# MOEA-SSL (NSGA-II Bi-Objective)
python3 moea2f_ssl.py --dataset water --popEA 36 --generations 50 --pcx 0.85 --pmut 0.35

# EA-SSL (Mono-Objective Scalar Evolutionary Search)
python3 ea_ssl.py --dataset water --popEA 36 --generations 50 --pcx 0.85 --pmut 0.35

# Supervised Baselines (Logistic Regression, Linear SVM, Random Forest, HistGradientBoosting)
python3 baseline_supervised.py --dataset water --labeled_fracs 0.01 0.05 0.10
```

### 4. Running on High-Performance Computing Clusters (SLURM)
To execute the full paper evaluation on a SLURM cluster ($3 \text{ fractions} \times 15 \text{ seeds} = 45 \text{ parallel jobs}$ per dataset):
```bash
sbatch scripts/slurm_run_water.sh
sbatch scripts/slurm_run_grid.sh
sbatch scripts/slurm_run_building.sh
```

After the cluster runs complete, merge the individual task CSVs using the provided helper:
```bash
python3 scripts/merge_results.py --input_dir results_slurm/water/csv --out_csv results_water.csv
python3 scripts/merge_results.py --input_dir results_slurm/grid/csv --out_csv results_grid.csv
python3 scripts/merge_results.py --input_dir results_slurm/building/csv --out_csv results_building.csv
```

---

## 📊 Reproducing Tables and Figures

Once experiment result files (`results_building.csv`, `results_grid.csv`, `results_water.csv`) are generated in the project root or in a specific directory:

```bash
# Analyze results from current directory:
python3 analyze_results.py --results_dir .

# Or specify a custom results directory:
python3 analyze_results.py --results_dir /path/to/results
```

This command computes and displays:
1. **Table 3**: Test MacroF1 performance across all SSL methods (`Self-training`, `Heuristic co-training`, `EA-SSL`, `MOEA-SSL`) and supervised reference models (`LR`, `SVM`, `RF`, `HGB`) with medians and interquartile ranges [IQR].
2. **Table 4**: Feature usage ($z_{\text{perf}}$ vs `EA-SSL`), reduction percentages, and paired Wilcoxon signed-rank tests with joint Holm-Bonferroni correction across the 9 conditions.
3. **Table 5**: Performance-retention analysis ($z_{\text{perf}}$ vs $z_{1\text{pp}}$), showing feature counts, retention rates, and test MacroF1 preservation.
4. **Table 6**: Multi-objective quality via Hypervolume (HV) and computational runtime (seconds) with speedup ratios.
5. **Figure 1**: Generates the 3-panel test-space projection of representative validation Pareto fronts saved to `figures/fig1_pareto_fronts.png` and `figures/fig1_pareto_fronts.pdf`.

---

## 📈 Paper Experimental Results Reference

For direct reference, the benchmark results reported in the paper are summarized below:

### Table 3: Test MacroF1 (Median [IQR])
| Dataset | $\rho$ | Self-Training | Heuristic Co-Train | EA-SSL | MOEA-SSL ($z_{\text{perf}}$) | Best Supervised |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Building** | 1% | 0.950 [0.055] | 0.881 [0.040] | 0.977 [0.000] | **0.977 [0.000]** | 0.977 [0.005] (SVM) |
| | 5% | 0.977 [0.000] | 0.947 [0.014] | 0.977 [0.000] | **0.977 [0.000]** | 0.978 [0.000] (SVM) |
| | 10% | 0.977 [0.000] | 0.948 [0.014] | 0.977 [0.000] | **0.977 [0.001]** | 0.978 [0.001] (SVM) |
| **Grid** | 1% | 0.683 [0.076] | 0.660 [0.166] | 0.746 [0.032] | **0.755 [0.021]** | 0.737 [0.040] (SVM) |
| | 5% | 0.757 [0.020] | 0.533 [0.174] | 0.764 [0.017] | **0.770 [0.010]** | 0.829 [0.016] (HGB) |
| | 10% | 0.782 [0.007] | 0.728 [0.026] | 0.761 [0.010] | **0.777 [0.009]** † | 0.862 [0.013] (HGB) |
| **Water** | 1% | 0.424 [0.066] | 0.379 [0.011] | 0.491 [0.031] | **0.500 [0.011]** † | 0.495 [0.028] (SVM) |
| | 5% | 0.383 [0.038] | 0.379 [0.010] | 0.471 [0.027] | **0.506 [0.022]** † | 0.517 [0.029] (HGB) |
| | 10% | 0.385 [0.011] | 0.379 [0.003] | 0.439 [0.039] | **0.471 [0.071]** | 0.536 [0.016] (HGB) |

*† denotes statistically significant difference against EA-SSL after Holm correction ($p < 0.05$).*

### Table 4: Feature Usage Reduction ($z_{\text{perf}}$ vs EA-SSL)
| Dataset | $\rho$ | MOEA-SSL | EA-SSL | Reduction (%) | $p_{\text{Holm}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Building** | 1% | 0.429 (3/7) | 0.857 (6/7) | **50.0%** | $0.0005$ |
| | 5% | 0.429 (3/7) | 0.714 (5/7) | **40.0%** | $0.010$ |
| | 10% | 0.286 (2/7) | 0.714 (5/7) | **60.0%** | $0.007$ |
| **Grid** | 1% | 0.667 (8/12) | 0.833 (10/12) | **20.0%** | $0.011$ |
| | 5% | 0.667 (8/12) | 0.833 (10/12) | **20.0%** | $0.010$ |
| | 10% | 0.750 (9/12) | 0.917 (11/12) | **18.2%** | $0.014$ |
| **Water** | 1% | 0.444 (4/9) | 0.667 (6/9) | **33.3%** | $0.045$ |
| | 5% | 0.556 (5/9) | 0.667 (6/9) | **16.7%** | $0.045$ |
| | 10% | 0.556 (5/9) | 0.778 (7/9) | **28.6%** | $0.013$ |

*All 9 paired FeatureRatio differences remain statistically significant after joint Holm correction.*

---

## 📊 Datasets Summary

| Dataset | Domain | Samples | Features ($d$) | Classes | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Building Occupancy** | Smart buildings | 10,808 | 7 | 2 | Office occupancy estimation from temperature, light, CO2, humidity sensors [Candanedo, 2016]. |
| **Grid Stability** | Smart energy | 10,000 | 12 | 2 | Decentralized 4-node electrical grid stability prediction [Arzamasov, 2018]. |
| **Water Potability** | Smart water | 3,276 | 9 | 2 | Drinking water quality and chemical potability safety assessment [Kadiwal, 2021]. |

For complete preprocessing details, variable lists, and data citations, see [`datasets/README.md`](datasets/README.md).

---

## 📜 Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{sedeno2026moeassl,
  title={Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification},
  author={Sede{\~n}o, Francisco Jos{\'e} and Toutouh, Jamal},
  journal={arXiv preprint},
  year={2026},
  publisher={Springer}
}
```

---

## 🤝 Acknowledgments

This work has been partially funded by:
- **AIM-Zero** (PID2024-158752OB-I00)
- **AGENT-SAM** (PID2025-174277OA-I00)
- High-Performance Computing infrastructure provided by the Spanish Supercomputing Network (RES).

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
