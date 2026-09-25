# Execution Scripts

This directory contains execution scripts for both local testing and cluster-scale execution on SLURM-managed High Performance Computing (HPC) clusters.

## 1. Quick Demonstration (`run_quick_demo.sh`)
Runs a fast smoke test (~10 seconds) on the Water Potability dataset with reduced generations to verify that the environment, models, and genetic operators are working properly:
```bash
bash scripts/run_quick_demo.sh
```

## 2. High-Performance Cluster Execution (SLURM)
The scripts `slurm_run_building.sh`, `slurm_run_grid.sh`, and `slurm_run_water.sh` execute the full experimental protocol described in Section 5 of the paper:
- **Population size**: $N = 36$
- **Generations**: $G = 50$
- **Crossover probability**: $p_{cx} = 0.85$
- **Mutation probability**: $p_{mut} = 0.35$
- **Robustness penalty**: $\alpha = 0.4$
- **Fixed Resamples**: $R = 10$
- **Seeds**: 15 seeds (0 to 14)
- **Labeled Fractions**: $\rho \in \{0.01, 0.05, 0.10\}$
- **Tasks per array**: 45 parallel jobs ($3 \text{ fractions} \times 15 \text{ seeds}$)

### Submitting Jobs:
```bash
# Submit single dataset
sbatch scripts/slurm_run_water.sh
sbatch scripts/slurm_run_grid.sh
sbatch scripts/slurm_run_building.sh
```

### Consolidating Results:
After SLURM array jobs complete, consolidate individual task outputs into the final results files:
```bash
python3 scripts/merge_results.py --input_dir results_slurm/water/csv --out_csv results/results_water.csv
python3 scripts/merge_results.py --input_dir results_slurm/grid/csv --out_csv results/results_grid.csv
python3 scripts/merge_results.py --input_dir results_slurm/building/csv --out_csv results/results_building.csv
```
