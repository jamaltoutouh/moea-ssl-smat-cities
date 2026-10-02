# Smart City Benchmark Datasets

This directory contains the three Smart City classification datasets evaluated in:
**"Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification"**
by Francisco José Sedeño and Jamal Toutouh (ITIS Software, University of Malaga).

---

## Summary of Dataset Characteristics (Table 2 in Paper)

| Dataset | Domain | Samples | Features ($d$) | Classes | Target Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Building Occupancy** | Smart buildings | 10,808 | 7 | 2 | Room occupancy from environmental IoT sensors |
| **Grid Stability** | Smart energy | 10,000 | 12 | 2 | Decentralized smart electric grid dynamic stability |
| **Water Potability** | Smart water | 3,276 | 9 | 2 | Drinking water quality and chemical potability |

---

## 1. Building Occupancy Detection
- **Folder**: `datasets/building_occupancy_detection/`
- **Files**: `datatraining.txt` (8,143 samples), `datatest.txt` (2,665 samples), `datatest2.txt`
- **Predictors (7 features)**:
  - `Temperature` (°C)
  - `Humidity` (%)
  - `Light` (Lux)
  - `CO2` (ppm)
  - `HumidityRatio` (kg water / kg dry air)
  - `Hour` (Engineered timestamp feature, 0–23)
  - `IsWeekend` (Engineered calendar feature, 0 or 1)
- **Target**: `Occupancy` (0 = Unoccupied, 1 = Occupied).
- **Data Splitting Protocol**: Uses the official repository training partition (`datatraining.txt`, 8,143 samples) and first held-out test partition (`datatest.txt`, 2,665 samples). 20% of the training partition is held out for validation (1,629 samples). The remainder forms $D_L \cup D_U$, where labeled pool $D_L$ is formed using labeled fraction $\rho \in \{0.01, 0.05, 0.10\}$.
- **Reference**:
  > Candanedo, L.M., Feldheim, V.: Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Energy and Buildings 112, 28–39 (2016). UCI Machine Learning Repository.

---

## 2. Electrical Grid Stability
- **Folder**: `datasets/electric_grid_stability/`
- **File**: `Data_for_UCI_named.csv` (10,000 samples)
- **Predictors (12 features)**:
  - $\tau_1, \tau_2, \tau_3, \tau_4$: Reaction times of electricity network participants (supplier and consumers).
  - $p_1, p_2, p_3, p_4$: Power produced ($p_1 > 0$) or consumed ($p_2, p_3, p_4 < 0$).
  - $g_1, g_2, g_3, g_4$: Price elasticity coefficients of participants.
- **Target**: `stabf` (Binary stability: `stable` vs `unstable`). The continuous root `stab` (maximal real part of characteristic differential equation roots) is excluded to prevent direct target leakage. The redundant variable $p_1$ is retained because feature redundancy is explicitly part of the optimization problem.
- **Data Splitting Protocol**: Stratified 70/30 train/test split (3,000 test samples held out). 20% of the remaining development partition is used for validation. The remainder forms $D_L \cup D_U$ with $\rho \in \{0.01, 0.05, 0.10\}$.
- **Reference**:
  > Arzamasov, V., Böhm, K., Jochem, P.: Towards concise models of grid stability. In: IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm), pp. 1–6 (2018). UCI Machine Learning Repository.

---

## 3. Drinking Water Potability
- **Folder**: `datasets/water_patability/`
- **File**: `water_potability.csv` (3,276 samples)
- **Predictors (9 features)**:
  - `ph`: Water pH level (0–14)
  - `Hardness`: Capacity of water to precipitate soap in mg/L
  - `Solids`: Total dissolved solids (TDS) in ppm
  - `Chloramines`: Chloramine concentration in ppm
  - `Sulfate`: Sulfate concentration in mg/L
  - `Conductivity`: Electrical conductivity in $\mu$S/cm
  - `Organic_carbon`: Total organic carbon in ppm
  - `Trihalomethanes`: THM concentration in $\mu$g/L
  - `Turbidity`: Water clarity measure in NTU
- **Target**: `Potability` (0 = Not Potable / unsafe, 1 = Potable / safe for consumption).
- **Missing Values & Preprocessing**: Missing observations are median-imputed strictly using parameters fitted on the training split during evaluation.
- **Data Splitting Protocol**: Stratified 70/30 train/test split (983 test samples held out). 20% of the development partition is held out for validation (459 samples). The remainder forms $D_L \cup D_U$ with $\rho \in \{0.01, 0.05, 0.10\}$.
- **Reference**:
  > Kadiwal, A.: Water Quality: Drinking Water Potability. Kaggle dataset (2021). Available at: https://www.kaggle.com/datasets/adityakadiwal/water-potability
