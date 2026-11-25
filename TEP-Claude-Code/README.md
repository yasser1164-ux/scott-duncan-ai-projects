# TEP Fault Detection: MSPC vs. Machine Learning

A benchmark comparison of **Multivariate Statistical Process Control (MSPC)** and modern **ML classifiers** for fault detection on the Tennessee Eastman Process (TEP) — a classic chemical engineering simulation used widely in process monitoring research.

---

## Background

The **Tennessee Eastman Process** (Downs & Vogel, 1993) is a realistic simulation of a chemical plant with 5 major unit operations, 52 measured variables (41 process measurements + 11 manipulated variables), and 20 distinct fault types. It has become the standard benchmark for evaluating process monitoring and fault detection methods.

This project asks a practical question: **how does traditional MSPC (PCA-based) compare to modern ML classifiers** when detecting faults across all 20 fault types?

### Key Design Choice: Post-Fault Evaluation

In the TEP dataset, faulty simulations share samples 1–159 with normal operation — the fault is only introduced at sample 160. Evaluating on the full test set is therefore misleading, as pre-fault samples are labeled "faulty" but are indistinguishable from normal. All models in this project are evaluated on **post-fault samples only (sample ≥ 160)**, which reflects realistic detection performance.

---

## Models Compared

| Model | Type | Trains On |
|---|---|---|
| MSPC T² | Hotelling T² statistic, 99% F-distribution limit | Fault-free data only |
| MSPC SPE | Squared Prediction Error (Q statistic), Jackson–Mudholkar limit | Fault-free data only |
| MSPC Combined | Alarm if T² **or** SPE exceeds limit | Fault-free data only |
| Logistic Regression | Linear classifier (SAGA solver, balanced class weight) | Labeled fault-free + faulty |
| LightGBM | Gradient boosted trees (early stopping) | Labeled fault-free + faulty |
| Random Forest | Bagged trees (balanced subsample) | Labeled fault-free + faulty |
| XGBoost | Gradient boosted trees (hist method, early stopping) | Labeled fault-free + faulty |

MSPC uses **31 principal components** (retaining 90% of variance) computed from fault-free training data. Control limits are set at the 99% confidence level.

---

## Dataset

The TEP dataset is not included in this repository due to file size. The source CSVs are publicly available on Kaggle:

**[Tennessee Eastman Process — CSV files](https://www.kaggle.com/datasets/afrniomelo/tep-csv)**

Download all four CSV files, convert them to parquet, and place them in the `data/` directory. The scripts expect parquet format for faster loading (~10s vs ~60s for CSV at this data size).

### Conversion script

```python
import pandas as pd

files = [
    "TEP_FaultFree_Training",
    "TEP_FaultFree_Testing",
    "TEP_Faulty_Training",
    "TEP_Faulty_Testing",
]

for f in files:
    df = pd.read_csv(f"{f}.csv")
    df.to_parquet(f"data/{f}.parquet", index=False)
    print(f"Converted {f}")
```

**Expected columns after conversion:** `simulationRun`, `sample`, `faultNumber` (faulty files only), `xmeas_1` … `xmeas_41`, `xmv_1` … `xmv_11`

---

## Project Structure

```
├── data/                          # Place parquet files here (gitignored)
├── results/                       # Auto-created; CSVs written here (gitignored)
├── plots/                         # Auto-created; PNGs written here (gitignored)
│   └── contributions/             # Per-fault variable contribution plots
│
├── tep_train_and_save.py          # Step 1 — train all 7 models, save results to CSV
├── tep_plot_comparison.py         # Step 2 — all model comparison plots (reads CSV)
├── tep_plot_mspc_diagnostics.py   # Step 3 — MSPC diagnostic plots (scree, control charts, contributions)
│
├── tep_fault_onset_viz.py         # Standalone — visualise the pre/post-fault window
├── tep_binary_classification.py   # Standalone — quick ML baseline (console output only)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Usage

### Recommended run order

**Step 1 — Train all models and save results** *(slow: ~5–15 min depending on hardware)*

```bash
python tep_train_and_save.py
```

Writes `results/overall_metrics.csv` and `results/per_fault_metrics.csv`. Run once; all plot scripts read from these files.

---

**Step 2 — Generate all model comparison plots** *(fast)*

```bash
python tep_plot_comparison.py
```

Outputs to `plots/`:

| File | Description |
|---|---|
| `plot_01_overall_comparison.png` | 3×3 bar grid + radar chart — all 7 models, all 8 metrics |
| `plot_02_per_fault_FDR.png` | Fault Detection Rate per fault, all 7 models |
| `plot_03_per_fault_FAR.png` | False Alarm Rate per fault, all 7 models |
| `plot_04_per_fault_Precision.png` | Precision per fault, all 7 models |
| `plot_05_per_fault_F1.png` | F1-Score per fault, all 7 models |
| `plot_06_per_fault_AUC.png` | ROC-AUC per fault, all 7 models |
| `plot_07_per_fault_MCC.png` | Matthews Correlation Coefficient per fault, all 7 models |

---

**Step 3 — Generate MSPC diagnostic plots** *(fast: ~10–30s)*

```bash
python tep_plot_mspc_diagnostics.py
```

Outputs to `plots/`:

| File | Description |
|---|---|
| `diag_01_variance_explained.png` | Scree plot and cumulative variance curve |
| `diag_02_control_charts.png` | T² and SPE control charts for 4 example faults |
| `contributions/fault_01.png` … `fault_20.png` | Variable contribution plots for each fault |

If `results/per_fault_metrics.csv` exists (from Step 1), contribution plots are annotated with T² and SPE FDR values. Otherwise they are generated without annotations.

---

### Standalone scripts

These scripts run independently without any CSV dependencies:

```bash
python tep_fault_onset_viz.py       # saves plots/tep_fault_onset_viz.png
python tep_binary_classification.py # prints a console comparison table
```

`tep_fault_onset_viz.py` visualises two faults (1 and 6) to show that pre-fault samples are indistinguishable from normal operation. `tep_binary_classification.py` is a quick reference that trains the 4 ML models on the full test set (including pre-fault) and prints metrics — useful for sanity checking but not the rigorous evaluation.

---

## Key Findings

- **LightGBM and XGBoost** are the strongest overall performers on post-fault evaluation — essentially tied on FDR (~0.79), Precision (~0.99), F1 (~0.88), and AUC (~0.91).

- **MSPC Combined** is competitive on easy step-change faults (e.g. Faults 1, 2, 4, 6) but struggles significantly on three known-hard faults: **Fault 3** (feed D temperature step — near-undetectable), **Fault 9** (random variation in feed D temperature), and **Fault 15** (sticking condenser cooling water valve).

- **MSPC trains on fault-free data only** — no labeled fault examples required. This is a major practical advantage in real processes where historical fault data is scarce.

- **Logistic Regression** underperforms the other ML models due to the nonlinear nature of the faults, despite being the fastest to train.

- **The pre-fault window matters** — FDR scores on the full test set (including pre-fault) are misleadingly low because samples 1–159 of every faulty simulation look normal. Post-fault evaluation (sample ≥ 160) gives a truer picture.

- **MSPC contribution plots** (see `plots/contributions/`) identify which process variables are responsible for each fault, providing a degree of interpretability that the ML models lack.

---

## Metrics

| Metric | Description | Better |
|---|---|---|
| FDR (Recall) | Fault Detection Rate — fraction of faulty samples correctly alarmed | Higher |
| FAR | False Alarm Rate — fraction of normal samples incorrectly alarmed | Lower |
| Precision | Fraction of alarms that are genuine faults | Higher |
| F1-Score | Harmonic mean of FDR and Precision | Higher |
| ROC-AUC | Area under the ROC curve | Higher |
| MCC | Matthews Correlation Coefficient — balanced metric for imbalanced classes | Higher |

---

## References

- Downs, J. J., & Vogel, E. F. (1993). A plant-wide industrial process control problem. *Computers & Chemical Engineering*, 17(3), 245–255.
- Chiang, L. H., Russell, E. L., & Braatz, R. D. (2000). *Fault Detection and Diagnosis in Industrial Systems*. Springer.
- Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals associated with principal component analysis. *Technometrics*, 21(3), 341–349.

---

## License

MIT — see [LICENSE](LICENSE)
