"""
TEP Plot Comparison — reads results CSVs and produces all model comparison plots.
Run tep_train_and_save.py first to generate the CSVs in results/.

All 7 models are plotted against each other:
  MSPC T2, MSPC SPE, MSPC Combined, Logistic Regression, LightGBM, Random Forest, XGBoost

Outputs written to plots/:
  plot_01_overall_comparison.png     — bar grid (8 metrics) + radar chart
  plot_02_per_fault_FDR.png          — FDR per fault, all 7 models
  plot_03_per_fault_FAR.png          — FAR per fault, all 7 models
  plot_04_per_fault_Precision.png    — Precision per fault, all 7 models
  plot_05_per_fault_F1.png           — F1-Score per fault, all 7 models
  plot_06_per_fault_AUC.png          — ROC-AUC per fault, all 7 models
  plot_07_per_fault_MCC.png          — MCC per fault, all 7 models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
if not os.path.exists(os.path.join(RESULTS_DIR, "overall_metrics.csv")):
    raise FileNotFoundError(
        "results/overall_metrics.csv not found. Run tep_train_and_save.py first."
    )

df_overall   = pd.read_csv(os.path.join(RESULTS_DIR, "overall_metrics.csv"), index_col="model")
df_per_fault = pd.read_csv(os.path.join(RESULTS_DIR, "per_fault_metrics.csv"))

FAULT_TYPES = list(range(1, 21))

# ── Model order, colors, short labels ────────────────────────────────────────
MODELS = [
    "MSPC T2", "MSPC SPE", "MSPC Combined",
    "Logistic Regression", "LightGBM", "Random Forest", "XGBoost",
]
COLORS = {
    "MSPC T2":             "#9B59B6",
    "MSPC SPE":            "#D7BDE2",
    "MSPC Combined":       "#6C3483",
    "Logistic Regression": "#4C72B0",
    "LightGBM":            "#DD8452",
    "Random Forest":       "#55A868",
    "XGBoost":             "#C44E52",
}
SHORT = {
    "MSPC T2":             "MSPC T2",
    "MSPC SPE":            "MSPC SPE",
    "MSPC Combined":       "MSPC\nComb.",
    "Logistic Regression": "Log. Reg.",
    "LightGBM":            "LightGBM",
    "Random Forest":       "Rand.\nForest",
    "XGBoost":             "XGBoost",
}
LOWER_BETTER = {"FAR", "Train_Time_s", "Infer_Time_s"}
c_list = [COLORS[m] for m in MODELS]

def save(fname):
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Overall Comparison  (3×3 bar grid + radar)
# ─────────────────────────────────────────────────────────────────────────────
BAR_METRICS = [
    ("Train_Time_s", "Train Time (s)"),
    ("Infer_Time_s", "Inference Time (s)"),
    ("FAR",          "False Alarm Rate"),
    ("FDR",          "FDR / Recall"),
    ("Precision",    "Precision"),
    ("F1",           "F1-Score"),
    ("AUC",          "ROC-AUC"),
    ("MCC",          "Matthews Corr. Coeff."),
]

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle(
    "TEP Fault Detection — All Models Comparison\n"
    "Post-fault evaluation (sample ≥ 160)  |  MSPC trains on fault-free data only",
    fontsize=15, fontweight="bold", y=0.99,
)
plt.subplots_adjust(hspace=0.65, wspace=0.40)

x     = np.arange(len(MODELS))
bar_w = 0.60
labels = [SHORT[m] for m in MODELS]

for ax_idx, (metric, title) in enumerate(BAR_METRICS):
    ax   = axes[ax_idx // 3, ax_idx % 3]
    vals = [df_overall.loc[m, metric] for m in MODELS]
    low  = metric in LOWER_BETTER
    bars = ax.bar(x, vals, width=bar_w, color=c_list, edgecolor="white", linewidth=0.8)
    best = (np.argmin if low else np.argmax)(vals)
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    finite = [v for v in vals if not np.isnan(v)]
    ymin = max(0, min(finite) * 0.80)
    ymax = max(finite) * 1.22
    ax.set_ylim(ymin, ymax)
    for bar, v in zip(bars, vals):
        lbl = f"{v:.2f}s" if "Time" in metric else f"{v:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (ymax - ymin) * 0.01,
                lbl, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("lower is better" if low else "higher is better", fontsize=7, color="gray")

# Radar chart in position [2, 2]
radar_ax = fig.add_subplot(3, 3, 9, polar=True)
axes[2, 2].set_visible(False)

r_keys   = ["FDR", "Precision", "F1", "AUC", "MCC", "FAR"]
r_labels = ["FDR\n(Recall)", "Precision", "F1-Score", "ROC-AUC", "MCC", "1-FAR"]
angles   = np.linspace(0, 2*np.pi, len(r_labels), endpoint=False).tolist()
angles  += angles[:1]

radar_ax.set_theta_offset(np.pi/2); radar_ax.set_theta_direction(-1)
radar_ax.set_thetagrids(np.degrees(angles[:-1]), r_labels, fontsize=8)
radar_ax.set_ylim(0, 1); radar_ax.set_yticks([0.25, 0.5, 0.75, 1.0])
radar_ax.set_yticklabels(["0.25","0.50","0.75","1.0"], fontsize=6, color="gray")
radar_ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

for name in MODELS:
    vals_r = [1 - df_overall.loc[name, k] if k == "FAR" else df_overall.loc[name, k]
              for k in r_keys]
    vals_r += vals_r[:1]
    radar_ax.plot(angles, vals_r, color=COLORS[name], linewidth=1.8)
    radar_ax.fill(angles, vals_r, color=COLORS[name], alpha=0.07)
radar_ax.set_title("Radar: Quality Metrics", fontsize=10, fontweight="bold", pad=18)

patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
patches.append(mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2,
                               label="Best per chart"))
fig.legend(handles=patches, loc="lower center", ncol=4,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.003))

save("plot_01_overall_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS 2-7 — Per-fault bar charts, one per metric
# ─────────────────────────────────────────────────────────────────────────────
PER_FAULT_METRICS = [
    ("FDR",       "FDR / Recall",         "higher = better", "plot_02_per_fault_FDR.png"),
    ("FAR",       "False Alarm Rate",      "lower = better",  "plot_03_per_fault_FAR.png"),
    ("Precision", "Precision",             "higher = better", "plot_04_per_fault_Precision.png"),
    ("F1",        "F1-Score",              "higher = better", "plot_05_per_fault_F1.png"),
    ("AUC",       "ROC-AUC",              "higher = better", "plot_06_per_fault_AUC.png"),
    ("MCC",       "Matthews Corr. Coeff.", "higher = better", "plot_07_per_fault_MCC.png"),
]

n_models = len(MODELS)
bw       = 0.11
fx       = np.arange(len(FAULT_TYPES))
offsets  = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bw

for metric, ylabel, direction, fname in PER_FAULT_METRICS:
    low = "lower" in direction
    fig, ax = plt.subplots(figsize=(22, 7))

    for i, name in enumerate(MODELS):
        vals = [
            df_per_fault.loc[
                (df_per_fault["model"] == name) & (df_per_fault["fault"] == fid),
                metric
            ].values[0]
            for fid in FAULT_TYPES
        ]
        ax.bar(fx + offsets[i], vals, width=bw,
               label=name, color=COLORS[name],
               edgecolor="white", linewidth=0.4, alpha=0.92)

    # Gold star marks the best model per fault
    for fi, fid in enumerate(FAULT_TYPES):
        sub    = df_per_fault[df_per_fault["fault"] == fid]
        scores = {m: sub.loc[sub["model"] == m, metric].values[0] for m in MODELS}
        best_m = min(scores, key=scores.get) if low else max(scores, key=scores.get)
        bi     = MODELS.index(best_m)
        ax.plot(fx[fi] + offsets[bi], scores[best_m] + 0.025,
                marker="*", color="gold", markersize=8, zorder=5,
                markeredgecolor="black", markeredgewidth=0.4)

    ax.set_title(
        f"TEP — {ylabel} by Fault Type  [all models  |  post-fault: sample ≥ 160]",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Fault Number", fontsize=12)
    ax.set_ylabel(f"{ylabel}  ({direction})", fontsize=11)
    ax.set_xticks(fx)
    ax.set_xticklabels([f"Fault {f}" for f in FAULT_TYPES],
                       rotation=40, ha="right", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.14)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Annotate known hard faults
    for fid, note in [(3, "F3\n(hard)"), (9, "F9\n(hard)"), (15, "F15\n(hard)")]:
        xi = FAULT_TYPES.index(fid)
        ax.text(xi, 1.07, note, ha="center", va="bottom", fontsize=8, color="#7F8C8D",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#BDC3C7", lw=0.8))

    star_h = Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                    markeredgecolor="black", markersize=10, label="Best per fault")
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [star_h], labels=labels_leg + ["Best per fault"],
              fontsize=9, framealpha=0.93, loc="lower right", ncol=4)

    plt.tight_layout()
    save(fname)

print("\nAll plots saved to plots/")
