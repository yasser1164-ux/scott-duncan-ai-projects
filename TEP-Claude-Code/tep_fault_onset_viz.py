"""
TEP Fault Onset Visualization
Shows how faulty data looks identical to normal operation in samples 1-159,
then diverges from sample 160 onward when the fault is introduced.

Output: plots/tep_fault_onset_viz.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Loading data …")
fy_test = pd.read_parquet(os.path.join(DATA_DIR, "TEP_Faulty_Testing.parquet"))
ff_test = pd.read_parquet(os.path.join(DATA_DIR, "TEP_FaultFree_Testing.parquet"))

FAULT_ONSET = 160   # fault introduced at sample 160

# ── Features to plot per fault ────────────────────────────────────────────────
FAULT_CONFIG = {
    1: {
        "title": "Fault 1 — A/C Feed Ratio Step Change",
        "description": "Step change in A/C feed ratio (stream 4). Dramatic jump visible in xmv_3 and downstream sensors.",
        "features": ["xmv_3", "xmeas_19", "xmeas_3", "xmv_9"],
        "ylabels": ["D-Feed Flow Valve (%)", "Stripper Temp (°C)", "Reactor Feed (kscmh)", "Recycle Flow Valve (%)"],
    },
    6: {
        "title": "Fault 6 — A Feed Loss (Total)",
        "description": "Complete loss of feed A. Causes large pressure and flow changes throughout the plant.",
        "features": ["xmeas_16", "xmeas_7", "xmeas_13", "xmv_3"],
        "ylabels": ["Stripper Pressure (kPa)", "Reactor Pressure (kPa)", "Prod Sep Pressure (kPa)", "D-Feed Flow Valve (%)"],
    },
}

# ── Compute per-sample stats across all simulation runs ───────────────────────
def get_sample_stats(df_fault, ff_df, features):
    stats = {}
    samples  = np.arange(1, 961)
    ff_means = ff_df[features].mean()

    for feat in features:
        grp = df_fault.groupby("sample")[feat]
        stats[feat] = {
            "samples": samples,
            "median":  grp.median().values,
            "p25":     grp.quantile(0.25).values,
            "p75":     grp.quantile(0.75).values,
            "ff_mean": ff_means[feat],
        }
    return stats

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 9), constrained_layout=True)
fig.suptitle(
    "TEP Faulty Test Data — Samples 1–159 Mirror Normal Operation; Fault Active from Sample 160",
    fontsize=14, fontweight="bold",
)

FAULT_COLOR = "#C44E52"
FF_COLOR    = "#4C72B0"
SHADE_PRE   = "#d0e8ff"
SHADE_POST  = "#ffe0e0"

for row_idx, (fault_id, cfg) in enumerate(FAULT_CONFIG.items()):
    df_f  = fy_test[fy_test["faultNumber"] == fault_id]
    stats = get_sample_stats(df_f, ff_test, cfg["features"])

    for col_idx, (feat, ylabel) in enumerate(zip(cfg["features"], cfg["ylabels"])):
        ax = axes[row_idx, col_idx]
        s  = stats[feat]
        xs = s["samples"]

        ax.axvspan(1,           FAULT_ONSET - 1, color=SHADE_PRE,  alpha=0.5, zorder=0)
        ax.axvspan(FAULT_ONSET, 960,             color=SHADE_POST, alpha=0.4, zorder=0)
        ax.axhline(s["ff_mean"], color=FF_COLOR, linewidth=1.4, linestyle="--",
                   alpha=0.85, label="Fault-free mean", zorder=2)
        ax.fill_between(xs, s["p25"], s["p75"], color=FAULT_COLOR, alpha=0.25,
                        zorder=3, label="Faulty IQR (25–75%)")
        ax.plot(xs, s["median"], color=FAULT_COLOR, linewidth=1.6,
                label="Faulty median", zorder=4)
        ax.axvline(FAULT_ONSET, color="black", linewidth=1.8, linestyle=":",
                   zorder=5, label="Fault onset (sample 160)")

        if col_idx == 0:
            ax.annotate("← Normal\n   (samples 1–159)",
                        xy=(80, s["p75"][80]), xytext=(20, s["p75"].max()),
                        fontsize=7.5, color="#1a5276",
                        arrowprops=dict(arrowstyle="->", color="#1a5276", lw=0.8))
            ax.annotate("Fault active →\n(samples 160–960)",
                        xy=(200, s["median"][200]), xytext=(300, s["median"].max()),
                        fontsize=7.5, color="#922b21",
                        arrowprops=dict(arrowstyle="->", color="#922b21", lw=0.8))

        ax.set_xlim(1, 960)
        ax.set_xlabel("Sample index (3-min intervals)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

    axes[row_idx, 0].set_ylabel(
        f"{cfg['title']}\n\n{axes[row_idx, 0].get_ylabel()}", fontsize=8.5
    )

pre_patch  = mpatches.Patch(color=SHADE_PRE,  alpha=0.8,
                             label="Pre-fault window (samples 1–159)  →  looks normal, labeled faulty")
post_patch = mpatches.Patch(color=SHADE_POST, alpha=0.7,
                             label="Post-fault window (samples 160–960)  →  fault active")
fig.legend(handles=[pre_patch, post_patch], loc="lower center", ncol=2,
           fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, -0.03))

out = os.path.join(PLOTS_DIR, "tep_fault_onset_viz.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved plots/tep_fault_onset_viz.png")
plt.close()
