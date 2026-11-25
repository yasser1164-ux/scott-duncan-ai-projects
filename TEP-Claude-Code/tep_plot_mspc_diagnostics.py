"""
TEP MSPC Diagnostic Plots — re-fits the MSPC model (~3s) and produces plots
that require the underlying PCA model object.

Optionally reads results/per_fault_metrics.csv (from tep_train_and_save.py)
to annotate contribution plots with T2 / SPE FDR values. If the CSV is absent
the plots are still generated without annotations.

Outputs written to plots/:
  diag_01_variance_explained.png     — scree plot + cumulative variance
  diag_02_control_charts.png         — T² & SPE control charts for 4 example faults
  contributions/fault_01.png … fault_20.png — variable contribution plots
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
CONTRIB_DIR = os.path.join(PLOTS_DIR, "contributions")
os.makedirs(CONTRIB_DIR, exist_ok=True)

RANDOM_SEED   = 42
FAULT_ONSET   = 160
FEATURE_COLS  = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
FAULT_TYPES   = list(range(1, 21))
VAR_THRESHOLD = 0.90
ALPHA         = 0.01

# ── Human-readable variable names (Downs & Vogel 1993) ────────────────────────
VAR_NAMES = {
    "xmeas_1":  "A Feed\n(s1)",     "xmeas_2":  "D Feed\n(s2)",     "xmeas_3":  "E Feed\n(s3)",
    "xmeas_4":  "A&C Feed\n(s4)",   "xmeas_5":  "Recycle\nFlow",    "xmeas_6":  "Rxtr Feed\nRate",
    "xmeas_7":  "Rxtr\nPressure",   "xmeas_8":  "Rxtr\nLevel",      "xmeas_9":  "Rxtr\nTemp",
    "xmeas_10": "Purge\nRate",      "xmeas_11": "Sep\nTemp",         "xmeas_12": "Sep\nLevel",
    "xmeas_13": "Sep\nPressure",    "xmeas_14": "Sep\nUnderflow",    "xmeas_15": "Strip\nLevel",
    "xmeas_16": "Strip\nPressure",  "xmeas_17": "Strip\nUnderflow",  "xmeas_18": "Strip\nTemp",
    "xmeas_19": "Strip\nSteam",     "xmeas_20": "Compressor\nWork",
    "xmeas_21": "Rxtr CW\nOut Temp","xmeas_22": "Sep CW\nOut Temp",
    "xmeas_23": "S6\nComp A",       "xmeas_24": "S6\nComp B",       "xmeas_25": "S6\nComp C",
    "xmeas_26": "S6\nComp D",       "xmeas_27": "S6\nComp E",       "xmeas_28": "S6\nComp F",
    "xmeas_29": "S9\nComp A",       "xmeas_30": "S9\nComp B",       "xmeas_31": "S9\nComp C",
    "xmeas_32": "S9\nComp D",       "xmeas_33": "S9\nComp E",       "xmeas_34": "S9\nComp F",
    "xmeas_35": "S9\nComp G",       "xmeas_36": "S9\nComp H",
    "xmeas_37": "S11\nComp D",      "xmeas_38": "S11\nComp E",      "xmeas_39": "S11\nComp F",
    "xmeas_40": "S11\nComp G",      "xmeas_41": "S11\nComp H",
    "xmv_1":  "D Feed\nValve",      "xmv_2":  "E Feed\nValve",      "xmv_3":  "A Feed\nValve",
    "xmv_4":  "A&C Feed\nValve",    "xmv_5":  "Comp\nRecycle V",    "xmv_6":  "Purge\nValve",
    "xmv_7":  "Sep Liq\nValve",     "xmv_8":  "Strip Liq\nValve",   "xmv_9":  "Strip\nSteam V",
    "xmv_10": "Rxtr CW\nFlow",      "xmv_11": "Cond CW\nFlow",
}
DISPLAY_NAMES = [VAR_NAMES[c] for c in FEATURE_COLS]

FAULT_DESCRIPTIONS = {
    1:  "Step — A/C feed ratio, B composition",
    2:  "Step — B composition",
    3:  "Step — Feed D temperature  [near-undetectable]",
    4:  "Step — Reactor cooling water inlet temp",
    5:  "Step — Condenser cooling water inlet temp",
    6:  "Step — Feed A loss (complete)",
    7:  "Step — Header pressure loss (C feed)",
    8:  "Random variation — A, B, C feed composition",
    9:  "Random variation — Feed D temperature  [hard]",
    10: "Random variation — C feed temperature",
    11: "Random variation — Reactor cooling water temp",
    12: "Random variation — Condenser cooling water temp",
    13: "Slow drift — Reaction kinetics",
    14: "Sticking — Reactor cooling water valve",
    15: "Sticking — Condenser cooling water valve  [hard]",
    16: "Unknown", 17: "Unknown", 18: "Unknown", 19: "Unknown", 20: "Unknown",
}

def save(path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {os.path.relpath(path, SCRIPT_DIR)}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA AND FIT MSPC
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data and fitting MSPC …", end=" ", flush=True)

ff_train = pd.read_parquet(os.path.join(DATA_DIR, "TEP_FaultFree_Training.parquet"))
ff_test  = pd.read_parquet(os.path.join(DATA_DIR, "TEP_FaultFree_Testing.parquet"))
fy_test  = pd.read_parquet(os.path.join(DATA_DIR, "TEP_Faulty_Testing.parquet"))

for col in FEATURE_COLS:
    ff_train[col] = ff_train[col].astype("float32")
    ff_test[col]  = ff_test[col].astype("float32")
    fy_test[col]  = fy_test[col].astype("float32")

X_train  = ff_train[FEATURE_COLS].values
scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
n, p     = Xs_train.shape

pca_full = PCA().fit(Xs_train)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
k        = int(np.searchsorted(cum_var, VAR_THRESHOLD)) + 1
pca      = PCA(n_components=k).fit(Xs_train)
lam      = pca.explained_variance_
P        = pca.components_    # (k, p) — loadings matrix

# Control limits
F_crit        = stats.f.ppf(1 - ALPHA, k, n - k)
T2_lim        = k * (n + 1) * (n - 1) / (n * (n - k)) * F_crit
ev_tail       = pca_full.explained_variance_[k:]
th1,th2,th3   = ev_tail.sum(), (ev_tail**2).sum(), (ev_tail**3).sum()
h0            = 1 - (2*th1*th3) / (3*th2**2)
za            = stats.norm.ppf(1 - ALPHA)
if h0 > 0 and th1 > 1e-10:
    SPE_lim = th1 * (1 + za*np.sqrt(2*th2*h0**2)/th1 + th2*h0*(h0-1)/th1**2)**(1/h0)
else:
    T_tr    = pca.transform(Xs_train)
    SPE_lim = np.percentile(np.sum((Xs_train - pca.inverse_transform(T_tr))**2, axis=1),
                             (1-ALPHA)*100)

print(f"done  (k={k} PCs, T2_lim={T2_lim:.1f}, SPE_lim={SPE_lim:.3f})")

def mspc_stats_normalized(X_raw):
    """Return T²/T²_lim and SPE/SPE_lim for an array of raw (unscaled) samples."""
    Xs  = scaler.transform(X_raw.astype("float32"))
    T   = pca.transform(Xs)
    T2  = np.sum((T / np.sqrt(lam))**2, axis=1)
    SPE = np.sum((Xs - pca.inverse_transform(T))**2, axis=1)
    return T2 / T2_lim, SPE / SPE_lim

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Scree + Cumulative Variance
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots …")
ev     = pca_full.explained_variance_ratio_ * 100
cum_ev = np.cumsum(ev)
pc_idx = np.arange(1, len(ev) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "PCA Variance Analysis — TEP Fault-Free Training Data (52 features)",
    fontsize=13, fontweight="bold",
)

ax = axes[0]
ax.bar(pc_idx[:40], ev[:40], color="#9B59B6", alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(k, color="red", linestyle="--", linewidth=1.5,
           label=f"Selected: {k} PCs ({cum_ev[k-1]:.1f}%)")
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Individual Variance Explained (%)", fontsize=11)
ax.set_title("Scree Plot (first 40 PCs)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(pc_idx, cum_ev, color="#9B59B6", linewidth=2.2)
ax.fill_between(pc_idx, cum_ev, alpha=0.12, color="#9B59B6")
ax.axvline(k, color="red", linestyle="--", linewidth=1.5,
           label=f"{k} PCs → {cum_ev[k-1]:.1f}%")
for pct, col in [(80, "#2ECC71"), (90, "#E74C3C"), (95, "#3498DB")]:
    n_pc = int(np.searchsorted(cum_ev / 100, pct / 100)) + 1
    ax.axhline(pct, color=col, linestyle=":", linewidth=1.2, alpha=0.8,
               label=f"{pct}% @ PC {n_pc}")
ax.set_xlabel("Number of Principal Components", fontsize=11)
ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
ax.set_title("Cumulative Variance Explained", fontsize=11, fontweight="bold")
ax.set_ylim(0, 103); ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save(os.path.join(PLOTS_DIR, "diag_01_variance_explained.png"))

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Control Charts (T² and SPE over time for 4 example faults)
# ─────────────────────────────────────────────────────────────────────────────
CHART_FAULTS = [0, 1, 6, 3]
CHART_TITLES = [
    "Fault-Free (reference)",
    "Fault 1 — A/C feed ratio step",
    "Fault 6 — total loss of feed A",
    "Fault 3 — feed D temp step (hard)",
]

fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
fig.suptitle(
    "MSPC Control Charts — T² and SPE (normalized to control limit = 1.0)\n"
    "One simulation run per fault  |  Red dashed = 99% control limit  |  Shaded = post-fault window",
    fontsize=12, fontweight="bold",
)
plt.subplots_adjust(hspace=0.35, wspace=0.28)

for ci, (fid, title) in enumerate(zip(CHART_FAULTS, CHART_TITLES)):
    if fid == 0:
        run_df = ff_test[ff_test["simulationRun"] == 1].copy()
    else:
        run_df = fy_test[(fy_test["faultNumber"] == fid) &
                         (fy_test["simulationRun"] == 1)].copy()
    run_df  = run_df.sort_values("sample").reset_index(drop=True)
    samples = run_df["sample"].values
    T2n, SPEn = mspc_stats_normalized(run_df[FEATURE_COLS].values)

    for ri, (stat, ylabel) in enumerate([(T2n, "T² / T²_limit"), (SPEn, "SPE / SPE_limit")]):
        ax  = axes[ri, ci]
        col = "#C0392B" if fid != 0 else "#2980B9"
        if fid != 0:
            ax.axvspan(FAULT_ONSET, samples.max(), color="#ffe0e0", alpha=0.45, zorder=0)
            ax.axvline(FAULT_ONSET, color="black", linewidth=1.2, linestyle=":", zorder=4,
                       label="Fault onset")
        ax.plot(samples, stat, color=col, linewidth=0.85, alpha=0.85, zorder=3)
        ax.axhline(1.0, color="red", linewidth=1.6, linestyle="--", zorder=5,
                   label="Control limit")
        ax.set_ylim(0, max(stat.max() * 1.1, 2.5))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        if ri == 0:
            ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
        if ri == 1:
            ax.set_xlabel("Sample index", fontsize=9)
        if ri == 0 and ci == 0:
            ax.legend(fontsize=7.5, loc="upper right")

save(os.path.join(PLOTS_DIR, "diag_02_control_charts.png"))

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS 3-22 — Variable Contribution Plots (one per fault)
# ─────────────────────────────────────────────────────────────────────────────

# Fault-free baseline contributions
Xs_ff         = scaler.transform(ff_test[FEATURE_COLS].values.astype("float32"))
T_ff          = pca.transform(Xs_ff)
contrib_T2_ff = Xs_ff * ((T_ff / lam) @ P)
e_ff          = Xs_ff - pca.inverse_transform(T_ff)
mean_T2_ff    = contrib_T2_ff.mean(axis=0)
mean_SPE_ff   = (e_ff**2).mean(axis=0)

# Load per-fault FDR from saved results if available
df_pf     = None
fdr_path  = os.path.join(RESULTS_DIR, "per_fault_metrics.csv")
if os.path.exists(fdr_path):
    df_pf = pd.read_csv(fdr_path)
    print("  (FDR annotations loaded from results/per_fault_metrics.csv)")
else:
    print("  (results/per_fault_metrics.csv not found — FDR annotations skipped)")

N_MEAS     = 41; N_MV = 11
bar_colors = ["#2980B9"] * N_MEAS + ["#E67E22"] * N_MV
x_pos      = np.arange(len(FEATURE_COLS))

def label_top_n(ax, x_pos, values, n=6, offset_frac=0.03):
    top_idx = np.argsort(np.abs(values))[-n:]
    yrange  = ax.get_ylim()[1] - ax.get_ylim()[0]
    for idx in top_idx:
        v    = values[idx]
        yoff = offset_frac * yrange * (1 if v >= 0 else -1)
        ax.text(x_pos[idx], v + yoff,
                DISPLAY_NAMES[idx].replace("\n", " "),
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=6.5, rotation=90, color="#1a1a1a", fontweight="bold")

print(f"\nGenerating {len(FAULT_TYPES)} contribution plots …")
for fault_id in FAULT_TYPES:
    df_f  = fy_test[(fy_test["faultNumber"] == fault_id) &
                    (fy_test["sample"] >= FAULT_ONSET)]
    Xs_f  = scaler.transform(df_f[FEATURE_COLS].values.astype("float32"))
    T_f   = pca.transform(Xs_f)

    excess_T2  = Xs_f * ((T_f / lam) @ P)
    excess_T2  = excess_T2.mean(axis=0) - mean_T2_ff

    e_f        = Xs_f - pca.inverse_transform(T_f)
    excess_SPE = (e_f**2).mean(axis=0) - mean_SPE_ff

    # FDR annotation string
    fdr_str = ""
    if df_pf is not None:
        t2_row  = df_pf[(df_pf["model"] == "MSPC T2")  & (df_pf["fault"] == fault_id)]
        spe_row = df_pf[(df_pf["model"] == "MSPC SPE") & (df_pf["fault"] == fault_id)]
        if len(t2_row) and len(spe_row):
            fdr_str = (f"   T2 FDR: {t2_row['FDR'].values[0]:.3f}"
                       f"   SPE FDR: {spe_row['FDR'].values[0]:.3f}")

    fig, axes_c = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        f"Fault {fault_id:02d}  —  {FAULT_DESCRIPTIONS[fault_id]}\n"
        f"Mean variable contributions{fdr_str}   [post-fault samples, n={len(df_f):,}]",
        fontsize=12, fontweight="bold",
    )

    for panel_idx, (excess, stat_name, limit_val) in enumerate([
        (excess_T2,  "T2",  T2_lim  / len(FEATURE_COLS)),
        (excess_SPE, "SPE", SPE_lim / len(FEATURE_COLS)),
    ]):
        ax = axes_c[panel_idx]
        bc = ["#C0392B" if v > 0 else "#2980B9" for v in excess] if stat_name == "T2" else bar_colors

        ax.bar(x_pos, excess, color=bc, edgecolor="none", alpha=0.85, width=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(limit_val, color="red", linewidth=1, linestyle="--", alpha=0.6,
                   label=f"{stat_name} limit / p  ({limit_val:.3f})")
        if stat_name == "T2":
            ax.axhline(-limit_val, color="red", linewidth=1, linestyle="--", alpha=0.6)

        label_top_n(ax, x_pos, excess, n=6)

        title_txt = (
            "T2 Contribution (excess over fault-free baseline)  | red = positive deviation, blue = negative"
            if stat_name == "T2"
            else "SPE Contribution (excess squared residual over fault-free baseline)  | higher = more deviation"
        )
        ax.set_title(title_txt, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(DISPLAY_NAMES, rotation=90, fontsize=6)
        ax.set_ylabel(f"Mean {stat_name} contribution (excess)", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.axvspan(-0.5, N_MEAS - 0.5, alpha=0.04, color="#2980B9", zorder=0)
        ax.axvspan(N_MEAS - 0.5, len(FEATURE_COLS) - 0.5, alpha=0.04, color="#E67E22", zorder=0)

    meas_patch = mpatches.Patch(color="#2980B9", alpha=0.6,
                                label="xmeas (process measurements, 1–41)")
    mv_patch   = mpatches.Patch(color="#E67E22", alpha=0.6,
                                label="xmv (manipulated variables, 1–11)")
    fig.legend(handles=[meas_patch, mv_patch], loc="lower center",
               ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out = os.path.join(CONTRIB_DIR, f"fault_{fault_id:02d}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Fault {fault_id:02d} … saved")

print(f"\nAll diagnostic plots saved to plots/")
