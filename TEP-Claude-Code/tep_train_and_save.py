"""
TEP Train and Save — runs all 7 models once and writes results to CSV.

Models trained:
  MSPC T2       — PCA Hotelling T² statistic (trains on fault-free data only)
  MSPC SPE      — PCA Q / SPE statistic      (trains on fault-free data only)
  MSPC Combined — alarm if T² OR SPE exceeds limit
  Logistic Regression
  LightGBM
  Random Forest
  XGBoost

Evaluation: post-fault only (sample >= 160), removes pre-fault ambiguity.

Outputs written to results/:
  overall_metrics.csv     — one row per model, all 8 metrics + timing
  per_fault_metrics.csv   — one row per (model x fault_id), all 6 quality metrics

Run this script once before running tep_plot_comparison.py or
tep_plot_mspc_diagnostics.py.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix,
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR   = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED   = 42
FAULT_ONSET   = 160
FEATURE_COLS  = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
FAULT_TYPES   = list(range(1, 21))
VAR_THRESHOLD = 0.90   # PCA variance to retain
ALPHA         = 0.01   # 99% confidence limits

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading parquet files …")
t0 = time.time()

ff_train = pd.read_parquet(os.path.join(DATA_DIR, "TEP_FaultFree_Training.parquet"))
ff_test  = pd.read_parquet(os.path.join(DATA_DIR, "TEP_FaultFree_Testing.parquet"))
fy_train = pd.read_parquet(os.path.join(DATA_DIR, "TEP_Faulty_Training.parquet"))
fy_test  = pd.read_parquet(os.path.join(DATA_DIR, "TEP_Faulty_Testing.parquet"))

for df in [ff_train, ff_test, fy_train, fy_test]:
    if "faultNumber" not in df.columns:
        df["faultNumber"] = 0

ff_train["label"] = 0;  ff_test["label"] = 0
fy_train["label"] = 1;  fy_test["label"] = 1

print(f"  Loaded {len(ff_train):,} + {len(ff_test):,} fault-free rows  "
      f"and {len(fy_train):,} + {len(fy_test):,} faulty rows  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SUBSAMPLE
#     Training faulty: stratified 500k across 20 faults
#     Test faulty:     post-fault samples only (sample >= 160), stratified 960k
# ─────────────────────────────────────────────────────────────────────────────
def subsample_by_fault(df, total, rng):
    per_group = total // df["faultNumber"].nunique()
    return pd.concat(
        [g.sample(n=min(len(g), per_group), random_state=rng)
         for _, g in df.groupby("faultNumber")],
        ignore_index=True,
    )

rng = np.random.RandomState(RANDOM_SEED)

fy_train_sub = subsample_by_fault(fy_train, 500_000, rng)
del fy_train

fy_test_post = fy_test[fy_test["sample"] >= FAULT_ONSET].copy()
fy_test_sub  = subsample_by_fault(fy_test_post, 960_000, rng)
del fy_test, fy_test_post

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BUILD TRAIN / TEST ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
ml_train_df = pd.concat([ff_train, fy_train_sub], ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_df = pd.concat([ff_test, fy_test_sub], ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

for col in FEATURE_COLS:
    ff_train[col]    = ff_train[col].astype("float32")
    ml_train_df[col] = ml_train_df[col].astype("float32")
    test_df[col]     = test_df[col].astype("float32")

# MSPC trains on fault-free only; ML trains on combined
X_mspc_train = ff_train[FEATURE_COLS].values
X_ml_train   = ml_train_df[FEATURE_COLS].values
y_ml_train   = ml_train_df["label"].values.astype("int32")
X_test       = test_df[FEATURE_COLS].values
y_test       = test_df["label"].values.astype("int32")
test_faults  = test_df["faultNumber"].values

del ff_train, ff_test, fy_train_sub, fy_test_sub, ml_train_df

print(f"\n  MSPC train (fault-free only) : {len(X_mspc_train):>10,}")
print(f"  ML train                     : {len(X_ml_train):>10,}")
print(f"  Test (post-fault, sample>=160): {len(X_test):>10,}  "
      f"(fault-free={int((y_test==0).sum()):,}  faulty={int((y_test==1).sum()):,})")

# 10% validation split for early stopping (LightGBM / XGBoost)
val_n  = int(0.10 * len(X_ml_train))
shuf   = np.random.RandomState(RANDOM_SEED).permutation(len(X_ml_train))
X_tr,  y_tr  = X_ml_train[shuf[val_n:]], y_ml_train[shuf[val_n:]]
X_val, y_val = X_ml_train[shuf[:val_n]], y_ml_train[shuf[:val_n]]

# Scaler for Logistic Regression (fit on full training set)
lr_scaler     = StandardScaler()
X_ml_train_sc = lr_scaler.fit_transform(X_ml_train)
X_test_sc     = lr_scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIT MSPC  (PCA on fault-free training data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Fitting MSPC …", end=" ", flush=True)
t0 = time.time()

mspc_scaler = StandardScaler()
Xs_train    = mspc_scaler.fit_transform(X_mspc_train)
n, _        = Xs_train.shape

pca_full = PCA().fit(Xs_train)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
k        = int(np.searchsorted(cum_var, VAR_THRESHOLD)) + 1
pca      = PCA(n_components=k).fit(Xs_train)
lam      = pca.explained_variance_

# T² control limit — exact F-distribution (Phase II)
F_crit = stats.f.ppf(1 - ALPHA, k, n - k)
T2_lim = k * (n + 1) * (n - 1) / (n * (n - k)) * F_crit

# SPE control limit — Jackson–Mudholkar approximation
ev_tail       = pca_full.explained_variance_[k:]
th1, th2, th3 = ev_tail.sum(), (ev_tail**2).sum(), (ev_tail**3).sum()
h0            = 1 - (2 * th1 * th3) / (3 * th2**2)
za            = stats.norm.ppf(1 - ALPHA)
if h0 > 0 and th1 > 1e-10:
    SPE_lim = th1 * (1 + za * np.sqrt(2*th2*h0**2)/th1 + th2*h0*(h0-1)/th1**2)**(1/h0)
else:
    T_tr    = pca.transform(Xs_train)
    SPE_lim = np.percentile(np.sum((Xs_train - pca.inverse_transform(T_tr))**2, axis=1),
                             (1-ALPHA)*100)

mspc_train_time = time.time() - t0
print(f"{mspc_train_time:.2f}s  |  k={k} PCs, T2_lim={T2_lim:.1f}, SPE_lim={SPE_lim:.3f}")

# Compute T² and SPE on test set
t0       = time.time()
Xs_test  = mspc_scaler.transform(X_test)
T_test   = pca.transform(Xs_test)
T2_test  = np.sum((T_test / np.sqrt(lam))**2, axis=1)
SPE_test = np.sum((Xs_test - pca.inverse_transform(T_test))**2, axis=1)
mspc_infer_time = time.time() - t0

pred_T2   = (T2_test  > T2_lim).astype(int)
pred_SPE  = (SPE_test > SPE_lim).astype(int)
pred_comb = (pred_T2 | pred_SPE).astype(int)
score_T2   = T2_test  / T2_lim
score_SPE  = SPE_test / SPE_lim
score_comb = np.maximum(score_T2, score_SPE)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training ML models …")

model_preds = {}   # name -> (y_pred, y_score, train_time, infer_time)

# — Logistic Regression (trained on full training set, scaled) —
print("  Logistic Regression …", end=" ", flush=True)
t  = time.time()
lr = LogisticRegression(solver="saga", max_iter=100, tol=1e-3,
                        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
lr.fit(X_ml_train_sc, y_ml_train)
tr_t = time.time() - t
t    = time.time()
pred_lr  = lr.predict(X_test_sc)
score_lr = lr.predict_proba(X_test_sc)[:, 1]
model_preds["Logistic Regression"] = (pred_lr, score_lr, tr_t, time.time() - t)
print(f"{tr_t:.1f}s")

# — LightGBM (90/10 split for early stopping) —
print("  LightGBM …", end=" ", flush=True)
t    = time.time()
lgbm = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=63, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
    is_unbalance=True, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
         callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
tr_t = time.time() - t
t    = time.time()
pred_lgbm  = lgbm.predict(X_test)
score_lgbm = lgbm.predict_proba(X_test)[:, 1]
model_preds["LightGBM"] = (pred_lgbm, score_lgbm, tr_t, time.time() - t)
print(f"{tr_t:.1f}s  |  best iteration: {lgbm.best_iteration_}")

# — Random Forest (trained on full training set) —
print("  Random Forest …", end=" ", flush=True)
t  = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=15, max_features="sqrt",
                             max_samples=0.5, class_weight="balanced_subsample",
                             n_jobs=-1, random_state=RANDOM_SEED)
rf.fit(X_ml_train, y_ml_train)
tr_t = time.time() - t
t    = time.time()
pred_rf  = rf.predict(X_test)
score_rf = rf.predict_proba(X_test)[:, 1]
model_preds["Random Forest"] = (pred_rf, score_rf, tr_t, time.time() - t)
print(f"{tr_t:.1f}s")

# — XGBoost (90/10 split for early stopping) —
print("  XGBoost …", end=" ", flush=True)
t    = time.time()
spw  = int((y_ml_train==0).sum()) / int((y_ml_train==1).sum())
xgbm = xgb.XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    scale_pos_weight=spw, eval_metric="logloss",
    early_stopping_rounds=20, random_state=RANDOM_SEED, verbosity=0)
xgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
tr_t = time.time() - t
t    = time.time()
pred_xgb  = xgbm.predict(X_test)
score_xgb = xgbm.predict_proba(X_test)[:, 1]
model_preds["XGBoost"] = (pred_xgb, score_xgb, tr_t, time.time() - t)
print(f"{tr_t:.1f}s  |  best iteration: {xgbm.best_iteration}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  METRIC HELPER
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "FDR":       round(tp/(tp+fn) if (tp+fn)>0 else 0.0, 4),
        "FAR":       round(fp/(fp+tn) if (fp+tn)>0 else 0.0, 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(y_true, y_score), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7.  OVERALL METRICS
# ─────────────────────────────────────────────────────────────────────────────
ALL_PREDS = {
    "MSPC T2":             (pred_T2,   score_T2,   mspc_train_time, mspc_infer_time),
    "MSPC SPE":            (pred_SPE,  score_SPE,  mspc_train_time, mspc_infer_time),
    "MSPC Combined":       (pred_comb, score_comb, mspc_train_time, mspc_infer_time),
    "Logistic Regression": model_preds["Logistic Regression"],
    "LightGBM":            model_preds["LightGBM"],
    "Random Forest":       model_preds["Random Forest"],
    "XGBoost":             model_preds["XGBoost"],
}

overall_rows = []
for name, (pred, score, tr_t, inf_t) in ALL_PREDS.items():
    m = compute_metrics(y_test, pred, score)
    m["Train_Time_s"] = round(tr_t, 2)
    m["Infer_Time_s"] = round(inf_t, 4)
    m["model"]        = name
    overall_rows.append(m)

df_overall = pd.DataFrame(overall_rows).set_index("model")
df_overall.to_csv(os.path.join(RESULTS_DIR, "overall_metrics.csv"))

# Console summary
print("\n\n" + "=" * 72)
print("POST-FAULT COMPARISON  (sample >= 160)")
print("=" * 72)
col_w = 16
metrics_order = ["Train_Time_s", "Infer_Time_s", "FDR", "FAR", "Precision", "F1", "AUC", "MCC"]
print(f"{'Model':<22}", end="")
for m in metrics_order:
    print(f"{m:>{col_w}}", end="")
print()
print("-" * (22 + col_w * len(metrics_order)))
for name in df_overall.index:
    print(f"{name:<22}", end="")
    for m in metrics_order:
        print(f"{df_overall.loc[name, m]:>{col_w}}", end="")
    print()
print(f"\nSaved results/overall_metrics.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PER-FAULT METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing per-fault metrics …")
ff_mask = test_faults == 0
records = []

for fid in FAULT_TYPES:
    mask = ff_mask | (test_faults == fid)
    yt   = y_test[mask]
    for name, (pred, score, _, __) in ALL_PREDS.items():
        m = compute_metrics(yt, pred[mask], score[mask])
        m["model"] = name
        m["fault"] = fid
        records.append(m)

df_per_fault = pd.DataFrame(records)[["model", "fault", "FDR", "FAR", "Precision", "F1", "AUC", "MCC"]]
df_per_fault.to_csv(os.path.join(RESULTS_DIR, "per_fault_metrics.csv"), index=False)
print(f"Saved results/per_fault_metrics.csv")
print("\nAll done. Run tep_plot_comparison.py and tep_plot_mspc_diagnostics.py to generate plots.")
