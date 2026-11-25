"""
TEP Binary Fault Detection — ML Baseline
Tennessee Eastman Process: fault-free (label=0) vs. faulty (label=1).
Trains on the full test set (including pre-fault window) as a quick reference.
Compares Logistic Regression, LightGBM, Random Forest, and XGBoost.

Note: for the rigorous post-fault evaluation (sample >= 160) and the full
MSPC vs. ML comparison, run tep_train_and_save.py instead.

Output: console table only (no saved files)
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.dirname(SCRIPT_DIR)
RANDOM_SEED  = 42
FEATURE_COLS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading parquet files …")
t0 = time.time()

ff_train = pd.read_parquet(os.path.join(DATA_DIR, "TEP_FaultFree_Training.parquet"))
ff_test  = pd.read_parquet(os.path.join(DATA_DIR, "TEP_FaultFree_Testing.parquet"))
fy_train = pd.read_parquet(os.path.join(DATA_DIR, "TEP_Faulty_Training.parquet"))
fy_test  = pd.read_parquet(os.path.join(DATA_DIR, "TEP_Faulty_Testing.parquet"))

print(f"  Fault-free train : {len(ff_train):>10,} rows")
print(f"  Fault-free test  : {len(ff_test):>10,} rows")
print(f"  Faulty train     : {len(fy_train):>10,} rows")
print(f"  Faulty test      : {len(fy_test):>10,} rows")
print(f"  Loaded in {time.time() - t0:.1f}s")

# ── 2. LABELS ─────────────────────────────────────────────────────────────────
ff_train["label"] = 0;  ff_test["label"] = 0
fy_train["label"] = 1;  fy_test["label"] = 1

# ── 3. SUBSAMPLE FAULTY ───────────────────────────────────────────────────────
def subsample_by_fault(df, total_target, rng):
    groups    = df["faultNumber"].unique()
    per_group = total_target // len(groups)
    return pd.concat(
        [df[df["faultNumber"] == g].sample(n=min(len(df[df["faultNumber"] == g]), per_group),
                                            random_state=rng)
         for g in sorted(groups)],
        ignore_index=True,
    )

rng          = np.random.RandomState(RANDOM_SEED)
fy_train_sub = subsample_by_fault(fy_train, 500_000, rng)
fy_test_sub  = subsample_by_fault(fy_test,  960_000, rng)
del fy_train, fy_test

# ── 4. COMBINE ────────────────────────────────────────────────────────────────
train_df = pd.concat([ff_train, fy_train_sub], ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_df  = pd.concat([ff_test,  fy_test_sub],  ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

train_df[FEATURE_COLS] = train_df[FEATURE_COLS].astype("float32")
test_df[FEATURE_COLS]  = test_df[FEATURE_COLS].astype("float32")

X_train = train_df[FEATURE_COLS].values;  y_train = train_df["label"].values.astype("int32")
X_test  = test_df[FEATURE_COLS].values;   y_test  = test_df["label"].values.astype("int32")

print(f"\n  Train : {len(X_train):,}   Test : {len(X_test):,}")
del ff_train, ff_test, fy_train_sub, fy_test_sub, train_df, test_df

# ── 5. VAL SPLIT + SCALING ────────────────────────────────────────────────────
val_n  = int(0.10 * len(X_train))
idx    = np.arange(len(X_train)); np.random.seed(RANDOM_SEED); np.random.shuffle(idx)
X_tr,  y_tr  = X_train[idx[val_n:]], y_train[idx[val_n:]]
X_val, y_val = X_train[idx[:val_n]], y_train[idx[:val_n]]

scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, train_time, infer_time):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Train Time (s)": round(train_time, 2),
        "Infer Time (s)": round(infer_time, 4),
        "FDR (Recall)":   round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "FAR":            round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "Precision":      round(precision_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":       round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":        round(roc_auc_score(y_true, y_prob), 4),
        "MCC":            round(matthews_corrcoef(y_true, y_pred), 4),
    }

results = {}

# ── MODEL 1: LOGISTIC REGRESSION ─────────────────────────────────────────────
print("\n" + "=" * 60 + "\nTraining Logistic Regression …")
lr = LogisticRegression(solver="saga", max_iter=100, tol=1e-3,
                        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
t = time.time(); lr.fit(X_train_sc, y_train); tr_t = time.time() - t
print(f"  Train time: {tr_t:.2f}s")
t = time.time(); lr_pred = lr.predict(X_test_sc); lr_prob = lr.predict_proba(X_test_sc)[:, 1]
results["Logistic Regression"] = compute_metrics(y_test, lr_pred, lr_prob, tr_t, time.time() - t)

# ── MODEL 2: LightGBM ─────────────────────────────────────────────────────────
print("\n" + "=" * 60 + "\nTraining LightGBM …")
lgbm_model = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=63, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
    is_unbalance=True, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
t = time.time()
lgbm_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
tr_t = time.time() - t
print(f"  Train time: {tr_t:.2f}s  |  Best iteration: {lgbm_model.best_iteration_}")
t = time.time(); lgbm_pred = lgbm_model.predict(X_test); lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]
results["LightGBM"] = compute_metrics(y_test, lgbm_pred, lgbm_prob, tr_t, time.time() - t)

# ── MODEL 3: RANDOM FOREST ────────────────────────────────────────────────────
print("\n" + "=" * 60 + "\nTraining Random Forest …")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, max_features="sqrt",
                             max_samples=0.5, class_weight="balanced_subsample",
                             n_jobs=-1, random_state=RANDOM_SEED)
t = time.time(); rf.fit(X_train, y_train); tr_t = time.time() - t
print(f"  Train time: {tr_t:.2f}s")
t = time.time(); rf_pred = rf.predict(X_test); rf_prob = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = compute_metrics(y_test, rf_pred, rf_prob, tr_t, time.time() - t)

# ── MODEL 4: XGBoost ──────────────────────────────────────────────────────────
print("\n" + "=" * 60 + "\nTraining XGBoost …")
spw = int((y_train == 0).sum()) / int((y_train == 1).sum())
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    scale_pos_weight=spw, eval_metric="logloss",
    early_stopping_rounds=20, random_state=RANDOM_SEED, verbosity=0)
t = time.time()
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
tr_t = time.time() - t
print(f"  Train time: {tr_t:.2f}s  |  Best iteration: {xgb_model.best_iteration}")
t = time.time(); xgb_pred = xgb_model.predict(X_test); xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
results["XGBoost"] = compute_metrics(y_test, xgb_pred, xgb_prob, tr_t, time.time() - t)

# ── PRINT TABLE ───────────────────────────────────────────────────────────────
metrics_order = ["Train Time (s)", "Infer Time (s)", "FDR (Recall)", "FAR",
                 "Precision", "F1-Score", "ROC-AUC", "MCC"]
models = list(results.keys())
col_w  = 22

print("\n\n" + "=" * 70 + "\nMODEL COMPARISON TABLE\n" + "=" * 70)
print(f"{'Metric':<20}", end="")
for m in models:
    print(f"{m:>{col_w}}", end="")
print()
print("-" * (20 + col_w * len(models)))
for metric in metrics_order:
    print(f"{metric:<20}", end="")
    for m in models:
        print(f"{results[m][metric]:>{col_w}}", end="")
    print()

lower_is_better = {"Train Time (s)", "Infer Time (s)", "FAR"}
print("\n" + "=" * 70 + "\nBEST MODEL PER METRIC\n" + "=" * 70)
for metric in metrics_order:
    scores = {m: results[m][metric] for m in models}
    best   = min(scores, key=scores.get) if metric in lower_is_better else max(scores, key=scores.get)
    print(f"  {metric:<18} ->  {best}  ({scores[best]})")

print("\nDone.")
