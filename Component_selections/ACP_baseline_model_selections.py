# ACP_baseline_train_vs_test.py
# -*- coding: utf-8 -*-
"""
- Read antiCP2.txt (merged dataset), randomly split 1/6 as TEST and the rest as TRAIN (stratified)
- Features: AAC(20) + CKSAAP_k (k=0..5 => 2400) + PCP16(16) = 2436 dims
- Training: fit each base learner with the **full TRAIN** set to obtain the final model
- Threshold: select a single threshold on **training-set probabilities** using ROC-Youden (TPR-FPR)
- Evaluation: with that threshold, compute ACC/Precision/Recall/F1/AUC/MCC for TRAIN (apparent) and TEST (holdout)
- Output: print to console + save to baseline_holdout_outputs/xxx.xlsx (two sheets: Train_Apparent & Test_Holdout)
- GPU: prefer GPU for XGBoost/LightGBM/CatBoost, fall back to CPU if unavailable
- Remove SVC(RBF)
"""

import os, time, warnings, gc
from datetime import datetime
from itertools import product
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef, roc_curve)
from sklearn.base import clone, BaseEstimator, ClassifierMixin

# Linear / generalized linear / nearest neighbors / SVM (linear) / trees / ensembles / Naive Bayes / MLP
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

# GPU-friendly ensembles
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# ============== Data reading (odd lines are headers, even lines are sequences) ==============
def read_fasta_pair_lines(file_path: str) -> Tuple[List[str], np.ndarray]:
    sequences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for i in range(0, len(lines), 2):
        header = lines[i]
        seq = lines[i + 1].strip().upper().replace(" ", "")
        label = 1 if "positive" in header.lower() else 0
        sequences.append(seq)
        labels.append(label)
    return sequences, np.array(labels, dtype=np.int32)

# ============== Features: AAC + CKSAAP_k + PCP16 (2436 dims) ==============
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
DIPEPTIDES = [''.join(p) for p in product(AA20, repeat=2)]  # 400

PCP16_MAP = {
    "acidic": set("DE"),
    "aliphatic": set("ILV"),
    "aromatic": set("FYW"),
    "basic": set("KRH"),
    "charged": set("DEKRH"),
    "cyclic": set("P"),
    "hydrophilic": set("RNDQEHKST"),
    "hydrophobic": set("AILMFWV"),
    "hydroxylic": set("ST"),
    "neutral_pH": set("ACFGHILMNPQSTVWY"),
    "nonpolar": set("ACFGILMPVWY"),
    "polar": set("RNDQEHKTYS".replace(" ", "")),
    "small": set("ACDGNPSTV"),
    "large": set("EFHKLMQRWY"),
    "sulfur": set("CM"),
    "tiny": set("ACGST"),
}
PCP16_KEYS = list(PCP16_MAP.keys())

def _check_seq(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "")
    if not seq or any(ch not in AA_SET for ch in seq):
        raise ValueError(f"Invalid residues in sequence: {seq}")
    return seq

def aac_vector(sequence: str) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    counts = {aa: 0.0 for aa in AA20}
    for ch in s:
        counts[ch] += 1.0
    if L > 0:
        for aa in counts:
            counts[aa] /= L
    return np.array([counts[aa] for aa in AA20], dtype=np.float32)  # 20

def cksaap_vector(sequence: str, k_max: int = 5) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    out = []
    for k in range(k_max + 1):
        counts = dict.fromkeys(DIPEPTIDES, 0.0)
        if L >= k + 2:
            for i in range(L - 1 - k):
                pair = s[i] + s[i + 1 + k]
                counts[pair] += 1.0
            total = sum(counts.values())
            if total > 0:
                for dp in counts:
                    counts[dp] /= total
        out.extend([counts[dp] for dp in DIPEPTIDES])
    return np.array(out, dtype=np.float32)  # (k_max+1)*400 = 2400 when k_max=5

def pcp16_vector(sequence: str) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    return np.array([sum(ch in PCP16_MAP[key] for ch in s) / L for key in PCP16_KEYS], dtype=np.float32)  # 16

def extract_features(seqs: List[str], k_max_cksaap: int = 5) -> np.ndarray:
    feats = []
    for seq in seqs:
        feats.append(np.concatenate([
            aac_vector(seq),
            cksaap_vector(seq, k_max=k_max_cksaap),
            pcp16_vector(seq),
        ], axis=0))
    return np.vstack(feats).astype(np.float32)  # (N, 2436)

# ============== Helper: wrap regressors as classifiers (used by a few models) ==============
class RegAsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, reg):
        self.reg = reg
    def fit(self, X, y):
        self.reg_ = clone(self.reg).fit(X, y.astype(float))
        return self
    def predict_proba(self, X):
        z = self.reg_.predict(X)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return np.vstack([1 - p, p]).T
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ============== Model collection (no RBF-SVC) ==============
def get_models() -> Dict[str, object]:
    models = {}
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs", random_state=42))
    ])
    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])
    models["SVM(LinearSVC+Calibrated)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), method="sigmoid", cv=3))
    ])
    models["DecisionTree"] = DecisionTreeClassifier(max_depth=None, random_state=42)
    models["RandomForest"] = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)
    models["GBDT(sklearn)"] = GradientBoostingClassifier(n_estimators=300, random_state=42)

    # XGBoost (prefer GPU)
    try:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="gpu_hist", predictor="gpu_predictor"
        )
    except Exception:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="hist"
        )

    # LightGBM (prefer GPU)
    try:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500, max_depth=-1, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            n_jobs=-1, device="gpu"
        )
        _ = models["LightGBM"].get_params()
    except Exception:
        try:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                n_jobs=-1, device_type="gpu"
            )
        except Exception:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
            )

    # CatBoost (prefer GPU)
    try:
        models["CatBoost"] = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05,
            loss_function='Logloss', verbose=False, task_type="GPU", devices="0", random_state=42
        )
    except Exception:
        models["CatBoost"] = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05,
            loss_function='Logloss', verbose=False, random_state=42
        )

    # Naive Bayes & MLP
    models["GaussianNB"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ])
    models["MultinomialNB"] = Pipeline([
        ("minmax", MinMaxScaler()),
        ("clf", MultinomialNB())
    ])
    models["BernoulliNB"] = Pipeline([
        ("minmax", MinMaxScaler()),
        ("bin", Binarizer(threshold=0.5)),
        ("clf", BernoulliNB())
    ])
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42))
    ])
    return models

# ============== Unified probabilities, metrics, and threshold ==============
def _to_proba(est, X) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        s = est.decision_function(X).astype(float)
        smin, smax = np.min(s), np.max(s)
        if smax - smin < 1e-12:
            return np.full_like(s, 0.5, dtype=float)
        return (s - smin) / (smax - smin)
    return est.predict(X).astype(float)

def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    best = float(thr[idx]) if np.isfinite(thr[idx]) else 0.5
    return float(np.clip(best, 0.0, 1.0))

def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    yb = (proba >= thr).astype(int)
    return dict(
        ACC=accuracy_score(y_true, yb),
        Precision=precision_score(y_true, yb, zero_division=0),
        Recall=recall_score(y_true, yb),
        F1=f1_score(y_true, yb),
        AUC=roc_auc_score(y_true, proba),
        MCC=matthews_corrcoef(y_true, yb),
        Threshold=float(thr)
    )

# ============== Main ==============
def main():
    # -------- Data paths --------
    DATA_DIR = "data"
    fp_all = os.path.join(DATA_DIR, "antiCP2.txt")
    if not os.path.exists(fp_all):
        alt_all = os.path.join(os.getcwd(), "antiCP2.txt")
        if os.path.exists(alt_all):
            fp_all = alt_all
    if not os.path.exists(fp_all):
        raise FileNotFoundError(f"File not found in ./data or CWD: {fp_all}")

    # -------- Read and split: 1/6 as TEST --------
    all_seq, all_y = read_fasta_pair_lines(fp_all)
    TEST_RATIO = 1.0 / 6.0
    RANDOM_STATE = 42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO, shuffle=True, stratify=all_y, random_state=RANDOM_STATE
    )
    print(f"[Loaded] total={len(all_seq)} | TRAIN={len(tr_seq)} | TEST={len(te_seq)}")

    # -------- Feature extraction (2436 dims)--------
    print("[Feature] Extracting TRAIN ...")
    X_train = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test  = extract_features(te_seq,  k_max_cksaap=5)
    print(f"[Shapes] Train={X_train.shape}, Test={X_test.shape} (expect 2436 dims)")

    # -------- Model set --------
    models = get_models()

    rows_train, rows_test = [], []

    for i, (name, model) in enumerate(models.items(), 1):
        t0 = time.time()
        print(f"\n[{i}/{len(models)}] Fit final model: {name}")

        # 1) Fit final model on the full training data
        m_final = clone(model)
        m_final.fit(X_train, y_train)

        # 2) Select threshold on the training set (apparent)
        p_train = _to_proba(m_final, X_train)
        thr = youden_threshold(y_train, p_train)

        # 3) Evaluate: TRAIN (apparent) and TEST (holdout)
        m_tr = metrics_from_proba(y_train, p_train, thr)
        p_test = _to_proba(m_final, X_test)
        m_te = metrics_from_proba(y_test, p_test, thr)

        rows_train.append({"Model": name, **m_tr})
        rows_test.append({"Model": name, **m_te})

        print(f"  -> Done in {time.time()-t0:.1f}s | Train AUC={m_tr['AUC']:.4f} | Test AUC={m_te['AUC']:.4f} | thr={thr:.3f}")

        del m_final
        gc.collect()

    # -------- Aggregate & export --------
    df_train = pd.DataFrame(rows_train).sort_values("AUC", ascending=False)
    df_test  = pd.DataFrame(rows_test).sort_values("AUC", ascending=False)

    os.makedirs("baseline_holdout_outputs", exist_ok=True)
    out_xlsx = os.path.join("baseline_holdout_outputs", f"baseline_train_vs_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_train.to_excel(w, sheet_name="Train_Apparent", index=False)
            df_test.to_excel(w,  sheet_name="Test_Holdout",  index=False)
        print(f"\nSaved to: {out_xlsx}")
    except Exception:
        out_csv_train = os.path.join("baseline_holdout_outputs", f"train_apparent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        out_csv_test  = os.path.join("baseline_holdout_outputs", f"test_holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_train.to_csv(out_csv_train, index=False)
        df_test.to_csv(out_csv_test, index=False)
        print(f"\n[Warn] openpyxl not available -> saved CSVs instead:\n  {out_csv_train}\n  {out_csv_test}")

    # -------- Print Top-10 --------
    print("\n=== Train (Apparent) Top-10 by AUC ===")
    print(df_train.head(10))
    print("\n=== Test (Holdout) Top-10 by AUC ===")
    print(df_test.head(10))

if __name__ == "__main__":
    main()
