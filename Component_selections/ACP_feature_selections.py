# -*- coding: utf-8 -*-
"""
ACP_RF_feature_combo_benchmark.py  (single combined dataset, ROC-based threshold)
"""

import os
from itertools import product, combinations
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, roc_curve,
)

# =============================
# Amino-acid space & constants
# =============================
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
DIPEPTIDES = [''.join(p) for p in product(AA20, repeat=2)]   # 400
TRIPEPTIDES = [''.join(p) for p in product(AA20, repeat=3)]  # 8000

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

KYTE_DOOLITTLE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# ============
# IO helpers
# ============
def read_fasta_pair_lines(file_path: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Odd lines are headers (e.g., >ACP_positive_x / >ACP_negative_x); the next even lines are sequences.
    Returns: list of sequences, numpy array of labels (1/0), list of raw headers.
    """
    sequences, labels, headers = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for i in range(0, len(lines), 2):
        header = lines[i]
        seq = lines[i + 1].strip().upper().replace(" ", "")
        label = 1 if "positive" in header.lower() else 0
        sequences.append(seq)
        labels.append(label)
        headers.append(header)
    return sequences, np.array(labels, dtype=np.int32), headers

# ====================
# Sequence validation
# ====================
def _check_seq(seq: str) -> str:
    s = seq.strip().upper().replace(" ", "")
    if not s or any(ch not in AA_SET for ch in s):
        raise ValueError(f"Invalid residues in sequence: {seq}")
    return s

# =================
# Feature builders
# =================
def vec_aac(sequence: str) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    counts = {aa: 0.0 for aa in AA20}
    for ch in s:
        counts[ch] += 1.0
    if L > 0:
        for aa in counts:
            counts[aa] /= L
    return np.array([counts[aa] for aa in AA20], dtype=np.float32)  # 20

def vec_dpc(sequence: str, normalized: bool = True) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    counts = dict.fromkeys(DIPEPTIDES, 0.0)
    if L >= 2:
        for i in range(L - 1):
            dp = s[i:i+2]
            counts[dp] += 1.0
        if normalized:
            denom = float(L - 1)
            for dp in counts:
                counts[dp] /= denom
    return np.array([counts[dp] for dp in DIPEPTIDES], dtype=np.float32)  # 400

def vec_cksaap(sequence: str, k_max: int = 5) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    out = []
    for k in range(k_max + 1):
        counts = dict.fromkeys(DIPEPTIDES, 0.0)
        if L >= k + 2:
            for i in range(L - 1 - k):
                pair = s[i] + s[i + 1 + k]
                counts[pair] += 1.0
            total = float(sum(counts.values()))
            if total > 0:
                for dp in counts:
                    counts[dp] /= total
        out.extend([counts[dp] for dp in DIPEPTIDES])
    return np.array(out, dtype=np.float32)  # (k_max+1)*400

def vec_pcp16(sequence: str) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    return np.array([
        sum(ch in PCP16_MAP[key] for ch in s) / L for key in PCP16_KEYS
    ], dtype=np.float32)  # 16

def vec_hydro_mb(sequence: str, max_lag: int = 5, center: bool = True) -> np.ndarray:
    s = _check_seq(sequence)
    vals = np.array([KYTE_DOOLITTLE[ch] for ch in s], dtype=np.float32)
    if center and vals.size > 0:
        vals = vals - float(vals.mean())  # mean-centering for MB autocorr
    L = len(vals)
    ac = []
    for d in range(1, max_lag + 1):
        if L - d <= 0:
            ac.append(0.0)
        else:
            ssum = 0.0
            for i in range(L - d):
                ssum += float(vals[i] * vals[i + d])
            ac.append(ssum / (L - d))
    return np.array(ac, dtype=np.float32)  # max_lag

def vec_atc(sequence: str, normalized: bool = True) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    counts = dict.fromkeys(TRIPEPTIDES, 0.0)
    if L >= 3:
        for i in range(L - 2):
            tri = s[i:i+3]
            counts[tri] += 1.0
        if normalized:
            denom = float(L - 2)
            for tri in counts:
                counts[tri] /= denom
    return np.array([counts[tri] for tri in TRIPEPTIDES], dtype=np.float32)  # 8000

# Feature registry (name -> builder function)
FEATURE_REGISTRY = {
    "AAC":       lambda seq: vec_aac(seq),
    "DPC":       lambda seq: vec_dpc(seq, normalized=True),
    "CKSAAP_k":  lambda seq: vec_cksaap(seq, k_max=5),                 # placeholder; parameter set later
    "PCP16":     lambda seq: vec_pcp16(seq),
    "HydroAC":   lambda seq: vec_hydro_mb(seq, max_lag=5, center=True),# placeholder; parameter set later
    "ATC":       lambda seq: vec_atc(seq, normalized=True),
}

# ============================
# Metrics & threshold utils
# ============================
def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return float('nan')

def select_threshold_by_youden(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Threshold selection based on the ROC curve: maximize Youden's J = TPR - FPR.
    Note: AUC is threshold-independent; here we use ROC geometry to pick a threshold consistent with AUC semantics.
    """
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    best = float(thr[idx]) if np.isfinite(thr[idx]) else 0.5
    return float(np.clip(best, 0.0, 1.0))

def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    yb = (proba >= thr).astype(np.int32)
    out = dict(
        ACC=accuracy_score(y_true, yb),
        Precision=precision_score(y_true, yb, zero_division=0),
        Recall=recall_score(y_true, yb),
        F1=f1_score(y_true, yb),
        AUC=safe_auc(y_true, proba),
        MCC=matthews_corrcoef(y_true, yb),
        Threshold=float(thr),
    )
    return out

# ==============
# CV helpers
# ==============
def oof_and_holdout_probs(
    X_train: np.ndarray, y_train: np.ndarray, X_hold: np.ndarray,
    rf_params: Dict, folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train RF with the given fold splits to obtain OOF probabilities and holdout probabilities
    (the holdout probabilities are averaged across fold models).
    """
    oof = np.zeros(X_train.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_hold.shape[0], len(folds)), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(folds):
        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]
        hold_mat[:, k] = clf.predict_proba(X_hold)[:, 1]
    return oof, hold_mat.mean(axis=1).astype(np.float32)

# ============================
# Feature cache utilities
# ============================
def compute_feature_matrix(seqs: List[str], feat_name: str) -> np.ndarray:
    fn = FEATURE_REGISTRY[feat_name]
    mats = [fn(s) for s in seqs]
    return np.vstack(mats).astype(np.float32)

def build_cached_features(seqs_dict: Dict[str, List[str]], features: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = {split: {} for split in seqs_dict}
    for split, seqs in seqs_dict.items():
        for feat in features:
            cache[split][feat] = compute_feature_matrix(seqs, feat)
            print(f"[Cache] {split} - {feat}: shape={cache[split][feat].shape}")
    return cache

def hstack_combo(cache: Dict[str, Dict[str, np.ndarray]], split: str, combo: Tuple[str, ...]) -> np.ndarray:
    mats = [cache[split][f] for f in combo]
    return np.hstack(mats).astype(np.float32)

# ============================
# Duplicate sequence report
# ============================
def report_duplicates(train_seqs: List[str], test_seqs: List[str]) -> None:
    st_tr, st_te = set(train_seqs), set(test_seqs)
    dup_te = st_tr & st_te
    print("\n[Duplicate check]")
    print(f"Train unique={len(st_tr)} | Test unique={len(st_te)} | Train∩Test={len(dup_te)}")
    if dup_te:
        print(f"  Examples Train∩Test (up to 5): {list(sorted(dup_te))[:5]}")

# =========
# main
# =========
def main():
    # Fixed paths
    DATA_DIR = "data"
    fp_all = os.path.join(DATA_DIR, "antiCP2.txt")

    # Fallback to current working directory (CWD)
    if not os.path.exists(fp_all):
        alt_all = os.path.join(os.getcwd(), "antiCP2.txt")
        if os.path.exists(alt_all):
            fp_all = alt_all

    if not os.path.exists(fp_all):
        raise FileNotFoundError(f"File not found in ./data or CWD: {fp_all}")

    # Read merged dataset
    all_seq, all_y, _ = read_fasta_pair_lines(fp_all)
    n_all = len(all_seq)
    print(f"[Loaded] antiCP2 total n={n_all}")

    # Randomly take 1/6 as TEST and the rest as TRAIN (stratified to preserve class ratio)
    TEST_RATIO = 1.0 / 6.0
    RANDOM_STATE = 42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO, shuffle=True,
        stratify=all_y, random_state=RANDOM_STATE
    )
    print(f"[Split] TRAIN n={len(tr_seq)} | TEST n={len(te_seq)} (test_ratio={TEST_RATIO:.3f})")
    report_duplicates(tr_seq, te_seq)

    # Tunables (if memory/speed is limited, set CKSAAP_K=3 or INCLUDE_ATC=False)
    CKSAAP_K   = 5
    HYDRO_LAG  = 5
    INCLUDE_ATC = True   # If the 8000-dim feature is too heavy, set to False

    N_SPLITS = 5
    RF_PARAMS = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # Bind parameterized features
    FEATURE_REGISTRY["CKSAAP_k"] = lambda seq: vec_cksaap(seq, k_max=CKSAAP_K)
    FEATURE_REGISTRY["HydroAC"]  = lambda seq: vec_hydro_mb(seq, max_lag=HYDRO_LAG, center=True)

    # Feature pool
    FEATURE_POOL = ["AAC", "DPC", "CKSAAP_k", "PCP16", "HydroAC"]
    if INCLUDE_ATC:
        FEATURE_POOL.append("ATC")

    # Cache features
    cache = build_cached_features({"TRAIN": tr_seq, "TEST": te_seq}, FEATURE_POOL)

    # Fold split
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(y_train)), y_train))

    # Enumerate all combinations of 1–4 features
    combos: List[Tuple[str, ...]] = []
    for k in range(1, min(4, len(FEATURE_POOL)) + 1):
        combos.extend(list(combinations(FEATURE_POOL, k)))
    print(f"\n[Combos] evaluating {len(combos)} combinations from pool: {FEATURE_POOL}")

    rows: List[Dict] = []

    for combo in combos:
        X_tr = hstack_combo(cache, "TRAIN", combo)
        X_te = hstack_combo(cache, "TEST", combo)

        # Train OOF probabilities + test-set probabilities
        oof, te_prob = oof_and_holdout_probs(X_tr, y_train, X_te, RF_PARAMS, folds)

        # Select threshold on OOF by ROC (Youden's J)
        thr = select_threshold_by_youden(y_train, oof)

        # Evaluation
        m_test = metrics_from_proba(y_test, te_prob, thr)
        oof_auc = safe_auc(y_train, oof)

        row = {
            "Combo": "+".join(combo),
            "Dims": int(X_tr.shape[1]),
            "Thr_OOF": float(thr),
            "OOF_ACC": accuracy_score(y_train, (oof >= thr).astype(np.int32)),
            "OOF_AUC": oof_auc,
            "TEST_ACC": m_test["ACC"],
            "TEST_Precision": m_test["Precision"],
            "TEST_Recall": m_test["Recall"],
            "TEST_F1": m_test["F1"],
            "TEST_AUC": m_test["AUC"],
            "TEST_MCC": m_test["MCC"],
        }
        rows.append(row)
        print(f"[Done] {row['Combo']}: OOF_AUC={row['OOF_AUC']:.4f} | TEST_AUC={row['TEST_AUC']:.4f} (ACC_test={row['TEST_ACC']:.4f}, thr={row['Thr_OOF']:.3f}, dims={row['Dims']})")

    # Result table
    df = pd.DataFrame(rows).sort_values("TEST_AUC", ascending=False)

    # Save
    outdir = "feature_combo_outputs"
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_xlsx = os.path.join(outdir, f"feature_combo_results_{ts}.xlsx")

    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Results", index=False)
        print(f"\nSaved to: {out_xlsx}")
    except Exception:
        out_csv = os.path.join(outdir, f"feature_combo_results_{ts}.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[Warn] openpyxl not available -> saved CSV instead: {out_csv}")

    # Top-10
    cols = ["Combo", "Dims", "Thr_OOF", "OOF_AUC", "TEST_AUC", "TEST_ACC", "TEST_F1", "TEST_MCC"]
    print("\n=== Top-10 by TEST AUC ===")
    print(df[cols].head(10))


if __name__ == "__main__":
    np.random.seed(42)
    main()
