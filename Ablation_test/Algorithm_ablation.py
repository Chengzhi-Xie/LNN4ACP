# ACP_LNN_algo_ablation_AAC_CKSAAP_PCP16.py
# -*- coding: utf-8 -*-
"""
Algorithm ablation version (model-combination ablation; features fixed as AAC+CKSAAP+PP16 = 2436 dims)
"""

import os, warnings, gc
from datetime import datetime
from itertools import product, combinations
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from ncps.tf import CfC  # Liquid neural network unit (LNN)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# ============== TF GPU (memory growth on demand) ==============
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[TF] Using {len(gpus)} GPU(s) with memory growth.")
    except Exception as e:
        print("[TF] set_memory_growth failed:", e)
else:
    print("[TF] No GPU found, running on CPU.")

# ============== Data loading (two-line FASTA: odd line header, even line sequence) ==============
def read_fasta_pair_lines(file_path: str) -> Tuple[List[str], np.ndarray]:
    sequences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for i in range(0, len(lines), 2):
        header = lines[i]
        seq = lines[i + 1].strip().upper().replace(" ", "")
        label = 1 if "positive" in header.lower() else 0
        sequences.append(seq)
        labels.append(label)
    return sequences, np.array(labels, dtype=np.int32)

# ============== Features: AAC(20) + CKSAAP(k=0..5→2400) + PP16(16) = 2436 dims ==============
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
from itertools import product as _prod
DIPEPTIDES = ["".join(p) for p in _prod(AA20, repeat=2)]  # 400

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
    "polar": set("RNDQEHKTYS"),
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
    s = _check_seq(sequence); L = len(s)
    counts = {aa: 0.0 for aa in AA20}
    for ch in s: counts[ch] += 1.0
    if L > 0:
        for aa in counts: counts[aa] /= L
    return np.array([counts[aa] for aa in AA20], dtype=np.float32)  # 20

def cksaap_vector(sequence: str, k_max: int = 5) -> np.ndarray:
    s = _check_seq(sequence); L = len(s); out = []
    for k in range(k_max + 1):
        counts = dict.fromkeys(DIPEPTIDES, 0.0)
        if L >= k + 2:
            for i in range(L - 1 - k):
                pair = s[i] + s[i + 1 + k]; counts[pair] += 1.0
            total = sum(counts.values())
            if total > 0:
                for dp in counts: counts[dp] /= total
        out.extend([counts[dp] for dp in DIPEPTIDES])
    return np.array(out, dtype=np.float32)  # 2400

def pcp16_vector(sequence: str) -> np.ndarray:
    s = _check_seq(sequence); L = len(s)
    return np.array([sum(ch in PCP16_MAP[key] for ch in s) / L for key in PCP16_KEYS], dtype=np.float32)  # 16

def extract_features(seqs: List[str], k_max_cksaap: int = 5) -> np.ndarray:
    feats = []
    for seq in seqs:
        feats.append(np.concatenate([
            aac_vector(seq),                        # 20
            cksaap_vector(seq, k_max=k_max_cksaap), # 2400
            pcp16_vector(seq),                      # 16
        ], axis=0))
    X = np.vstack(feats).astype(np.float32)
    assert X.shape[1] == 2436, f"Unexpected feature dim {X.shape[1]} != 2436"
    return X

# ============== Adaptive PCA dimension (avoid n_components out of range) ==============
def choose_pca_dim(X_train_sel: np.ndarray, cap: int = 64) -> int:
    n_samples, n_features = X_train_sel.shape[0], X_train_sel.shape[1]
    return int(max(1, min(cap, n_features, n_samples)))

# ============== Threshold and metrics ==============
def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    yb = (proba >= thr).astype(int)
    return dict(
        ACC=accuracy_score(y_true, yb),
        Precision=precision_score(y_true, yb, zero_division=0),
        Recall=recall_score(y_true, yb),
        F1=f1_score(y_true, yb),
        AUC=roc_auc_score(y_true, proba),
        MCC=matthews_corrcoef(y_true, yb),
        Threshold=float(thr),
    )

def search_best_threshold_multi(y_true: np.ndarray, proba: np.ndarray):
    """Two-stage threshold search (consistent with the original optimized version)"""
    def _score_one_thr(t):
        yb = (proba >= t).astype(int)
        acc = accuracy_score(y_true, yb)
        f1  = f1_score(y_true, yb)
        mcc = matthews_corrcoef(y_true, yb)
        return f1 + 0.3 * mcc + 0.1 * (acc - 0.5)

    coarse = np.linspace(0.1, 0.9, 81)
    best_thr, best_score = 0.5, -1.0
    for t in coarse:
        s = _score_one_thr(t)
        if s > best_score: best_thr, best_score = float(t), s

    fine = np.linspace(max(0.01, best_thr - 0.08), min(0.99, best_thr + 0.08), 161)
    for t in fine:
        s = _score_one_thr(t)
        if s > best_score: best_thr, best_score = float(t), s

    # Statistics
    yb = (proba >= best_thr).astype(int)
    stats = dict(
        ACC=accuracy_score(y_true, yb),
        Precision=precision_score(y_true, yb, zero_division=0),
        Recall=recall_score(y_true, yb),
        F1=f1_score(y_true, yb),
        MCC=matthews_corrcoef(y_true, yb),
        Score=best_score,
    )
    return best_thr, stats

# ============== Stage 1: 4 base models ==============
def get_base_models() -> Dict[str, object]:
    models = {}
    # LightGBM (prefer GPU, fallback to CPU if fails)
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

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=-1, tree_method="hist"
    )

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)
    return models

def _predict_proba_safely(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return s

def get_oof_and_holdout_with_folds(X_tr, y_tr, X_ho, estimator, folds):
    """Unified 5-fold OOF/Test; XGB tries GPU first, fallback to CPU on failure"""
    oof = np.zeros(X_tr.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_ho.shape[0], len(folds)), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
        ytr, yva = y_tr[tr_idx], y_tr[va_idx]

        base_params = estimator.get_params()
        clf = estimator.__class__(**base_params)
        if isinstance(clf, XGBClassifier):
            gpu_params = clf.get_params()
            gpu_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
            try:
                clf = XGBClassifier(**gpu_params)
                clf.fit(Xtr, ytr)
                print(f"[XGB][Fold {k+1}] Using GPU (gpu_hist).")
            except (XGBoostError, Exception) as e:
                print(f"[XGB][Fold {k+1}] GPU not available, fallback to CPU hist. Error: {e}")
                cpu_params = clf.get_params()
                cpu_params.pop("tree_method", None)
                cpu_params.pop("predictor", None)
                cpu_params.update({"tree_method": "hist"})
                clf = XGBClassifier(**cpu_params)
                clf.fit(Xtr, ytr)
        else:
            clf.fit(Xtr, ytr)

        oof[va_idx] = _predict_proba_safely(clf, Xva)
        hold_mat[:, k] = _predict_proba_safely(clf, X_ho)

    return oof, hold_mat.mean(axis=1).astype(np.float32)

def compute_meta_predictions_all4(X_train, y_train, X_test, folds, base_models, base_names):
    OOF_list, TEST_list = [], []
    for name in base_names:
        print(f"[Base] {name}: 5-fold OOF/Test ...")
        est = base_models[name]
        oof, te = get_oof_and_holdout_with_folds(X_train, y_train, X_test, est, folds)
        OOF_list.append(oof); TEST_list.append(te)
    meta_train = np.vstack(OOF_list).T.astype(np.float32)   # (N_train, 4)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)  # (N_test, 4)
    return meta_train, meta_test

# ============== Stage 2: LNN (CfC) meta-learner (same structure as original; accepts any input dim) ==============
def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(y.astype(int))
    total = float(len(y))
    return {cls: total / (2.0 * count) for cls, count in enumerate(counts) if count > 0}

def build_lnn_meta(input_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,), name="meta_plus_pca_input")
    x = layers.Dense(128, activation="relu", name="pre_dense_128")(inp)
    x = layers.Dropout(0.3, name="pre_dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="pre_dense_64")(x)
    x = layers.Dropout(0.3, name="pre_dropout_2")(x)

    x = layers.Reshape((1, 64), name="reshape_time")(x)
    x = CfC(64, mixed_memory=True, backbone_units=32, backbone_layers=1,
            backbone_dropout=0.1, return_sequences=True, name="cfc1")(x)
    x = CfC(64, mixed_memory=True, backbone_units=32, backbone_layers=1,
            backbone_dropout=0.1, return_sequences=False, name="cfc2")(x)

    x = layers.Dense(64, activation="relu", name="rep_dense")(x)
    x = layers.Dropout(0.4, name="rep_dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inp, out, name="ACP_LNN_Meta_Fused")
    opt = optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_lnn_kfold_meta(
    X_full: np.ndarray, y: np.ndarray,
    n_splits: int = 5, n_repeats: int = 2, random_state: int = 42,
    epochs: int = 100, batch_size: int = 32,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    models_list = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
        print(f"\n[LNN Meta] Fold {fold+1}/{n_splits}")
        X_tr, X_va = X_full[tr_idx], X_full[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        class_weight = get_class_weights(y_tr)
        preds_va_all = []
        for rep in range(n_repeats):
            print(f"[LNN Meta] Fold {fold+1} Repeat {rep+1}/{n_repeats}")
            tf.keras.utils.set_random_seed(random_state + fold * 100 + rep)
            model = build_lnn_meta(X_full.shape[1])
            es = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1)
            model.fit(
                X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=1,
                validation_data=(X_va, y_va), callbacks=[es, rl], class_weight=class_weight
            )
            preds_va = model.predict(X_va, verbose=0).ravel()
            preds_va_all.append(preds_va)
            models_list.append(model)
        oof_pred[va_idx] = np.mean(np.vstack(preds_va_all), axis=0)
    return models_list, oof_pred

def predict_ensemble_meta(models_list, X_full: np.ndarray) -> np.ndarray:
    preds = [m.predict(X_full, verbose=0).ravel() for m in models_list]
    return np.mean(np.vstack(preds), axis=0)

# ============== Evaluator: different "algorithm combinations" ==============
def eval_lnn_on_meta_subset(
    meta_train, meta_test, raw_pca_train, raw_pca_test, y_train, y_test,
    subset_idxs, label, RANDOM_STATE=42
):
    scaler_meta = StandardScaler()
    meta_tr_s = scaler_meta.fit_transform(meta_train[:, subset_idxs])
    meta_te_s = scaler_meta.transform(meta_test[:, subset_idxs])

    LNN_TRAIN = np.concatenate([meta_tr_s, raw_pca_train], axis=1)
    LNN_TEST  = np.concatenate([meta_te_s,  raw_pca_test],  axis=1)
    print(f"[LNN-Input] {label}: Train{LNN_TRAIN.shape}, Test{LNN_TEST.shape}")

    lnn_models, oof_train_proba = train_lnn_kfold_meta(
        LNN_TRAIN, y_train, n_splits=5, n_repeats=2, random_state=RANDOM_STATE, epochs=100, batch_size=32
    )
    p_tr = oof_train_proba
    p_te = predict_ensemble_meta(lnn_models, LNN_TEST)

    thr, _ = search_best_threshold_multi(y_train, p_tr)
    m_tr = {"Variant": label, "Set": "Train(OOF)", **metrics_from_proba(y_train, p_tr, thr)}
    m_te = {"Variant": label, "Set": "Test(Holdout)", **metrics_from_proba(y_test,  p_te, thr)}

    del lnn_models; gc.collect()
    return m_tr, m_te, thr

def eval_softvote_on_subset(meta_train, meta_test, y_train, y_test, subset_idxs, label):
    p_tr = meta_train[:, subset_idxs].mean(axis=1)
    p_te = meta_test[:, subset_idxs].mean(axis=1)
    thr, _ = search_best_threshold_multi(y_train, p_tr)
    m_tr = {"Variant": label, "Set": "Train(OOF)", **metrics_from_proba(y_train, p_tr, thr)}
    m_te = {"Variant": label, "Set": "Test(Holdout)", **metrics_from_proba(y_test,  p_te, thr)}
    return m_tr, m_te, thr

def eval_lnn_on_raw_only(raw_pca_train, raw_pca_test, y_train, y_test, label="LNN (raw 2436 → PCA)"):
    LNN_TRAIN = raw_pca_train
    LNN_TEST  = raw_pca_test
    print(f"[LNN-Input] {label}: Train{LNN_TRAIN.shape}, Test{LNN_TEST.shape}")
    lnn_models, oof_train_proba = train_lnn_kfold_meta(
        LNN_TRAIN, y_train, n_splits=5, n_repeats=2, random_state=42, epochs=100, batch_size=32
    )
    p_tr = oof_train_proba
    p_te = predict_ensemble_meta(lnn_models, LNN_TEST)
    thr, _ = search_best_threshold_multi(y_train, p_tr)
    m_tr = {"Variant": label, "Set": "Train(OOF)", **metrics_from_proba(y_train, p_tr, thr)}
    m_te = {"Variant": label, "Set": "Test(Holdout)", **metrics_from_proba(y_test,  p_te, thr)}
    del lnn_models; gc.collect()
    return m_tr, m_te, thr

# ============== Main workflow ==============
def main():
    # ---- Fixed/fallback path: antiCP2.txt ----
    DATA_DIR = "data"
    fp_all = os.path.join(DATA_DIR, "antiCP2.txt")
    if not os.path.exists(fp_all):
        alt_all = os.path.join(os.getcwd(), "antiCP2.txt")
        if os.path.exists(alt_all):
            fp_all = alt_all
    if not os.path.exists(fp_all):
        raise FileNotFoundError("File not found in ./data or CWD: %s" % fp_all)

    # ---- Load data & stratified split (1/6 as TEST) ----
    all_seq, all_y = read_fasta_pair_lines(fp_all)
    TEST_RATIO = 1.0 / 6.0
    RANDOM_STATE = 42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO, shuffle=True, stratify=all_y, random_state=RANDOM_STATE
    )
    print(f"[Loaded] total={len(all_seq)} | TRAIN={len(tr_seq)} | TEST={len(te_seq)}")

    # ---- Extract features (2436 dims; fixed AAC+CKSAAP+PP16) ----
    print("[Feature] Extracting TRAIN ...")
    X_train_full = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test_full  = extract_features(te_seq,  k_max_cksaap=5)
    print(f"[Shapes] Train={X_train_full.shape}, Test={X_test_full.shape} (expect 2436 dims)")

    # ---- Raw features -> standardize -> PCA(≤64, adaptive) (one-time, reused across all ablations) ----
    scaler_raw_lnn = StandardScaler().fit(X_train_full)
    X_train_s_lnn  = scaler_raw_lnn.transform(X_train_full)
    X_test_s_lnn   = scaler_raw_lnn.transform(X_test_full)
    pca_dim = choose_pca_dim(X_train_full, cap=64)
    print(f"[PCA] raw features → pca_dim={pca_dim}")
    pca_lnn = PCA(n_components=pca_dim).fit(X_train_s_lnn)
    raw_pca_train = pca_lnn.transform(X_train_s_lnn)
    raw_pca_test  = pca_lnn.transform(X_test_s_lnn)

    # ---- Unified 5 folds; compute OOF/holdout probabilities for the 4 models (one-time) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train_full, y_train))
    base_models = get_base_models()
    base_names  = ["HistGradientBoosting", "LightGBM", "RandomForest", "XGBoost"]
    meta_train, meta_test = compute_meta_predictions_all4(
        X_train_full, y_train, X_test_full, folds, base_models, base_names
    )

    # Abbreviation mapping
    abbrev = {
        "HistGradientBoosting": "HGB",
        "LightGBM": "LGB",
        "RandomForest": "RF",
        "XGBoost": "XGB",
    }

    rows, thrs = [], []

    # ---- 4ML+LNN (baseline) ----
    idx_all = np.arange(4)
    label = "4ML+LNN (" + "+".join([abbrev[n] for n in base_names]) + ")"
    mtr, mte, thr = eval_lnn_on_meta_subset(meta_train, meta_test, raw_pca_train, raw_pca_test,
                                            y_train, y_test, idx_all, label, RANDOM_STATE=RANDOM_STATE)
    rows.extend([mtr, mte]); thrs.append({"Variant": label, "Threshold_from_Train": thr})

    # ---- 4ML soft-vote (no LNN training) ----
    label = "4ML (soft-vote: " + "+".join([abbrev[n] for n in base_names]) + ")"
    mtr, mte, thr = eval_softvote_on_subset(meta_train, meta_test, y_train, y_test, idx_all, label)
    rows.extend([mtr, mte]); thrs.append({"Variant": label, "Threshold_from_Train": thr})

    # ---- LNN (raw-feature PCA only; no base meta) ----
    label = "LNN (raw 2436 → PCA)"
    mtr, mte, thr = eval_lnn_on_raw_only(raw_pca_train, raw_pca_test, y_train, y_test, label=label)
    rows.extend([mtr, mte]); thrs.append({"Variant": label, "Threshold_from_Train": thr})

    # ---- 3ML+LNN (choose 3 of 4, total 4) ----
    for comb in combinations(range(4), 3):
        label = "3ML+LNN (" + "+".join([abbrev[base_names[i]] for i in comb]) + ")"
        mtr, mte, thr = eval_lnn_on_meta_subset(meta_train, meta_test, raw_pca_train, raw_pca_test,
                                                y_train, y_test, np.array(comb), label, RANDOM_STATE=RANDOM_STATE)
        rows.extend([mtr, mte]); thrs.append({"Variant": label, "Threshold_from_Train": thr})

    # ---- 2ML+LNN (choose 2 of 4, total 6) ----
    for comb in combinations(range(4), 2):
        label = "2ML+LNN (" + "+".join([abbrev[base_names[i]] for i in comb]) + ")"
        mtr, mte, thr = eval_lnn_on_meta_subset(meta_train, meta_test, raw_pca_train, raw_pca_test,
                                                y_train, y_test, np.array(comb), label, RANDOM_STATE=RANDOM_STATE)
        rows.extend([mtr, mte]); thrs.append({"Variant": label, "Threshold_from_Train": thr})

    # ---- 1ML+LNN (choose 1 of 4, total 4) ----
    for i in range(4):
        label = "1ML+LNN (" + abbrev[base_names[i]] + ")"
        mtr, mte, thr = eval_lnn_on_meta_subset(meta_train, meta_test, raw_pca_train, raw_pca_test,
                                                y_train, y_test, np.array([i]), label, RANDOM_STATE=RANDOM_STATE)
        rows.extend([mtr, mte]); thrs.append({"Variant": label, "Threshold_from_Train": thr})

    # ---- Aggregate, rank, and save ----
    df_all = pd.DataFrame(rows)
    df_thr = pd.DataFrame(thrs)
    df_rank = (df_all[df_all["Set"].str.startswith("Test")]
               .sort_values(by="AUC", ascending=False)
               .reset_index(drop=True))

    print("\n=== Algorithm Ablation Summary (sorted by Test AUC) ===")
    print(df_rank[["Variant", "AUC", "ACC", "Precision", "Recall", "F1", "MCC", "Threshold"]])

    os.makedirs("algo_ablation_outputs", exist_ok=True)
    out_xlsx = os.path.join(
        "algo_ablation_outputs",
        f"LNN_algo_ablation_AAC_CKSAAP_PP16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_all.to_excel(w, sheet_name="AllMetrics", index=False)
            df_thr.to_excel(w, sheet_name="Thresholds", index=False)
            df_rank.to_excel(w, sheet_name="TestAUC_Ranking", index=False)
            # Keep the base-4 model meta OOF/Test (for reproducibility/plotting)
            base_cols = ["HGB", "LGB", "RF", "XGB"]
            pd.DataFrame(meta_train, columns=base_cols).to_excel(w, sheet_name="MetaTrain_OOF_all4", index=False)
            pd.DataFrame(meta_test,  columns=base_cols).to_excel(w, sheet_name="MetaTest_Preds_all4", index=False)
        print(f"\nSaved to: {out_xlsx}")
    except Exception as e:
        print("[Warn] openpyxl not available, writing CSVs.", e)
        df_all.to_csv(out_xlsx.replace(".xlsx", "_AllMetrics.csv"), index=False)
        df_thr.to_csv(out_xlsx.replace(".xlsx", "_Thresholds.csv"), index=False)
        df_rank.to_csv(out_xlsx.replace(".xlsx", "_TestAUC_Ranking.csv"), index=False)

    print("\n=== Done: LNN (CfC) — Algorithm ablation on fixed AAC+CKSAAP+PP16 ===")

if __name__ == "__main__":
    np.random.seed(42)
    main()
