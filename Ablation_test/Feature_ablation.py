# ACP_LNN_Stack_AAC_CKSAAP_PCP16_optim_repeat_thrrefine.py
# -*- coding: utf-8 -*-
"""
LNN (CfC)-based anticancer peptide screening model (AAC+CKSAAP+PCP16)
Optimized + Multi-Repeat LNN Ensemble + Refined Threshold Search
— Feature ablation version (only modifies the "feature-combination inputs"; model architecture and training pipeline remain unchanged)

Seven feature combinations consistent with Characteristic_ablation.py:
  1) AAC
  2) CKSAAP
  3) PP16
  4) AAC+CKSAAP
  5) CKSAAP+PP16
  6) AAC+PP16
  7) AAC+CKSAAP+PP16

Workflow (for each combination):
  - Stage 1: 4 tree models as base learners (HistGBDT / LightGBM / RF / XGBoost)
      Use unified 5-fold CV on training set to produce OOF probabilities + ensembled test probabilities as 4-D meta features
  - Stage 2: LNN (CfC) as the meta-learner (main model)
      Input = [4-D meta features (standardized)] + [selected raw features standardized then PCA(≤64, adaptive)]
  - Threshold: two-stage refined threshold search on Train(OOF) probabilities (search_best_threshold_multi)
  - Output: Train/Test metrics for each combination, and finally write a single Excel:
      AllMetrics / Thresholds / TestAUC_Ranking
"""

import os, warnings, gc
from datetime import datetime
from itertools import product
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost.core import XGBoostError  # Prefer capturing GPU-related errors first

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

# ============== Data loading (two-line FASTA: odd line header/label, even line sequence) ==============
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

# ============== Features: AAC + CKSAAP_k + PCP16 (total 2436 dims) ==============
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
DIPEPTIDES = ["".join(p) for p in product(AA20, repeat=2)]  # 400

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
    return np.array(out, dtype=np.float32)  # (k_max+1)*400 = 2400

def pcp16_vector(sequence: str) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    return np.array(
        [sum(ch in PCP16_MAP[key] for ch in s) / L for key in PCP16_KEYS],
        dtype=np.float32,
    )  # 16

def extract_features(seqs: List[str], k_max_cksaap: int = 5) -> np.ndarray:
    feats = []
    for seq in seqs:
        feats.append(
            np.concatenate(
                [
                    aac_vector(seq),                        # 20
                    cksaap_vector(seq, k_max=k_max_cksaap), # 2400
                    pcp16_vector(seq),                      # 16
                ],
                axis=0,
            )
        )
    X = np.vstack(feats).astype(np.float32)
    assert X.shape[1] == 2436, f"Unexpected feature dim {X.shape[1]} != 2436"
    return X

# ============== Feature sub-slices (consistent with Characteristic_ablation.py) ==============
def feature_slices_2436():
    """AAC[0:20], CKSAAP[20:2420], PP16[2420:2436]"""
    idx_aac = slice(0, 20)
    idx_ck  = slice(20, 2420)
    idx_pp  = slice(2420, 2436)
    return {"AAC": idx_aac, "CKSAAP": idx_ck, "PP16": idx_pp}

def select_features(X: np.ndarray, keys: List[str]) -> np.ndarray:
    sl = feature_slices_2436()
    cols = [X[:, sl[k]] for k in keys]
    return np.concatenate(cols, axis=1)

# ============== Adaptive PCA dimension (fix n_components upper-bound issue) ==============
def choose_pca_dim(X_train_sel: np.ndarray, cap: int = 64) -> int:
    """
    Return a feasible PCA dimension that satisfies:
      n_components <= min(n_samples, n_features)
    Default upper bound cap=64, preserving the original design of "expected cap at 64".
    """
    n_samples, n_features = X_train_sel.shape[0], X_train_sel.shape[1]
    return int(max(1, min(cap, n_features, n_samples)))

# ============== Metrics & threshold search ==============
def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    yb = (proba >= thr).astype(int)
    return dict(
        ACC=accuracy_score(y_true, yb),
        Precision=precision_score(y_true, yb, zero_division=0),
        Recall=recall_score(y_true, yb),
        F1=f1_score(y_true, yb),
        AUC=roc_auc_score(y_true, proba),
        AP=average_precision_score(y_true, proba),
        MCC=matthews_corrcoef(y_true, yb),
        Threshold=float(thr),
    )

def search_best_threshold_multi(y_true: np.ndarray, proba: np.ndarray):
    """
    Two-stage refined threshold search (only on Train(OOF)):
      1) Coarse grid: 0.1~0.9, step 0.01
      2) Fine grid: around coarse-best ±0.08 (step ~0.001)
    Scoring:
      score = F1 + 0.3 * MCC + 0.1 * (ACC - 0.5)
    """
    def _score_one_thr(thr_val: float):
        yb = (proba >= thr_val).astype(int)
        acc = accuracy_score(y_true, yb)
        f1  = f1_score(y_true, yb)
        mcc = matthews_corrcoef(y_true, yb)
        return f1 + 0.3 * mcc + 0.1 * (acc - 0.5)

    coarse_grid = np.linspace(0.1, 0.9, 81)
    best_thr = 0.5
    best_score = -1.0
    for thr in coarse_grid:
        s = _score_one_thr(thr)
        if s > best_score:
            best_score, best_thr = s, float(thr)

    fine_start = max(0.01, best_thr - 0.08)
    fine_end   = min(0.99, best_thr + 0.08)
    fine_grid  = np.linspace(fine_start, fine_end, 161)

    for thr in fine_grid:
        s = _score_one_thr(thr)
        if s > best_score:
            best_score, best_thr = s, float(thr)

    # Stats for record
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

# ============== Curve saving/plotting (optional) ==============
def _save_curve_points_roc(fpr, tpr, thr, save_csv):
    df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr})
    df.to_csv(save_csv, index=False)

def _save_curve_points_prc(precision, recall, thr, save_csv):
    thr_full = np.concatenate([thr, [np.nan]]) if thr is not None else np.full_like(precision, np.nan)
    df = pd.DataFrame({"Recall": recall, "Precision": precision, "Threshold": thr_full})
    df.to_csv(save_csv, index=False)

def plot_roc_prc(y_true, proba, set_name: str, out_dir: str, title_prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ROC
    fpr, tpr, thr = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC (AUC = %.4f)" % auc_val, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold", color="black")
    plt.title(f"{title_prefix} — {set_name} ROC", fontsize=14, fontweight="bold", color="black")
    plt.legend(loc="lower right")
    roc_png = os.path.join(out_dir, f"{ts}_{set_name}_ROC.png")
    plt.tight_layout(); plt.savefig(roc_png, dpi=300); plt.close()
    _save_curve_points_roc(fpr, tpr, thr, os.path.join(out_dir, f"{ts}_{set_name}_ROC_points.csv"))

    # PRC
    precision, recall, thr_pr = precision_recall_curve(y_true, proba)
    ap_val = average_precision_score(y_true, proba)
    plt.figure()
    plt.plot(recall, precision, label="PRC (AP = %.4f)" % ap_val, linewidth=2)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("Recall", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("Precision", fontsize=12, fontweight="bold", color="black")
    plt.title(f"{title_prefix} — {set_name} PRC", fontsize=14, fontweight="bold", color="black")
    plt.legend(loc="lower left")
    prc_png = os.path.join(out_dir, f"{ts}_{set_name}_PRC.png")
    plt.tight_layout(); plt.savefig(prc_png, dpi=300); plt.close()
    _save_curve_points_prc(precision, recall, thr_pr, os.path.join(out_dir, f"{ts}_{set_name}_PRC_points.csv"))
    print("[Plot]", set_name, "ROC→", roc_png, "| PRC→", prc_png)

# ============== Confusion-matrix plotting (kept; not mandatory to call) ==============
def _single_color_cmap(hex_color: str):
    return LinearSegmentedColormap.from_list("single_color", ["#FFFFFF", hex_color], N=256)

def plot_confmat(cm: np.ndarray, title: str, out_png: str, main_color: str = "#d32f2f"):
    cmap = _single_color_cmap(main_color)
    plt.figure(figsize=(5.2, 4.6))
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=14, fontweight="bold", color="black")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = ["Negative", "Positive"]
    plt.xticks(np.arange(2), ticks, fontsize=11, fontweight="bold", color="black")
    plt.yticks(np.arange(2), ticks, fontsize=11, fontweight="bold", color="black")
    plt.ylabel("True label", fontsize=12, fontweight="bold", color="black")
    plt.xlabel("Predicted label", fontsize=12, fontweight="bold", color="black")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            plt.text(
                j, i, "%d" % val,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=12, fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()
    print("[CM] Saved:", out_png)

# ============== Stage 1: 4 base models (structure unchanged) ==============
def get_base_models() -> Dict[str, object]:
    models = {}

    # LightGBM (prefer GPU, fallback to CPU if fails)
    try:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            device="gpu",
        )
        _ = models["LightGBM"].get_params()
    except Exception:
        try:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                device_type="gpu",
            )
        except Exception:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
            )

    # RandomForest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )

    # XGBoost: build with CPU hist by default; try GPU (gpu_hist) during training with auto fallback
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    # HistGradientBoosting
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)

    return models

def get_oof_and_holdout_with_folds(X_tr, y_tr, X_ho, estimator, folds):
    """
    Returns:
      oof:  (N_train,)   — OOF probabilities for this model
      hold: (N_holdout,) — predict holdout with each fold model, then average across folds

    For XGBoost: try GPU (gpu_hist) first; fallback to CPU (hist) on failure
    """
    from sklearn.base import clone

    oof = np.zeros(X_tr.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_ho.shape[0], len(folds)), dtype=np.float32)

    for k, (tr_idx, va_idx) in enumerate(folds):
        base_params = estimator.get_params()
        clf = estimator.__class__(**base_params)

        X_tr_fold, y_tr_fold = X_tr[tr_idx], y_tr[tr_idx]
        X_va_fold = X_tr[va_idx]

        # XGBoost: try GPU, fallback to CPU
        if isinstance(clf, XGBClassifier):
            gpu_params = clf.get_params()
            gpu_params.update({
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
            })
            try:
                clf = XGBClassifier(**gpu_params)
                clf.fit(X_tr_fold, y_tr_fold)
                print("[XGB][Fold %d] Using GPU (gpu_hist)." % (k + 1))
            except (XGBoostError, Exception) as e:
                print("[XGB][Fold %d] GPU not available, fallback to CPU hist. Error: %s" % (k + 1, e))
                cpu_params = clf.get_params()
                cpu_params.pop("tree_method", None)
                cpu_params.pop("predictor", None)
                cpu_params.update({"tree_method": "hist"})
                clf = XGBClassifier(**cpu_params)
                clf.fit(X_tr_fold, y_tr_fold)
        else:
            clf = clone(estimator)
            clf.fit(X_tr_fold, y_tr_fold)

        if hasattr(clf, "predict_proba"):
            oof[va_idx] = clf.predict_proba(X_va_fold)[:, 1]
            hold_mat[:, k] = clf.predict_proba(X_ho)[:, 1]
        else:
            s_va = clf.decision_function(X_va_fold)
            s_ho = clf.decision_function(X_ho)
            s_va = (s_va - s_va.min()) / (s_va.max() - s_va.min() + 1e-12)
            s_ho = (s_ho - s_ho.min()) / (s_ho.max() - s_ho.min() + 1e-12)
            oof[va_idx] = s_va
            hold_mat[:, k] = s_ho

    return oof, hold_mat.mean(axis=1).astype(np.float32)

# ============== Stage 2: LNN (CfC) meta-learner (structure unchanged) ==============
def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(y.astype(int))
    total = float(len(y))
    return {cls: total / (2.0 * count) for cls, count in enumerate(counts) if count > 0}

def build_lnn_meta(input_dim: int) -> tf.keras.Model:
    """
    LNN (CfC) as the meta-learner, fusing:
      - 4-D base-model meta features (already standardized)
      - 64-D (or smaller) PCA raw features (already standardized)
    Total input dimension = input_dim (usually = 4 + pca_dim), then compressed to 64 before CfC; architecture unchanged.
    """
    inp = layers.Input(shape=(input_dim,), name="meta_plus_pca_input")
    x = layers.Dense(128, activation="relu", name="pre_dense_128")(inp)
    x = layers.Dropout(0.3, name="pre_dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="pre_dense_64")(x)
    x = layers.Dropout(0.3, name="pre_dropout_2")(x)

    x = layers.Reshape((1, 64), name="reshape_time")(x)

    x = CfC(
        64,
        mixed_memory=True,
        backbone_units=32,
        backbone_layers=1,
        backbone_dropout=0.1,
        return_sequences=True,
        name="cfc1",
    )(x)
    x = CfC(
        64,
        mixed_memory=True,
        backbone_units=32,
        backbone_layers=1,
        backbone_dropout=0.1,
        return_sequences=False,
        name="cfc2",
    )(x)

    x = layers.Dense(64, activation="relu", name="rep_dense")(x)
    x = layers.Dropout(0.4, name="rep_dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inp, out, name="ACP_LNN_Meta_Fused")
    opt = optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_lnn_kfold_meta(
    X_full: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
):
    """
    Train LNN on fused features (meta + PCA(raw)) with 5-fold * n_repeats, returning:
      - models_list: total n_splits * n_repeats LNN models (for test ensemble)
      - oof_pred: OOF probabilities on training set (per fold = mean across repeats)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    models_list = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
        print("\n[LNN Meta] Fold %d/%d" % (fold + 1, n_splits))
        X_tr, X_va = X_full[tr_idx], X_full[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        class_weight = get_class_weights(y_tr)
        print("[ClassWeight]", class_weight)

        preds_va_all = []

        for rep in range(n_repeats):
            print("[LNN Meta] Fold %d Repeat %d/%d" % (fold + 1, rep + 1, n_repeats))

            tf.keras.utils.set_random_seed(random_state + fold * 100 + rep)
            model = build_lnn_meta(X_full.shape[1])

            es = callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
            rl = callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
            )

            model.fit(
                X_tr,
                y_tr,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                validation_data=(X_va, y_va),
                callbacks=[es, rl],
                class_weight=class_weight,
            )

            preds_va = model.predict(X_va, verbose=0).ravel()
            preds_va_all.append(preds_va)
            models_list.append(model)

        preds_va_mean = np.mean(np.vstack(preds_va_all), axis=0)
        oof_pred[va_idx] = preds_va_mean

    return models_list, oof_pred

def predict_ensemble_meta(models_list, X_full: np.ndarray) -> np.ndarray:
    preds = [m.predict(X_full, verbose=0).ravel() for m in models_list]
    return np.mean(np.vstack(preds), axis=0)

# ============== End-to-end pipeline for one feature combination (structure unchanged; only swap X_sel) ==============
def run_one_combo(
    combo_name: str,
    keys: List[str],
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    folds_base,
    base_names: List[str],
    base_models: Dict[str, object],
    RANDOM_STATE: int = 42,
    do_plot: bool = False,
    out_root: str = "lnn_stack_ablation_outputs",
):
    print(f"\n========== [Combo] {combo_name} ==========")
    # Select raw features for this combination
    X_train_sel = select_features(X_train_full, keys)
    X_test_sel  = select_features(X_test_full,  keys)
    print(f"[Feature-Select] {combo_name}: Train{X_train_sel.shape}, Test{X_test_sel.shape}")

    # ---- Stage 1: 4 base models OOF / holdout probabilities ----
    OOF_list, TEST_list = [], []
    for name in base_names:
        est = base_models[name]
        print(f"[Base] {name}: unified 5-fold OOF & holdout ...")
        oof, te = get_oof_and_holdout_with_folds(X_train_sel, y_train, X_test_sel, est, folds_base)
        OOF_list.append(oof)
        TEST_list.append(te)
    meta_train = np.vstack(OOF_list).T.astype(np.float32)   # (N_train, 4)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)  # (N_test, 4)

    # ---- Standardize meta features ----
    scaler_meta = StandardScaler()
    META_TRAIN_S = scaler_meta.fit_transform(meta_train)
    META_TEST_S  = scaler_meta.transform(meta_test)

    # ---- LNN: selected raw features -> standardize -> PCA(≤64, adaptive) ----
    scaler_raw_lnn = StandardScaler().fit(X_train_sel)
    X_train_s_lnn  = scaler_raw_lnn.transform(X_train_sel)
    X_test_s_lnn   = scaler_raw_lnn.transform(X_test_sel)

    pca_dim = choose_pca_dim(X_train_sel, cap=64)
    print(f"[PCA] {combo_name}: n_features={X_train_sel.shape[1]}, "
          f"n_samples={X_train_sel.shape[0]}, pca_dim={pca_dim}")

    pca_lnn = PCA(n_components=pca_dim).fit(X_train_s_lnn)
    X_train_pca = pca_lnn.transform(X_train_s_lnn)
    X_test_pca  = pca_lnn.transform(X_test_s_lnn)

    # ---- Fuse meta + PCA(raw[sel]) as LNN input ----
    LNN_TRAIN = np.concatenate([META_TRAIN_S, X_train_pca], axis=1)
    LNN_TEST  = np.concatenate([META_TEST_S,  X_test_pca],  axis=1)
    print(f"[LNN-Input] {combo_name}: LNN_TRAIN{LNN_TRAIN.shape}, LNN_TEST{LNN_TEST.shape}")

    # ---- LNN (CfC) 5-fold * n_repeats ensemble (structure unchanged) ----
    lnn_models, oof_train_proba = train_lnn_kfold_meta(
        LNN_TRAIN, y_train,
        n_splits=5, n_repeats=2, random_state=RANDOM_STATE,
        epochs=100, batch_size=32,
    )
    p_tr = oof_train_proba
    p_te = predict_ensemble_meta(lnn_models, LNN_TEST)

    # ---- Two-stage threshold search on training OOF probabilities ----
    thr_best, thr_stats_train = search_best_threshold_multi(y_train, p_tr)

    # ---- Metrics ----
    m_train = {"Combo": combo_name, "Set": "Train(OOF)", **metrics_from_proba(y_train, p_tr, thr_best)}
    m_test  = {"Combo": combo_name, "Set": "Test(Holdout)", **metrics_from_proba(y_test,  p_te, thr_best)}

    # ---- Optional plotting (kept consistent with original structure; off by default to save time/space) ----
    if do_plot:
        out_dir = os.path.join(out_root, f"{combo_name.replace('+','_')}")
        os.makedirs(out_dir, exist_ok=True)
        plot_roc_prc(y_train, p_tr, "Train", out_dir, title_prefix=f"LNN-Stack ({combo_name})")
        plot_roc_prc(y_test,  p_te, "Test",  out_dir, title_prefix=f"LNN-Stack ({combo_name})")

    # ---- Cleanup ----
    del lnn_models
    gc.collect()

    return (m_train, m_test), meta_train, meta_test, thr_best

# ============== Main: run 7 combinations and summarize outputs ==============
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
        all_seq,
        all_y,
        test_size=TEST_RATIO,
        shuffle=True,
        stratify=all_y,
        random_state=RANDOM_STATE,
    )
    print("[Loaded] total=%d | TRAIN=%d | TEST=%d" % (len(all_seq), len(tr_seq), len(te_seq)))

    # ---- One-time extraction of full 2436-dim features ----
    print("[Feature] Extracting TRAIN ...")
    X_train_full = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test_full  = extract_features(te_seq,  k_max_cksaap=5)
    print("[Shapes] Train=%s, Test=%s (expect 2436 dims)" % (X_train_full.shape, X_test_full.shape))

    # ---- Unified 5 folds (reused by all combinations) ----
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds_base = list(skf_base.split(X_train_full, y_train))

    base_models = get_base_models()
    base_names  = ["HistGradientBoosting", "LightGBM", "RandomForest", "XGBoost"]

    # ---- Seven feature combinations (consistent with Characteristic_ablation.py) ----
    combos = [
        ("AAC", ["AAC"]),
        ("CKSAAP", ["CKSAAP"]),
        ("PP16", ["PP16"]),
        ("AAC+CKSAAP", ["AAC", "CKSAAP"]),
        ("CKSAAP+PP16", ["CKSAAP", "PP16"]),
        ("AAC+PP16", ["AAC", "PP16"]),
        ("AAC+CKSAAP+PP16", ["AAC", "CKSAAP", "PP16"]),
    ]

    # ---- Run each combination and aggregate ----
    rows = []
    thresholds = []
    meta_oofs = {}
    meta_tests = {}

    out_root = "lnn_stack_ablation_outputs"
    os.makedirs(out_root, exist_ok=True)

    for cname, keys in combos:
        (m_tr, m_te), meta_tr, meta_te, thr = run_one_combo(
            combo_name=cname,
            keys=keys,
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            y_train=y_train,
            y_test=y_test,
            folds_base=folds_base,
            base_names=base_names,
            base_models=base_models,
            RANDOM_STATE=RANDOM_STATE,
            do_plot=False,                 # Set True if you want plotting for each combination
            out_root=out_root,
        )
        rows.extend([m_tr, m_te])
        thresholds.append({"Combo": cname, "Best_Threshold_from_Train_OOF": thr})
        meta_oofs[cname]  = meta_tr
        meta_tests[cname] = meta_te

    # ---- Summarize and save ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = os.path.join(out_root, f"lnn_LNN_CfC_ablation_{ts}.xlsx")

    df_all = pd.DataFrame(rows)
    df_thr = pd.DataFrame(thresholds)

    # Ranking (Test set only, by AUC descending)
    df_rank = (df_all[df_all["Set"].str.startswith("Test")]
               .sort_values(by="AUC", ascending=False)
               .reset_index(drop=True))

    print("\n=== LNN-Stack (CfC) Ablation Summary (sorted by Test AUC) ===")
    print(df_rank[["Combo", "AUC", "ACC", "Precision", "Recall", "F1", "AP", "MCC", "Threshold"]])

    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_all.to_excel(w, sheet_name="AllMetrics", index=False)
            df_thr.to_excel(w, sheet_name="Thresholds", index=False)
            df_rank.to_excel(w, sheet_name="TestAUC_Ranking", index=False)
            # Optional: save meta features per combo (4-D base OOF/Test for reproducibility/plotting)
            for cname, _ in combos:
                pd.DataFrame(meta_oofs[cname], columns=base_names).to_excel(
                    w, sheet_name=f"{cname}_MetaTrain_OOF", index=False
                )
                pd.DataFrame(meta_tests[cname], columns=base_names).to_excel(
                    w, sheet_name=f"{cname}_MetaTest_Preds", index=False
                )
        print(f"\nSaved to: {out_xlsx}")
    except Exception as e:
        print("[Warn] openpyxl not available, writing CSVs.", e)
        df_all.to_csv(out_xlsx.replace(".xlsx", "_AllMetrics.csv"), index=False)
        df_thr.to_csv(out_xlsx.replace(".xlsx", "_Thresholds.csv"), index=False)
        df_rank.to_csv(out_xlsx.replace(".xlsx", "_TestAUC_Ranking.csv"), index=False)

    print("\n=== Done: LNN (CfC) ACP Stack — Feature-combo ablation (structure unchanged) ===")

if __name__ == "__main__":
    np.random.seed(42)
    main()
