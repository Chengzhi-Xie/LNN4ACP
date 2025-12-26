# ACP_LNN_Stack_AAC_CKSAAP_PCP16_optim_repeat_thrrefine.py
# -*- coding: utf-8 -*-

import os, warnings, gc
from datetime import datetime
from itertools import product
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost.core import XGBoostError  # Catch invalid/unavailable GPU errors

from ncps.tf import CfC  # Liquid neural network unit (LNN)

# ===== SHAP dependency =====
try:
    import shap
except Exception as e:
    raise ImportError(
        "SHAP not available. Please install:\n"
        "  pip install shap\n"
        f"Original error: {e}"
    )

# ===== Composite image dependency (optional) =====
try:
    from PIL import Image
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# UMAP dependency (raise a clear error if not installed)
try:
    import umap
except Exception as e:
    raise ImportError(
        "UMAP not available. Please install:\n"
        "  pip install umap-learn\n"
        f"Original error: {e}"
    )

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

# ============== Features: AAC + CKSAAP_k + PCP16 (2436 dims in total) ==============
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

# ===== Build feature names (order strictly aligned with the concatenation order) =====
def build_feature_names(k_max_cksaap: int = 5) -> List[str]:
    names = []
    for aa in AA20:
        names.append(f"AAC_{aa}")
    for k in range(k_max_cksaap + 1):
        for dp in DIPEPTIDES:
            names.append(f"CKSAAP_{dp}_gap{k}")
    for key in PCP16_KEYS:
        names.append(f"PCP16_{key}")
    assert len(names) == 2436, f"Feature names length mismatch: {len(names)}"
    return names

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
      1) Coarse grid search from 0.1~0.9 with step 0.01
      2) Fine grid search around the best coarse threshold ±0.08 with a smaller step

    The scoring function is identical to the previous version:
      score = F1 + 0.3 * MCC + 0.1 * (ACC - 0.5)
    """

    def _score_one_thr(thr_val: float):
        yb = (proba >= thr_val).astype(int)
        acc = accuracy_score(y_true, yb)
        prec = precision_score(y_true, yb, zero_division=0)
        rec = recall_score(y_true, yb)
        f1 = f1_score(y_true, yb)
        mcc = matthews_corrcoef(y_true, yb)
        score = f1 + 0.3 * mcc + 0.1 * (acc - 0.5)
        stats = dict(ACC=acc, Precision=prec, Recall=rec, F1=f1, MCC=mcc, Score=score)
        return score, stats

    coarse_grid = np.linspace(0.1, 0.9, 81)
    best_thr_coarse = 0.5
    best_score_coarse = -1.0
    best_stats_coarse = None

    for thr in coarse_grid:
        score, stats = _score_one_thr(thr)
        if score > best_score_coarse:
            best_score_coarse = score
            best_thr_coarse = float(thr)
            best_stats_coarse = stats

    print("[ThrSearch-Coarse] best_thr=%.4f | ACC=%.4f, P=%.4f, R=%.4f, F1=%.4f, MCC=%.4f, Score=%.4f"
          % (best_thr_coarse,
             best_stats_coarse["ACC"], best_stats_coarse["Precision"],
             best_stats_coarse["Recall"], best_stats_coarse["F1"],
             best_stats_coarse["MCC"], best_stats_coarse["Score"]))

    fine_start = max(0.01, best_thr_coarse - 0.08)
    fine_end   = min(0.99, best_thr_coarse + 0.08)
    fine_grid  = np.linspace(fine_start, fine_end, 161)

    best_thr = best_thr_coarse
    best_score = best_score_coarse
    best_stats = best_stats_coarse

    for thr in fine_grid:
        score, stats = _score_one_thr(thr)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_stats = stats

    print("[ThrSearch-Fine]   best_thr=%.4f | ACC=%.4f, P=%.4f, R=%.4f, F1=%.4f, MCC=%.4f, Score=%.4f"
          % (best_thr,
             best_stats["ACC"], best_stats["Precision"],
             best_stats["Recall"], best_stats["F1"],
             best_stats["MCC"], best_stats["Score"]))

    return best_thr, best_stats

# ============== ROC / PRC plotting utilities ==============
def _save_curve_points_roc(fpr, tpr, thr, save_csv):
    df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr})
    df.to_csv(save_csv, index=False)

def _save_curve_points_prc(precision, recall, thr, save_csv):
    thr_full = np.concatenate([thr, [np.nan]]) if thr is not None else np.full_like(precision, np.nan)
    df = pd.DataFrame({"Recall": recall, "Precision": precision, "Threshold": thr_full})
    df.to_csv(save_csv, index=False)

def plot_roc_prc(
    y_true,
    proba,
    set_name: str,
    out_dir: str,
    title_prefix: str = "LNN-Stack (AAC+CKSAAP+PCP16)",
    title_override=None,
):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fpr, tpr, thr = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC (AUC = %.4f)" % auc_val, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold", color="black")
    roc_title = title_override[0] if title_override is not None else "%s — %s ROC" % (title_prefix, set_name)
    plt.title(roc_title, fontsize=14, fontweight="bold", color="black")
    plt.legend(loc="lower right")
    roc_png = os.path.join(out_dir, "%s_%s_ROC.png" % (ts, set_name))
    plt.tight_layout()
    plt.savefig(roc_png, dpi=300)
    plt.close()
    _save_curve_points_roc(fpr, tpr, thr, os.path.join(out_dir, "%s_%s_ROC_points.csv" % (ts, set_name)))

    precision, recall, thr_pr = precision_recall_curve(y_true, proba)
    ap_val = average_precision_score(y_true, proba)
    plt.figure()
    plt.plot(recall, precision, label="PRC (AP = %.4f)" % ap_val, linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("Recall", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("Precision", fontsize=12, fontweight="bold", color="black")
    prc_title = title_override[1] if title_override is not None else "%s — %s PRC" % (title_prefix, set_name)
    plt.title(prc_title, fontsize=14, fontweight="bold", color="black")
    plt.legend(loc="lower left")
    prc_png = os.path.join(out_dir, "%s_%s_PRC.png" % (ts, set_name))
    plt.tight_layout()
    plt.savefig(prc_png, dpi=300)
    plt.close()
    _save_curve_points_prc(precision, recall, thr_pr, os.path.join(out_dir, "%s_%s_PRC_points.csv" % (ts, set_name)))

    print("[Plot] %s: ROC→%s | PRC→%s" % (set_name, roc_png, prc_png))

# ============== PCA plotting utility ==============
def _scatter_pca(X2d, y, title, save_png):
    plt.figure(figsize=(6, 5))
    y = np.asarray(y).astype(int)
    plt.scatter(X2d[y == 1, 0], X2d[y == 1, 1], label="Positive", s=18)
    plt.scatter(X2d[y == 0, 0], X2d[y == 0, 1], label="Negative", s=18)
    plt.xlabel("PC1", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("PC2", fontsize=12, fontweight="bold", color="black")
    plt.title(title, fontsize=14, fontweight="bold", color="black")
    plt.xticks(fontsize=10, fontweight="bold", color="black")
    plt.yticks(fontsize=10, fontweight="bold", color="black")
    leg = plt.legend(loc="best", frameon=False)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(save_png, dpi=300)
    plt.close()
    print("[PCA] Saved: %s" % save_png)

# ============== UMAP / t-SNE plotting utilities ==============
def _scatter_umap(X2d, y, title, save_png):
    plt.figure(figsize=(6, 5))
    y = np.asarray(y).astype(int)
    plt.scatter(X2d[y == 1, 0], X2d[y == 1, 1], label="Positive", s=18)
    plt.scatter(X2d[y == 0, 0], X2d[y == 0, 1], label="Negative", s=18)
    plt.xlabel("UMAP1", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("UMAP2", fontsize=12, fontweight="bold", color="black")
    plt.title(title, fontsize=14, fontweight="bold", color="black")
    plt.xticks(fontsize=10, fontweight="bold", color="black")
    plt.yticks(fontsize=10, fontweight="bold", color="black")
    leg = plt.legend(loc="best", frameon=False)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(save_png, dpi=300)
    plt.close()
    print("[UMAP] Saved: %s" % save_png)

def _safe_tsne_perplexity(n):
    if n <= 10:
        return max(2, n // 2)
    p = min(30, max(5, (n - 1) // 3))
    return min(p, n - 1)

def _scatter_tsne(X2d, y, title, save_png):
    plt.figure(figsize=(6, 5))
    y = np.asarray(y).astype(int)
    plt.scatter(X2d[y == 1, 0], X2d[y == 1, 1], label="Positive", s=18)
    plt.scatter(X2d[y == 0, 0], X2d[y == 0, 1], label="Negative", s=18)
    plt.xlabel("t-SNE1", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("t-SNE2", fontsize=12, fontweight="bold", color="black")
    plt.title(title, fontsize=14, fontweight="bold", color="black")
    plt.xticks(fontsize=10, fontweight="bold", color="black")
    plt.yticks(fontsize=10, fontweight="bold", color="black")
    leg = plt.legend(loc="best", frameon=False)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(save_png, dpi=300)
    plt.close()
    print("[t-SNE] Saved: %s" % save_png)

# ============== Confusion matrix plotting utility ==============
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
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("[CM] Saved: %s" % out_png)

# ============== Stage 1: 4 base models ==============
def get_base_models() -> Dict[str, object]:
    models = {}

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

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )

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

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)

    return models

def get_oof_and_holdout_with_folds(X_tr, y_tr, X_ho, estimator, folds):
    oof = np.zeros(X_tr.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_ho.shape[0], len(folds)), dtype=np.float32)

    for k, (tr_idx, va_idx) in enumerate(folds):
        base_params = estimator.get_params()
        clf = estimator.__class__(**base_params)

        X_tr_fold, y_tr_fold = X_tr[tr_idx], y_tr[tr_idx]
        X_va_fold = X_tr[va_idx]

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
            except XGBoostError as e:
                print("[XGB][Fold %d] GPU not available, fallback to CPU hist. Error: %s" % (k + 1, e))
                cpu_params = clf.get_params()
                cpu_params.pop("tree_method", None)
                cpu_params.pop("predictor", None)
                cpu_params.update({"tree_method": "hist"})
                clf = XGBClassifier(**cpu_params)
                clf.fit(X_tr_fold, y_tr_fold)
        else:
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

# ============== Stage 2: LNN (CfC) meta-learner ==============
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
                X_tr, y_tr,
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

# ===== Fixed SHAP (key change: TreeExplainer uses raw) =====
def run_shap_analysis(X_train: np.ndarray, y_train: np.ndarray, out_dir: str, ts: str):
    print("\n[SHAP] Fitting LightGBM on TRAIN only for SHAP explanation (no leakage).")

    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,  # Optional: reduce LightGBM overhead warnings
    )
    lgbm.fit(X_train, y_train)

    feat_names = build_feature_names(k_max_cksaap=5)
    X_train_df = pd.DataFrame(X_train, columns=feat_names)

    # ---- Key fix: do not use probability; raw is the most compatible ----
    # The error comes from a conflict between model_output="probability" and tree_path_dependent
    explainer = shap.TreeExplainer(lgbm, model_output="raw")
    shap_values = explainer.shap_values(X_train_df, check_additivity=False)

    # Compatible with different SHAP return formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[-1]  # take the positive class
    else:
        shap_vals = shap_values

    # ===== Save full-feature CSV =====
    mean_shap = shap_vals.mean(axis=0)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    std_shap = shap_vals.std(axis=0)
    abs_std_shap = np.abs(shap_vals).std(axis=0)

    df_shap = pd.DataFrame({
        "feature": feat_names,
        "mean_shap": mean_shap,
        "mean_abs_shap": mean_abs_shap,
        "std_shap": std_shap,
        "abs_std_shap": abs_std_shap
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    df_shap["rank_by_mean_abs"] = np.arange(1, len(df_shap) + 1)

    csv_path = os.path.join(out_dir, "SHAP_all_features.csv")
    df_shap.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[SHAP] Saved full CSV → {csv_path}")

    # ===== Top-20 plots =====
    top_k = 20

    # A: beeswarm
    plt.figure(figsize=(7.2, 6.8))
    shap.summary_plot(
        shap_vals,
        features=X_train_df,
        feature_names=feat_names,
        max_display=top_k,
        show=False
    )
    fig = plt.gcf()
    fig.text(0.01, 0.98, "A", fontsize=18, fontweight="bold", va="top")

    # Try to enhance colorbar annotation as Feature value + High/Low
    try:
        cax = fig.axes[-1]
        cax.set_ylabel("Feature value", fontsize=12, fontweight="bold")
        cax.yaxis.set_label_position("right")
        cax.set_title("High", fontsize=11, fontweight="bold", pad=8)
        cax.text(0.5, -0.06, "Low", transform=cax.transAxes,
                 ha="center", va="top", fontsize=11, fontweight="bold")
    except Exception:
        pass

    beeswarm_png = os.path.join(out_dir, f"{ts}_SHAP_top20_beeswarm.png")
    plt.tight_layout()
    plt.savefig(beeswarm_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {beeswarm_png}")

    # B: bar
    plt.figure(figsize=(6.6, 6.8))
    shap.summary_plot(
        shap_vals,
        features=X_train_df,
        feature_names=feat_names,
        plot_type="bar",
        max_display=top_k,
        show=False
    )
    ax = plt.gca()
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=12, fontweight="bold", color="black")
    plt.gcf().text(0.01, 0.98, "B", fontsize=18, fontweight="bold", va="top")

    bar_png = os.path.join(out_dir, f"{ts}_SHAP_top20_bar.png")
    plt.tight_layout()
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {bar_png}")

    # Combine A/B
    if _PIL_OK:
        try:
            imgA = Image.open(beeswarm_png)
            imgB = Image.open(bar_png)
            h = max(imgA.height, imgB.height)
            canvas = Image.new("RGB", (imgA.width + imgB.width, h), (255, 255, 255))
            canvas.paste(imgA, (0, 0))
            canvas.paste(imgB, (imgA.width, 0))
            combo_path = os.path.join(out_dir, f"{ts}_SHAP_top20_AB_combined.png")
            canvas.save(combo_path, dpi=(300, 300))
            print(f"[SHAP] Saved combined A/B → {combo_path}")
        except Exception as e:
            print("[SHAP] Combine A/B failed (Pillow).", e)
    else:
        print("[SHAP] Pillow not installed, skipped combined A/B image.")

# ============== Main workflow ==============
def main():
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

    # ---- Extract features (2436 dims) ----
    print("[Feature] Extracting TRAIN ...")
    X_train = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test = extract_features(te_seq, k_max_cksaap=5)
    print("[Shapes] Train=%s, Test=%s (expect 2436 dims)" % (X_train.shape, X_test.shape))

    # ---- Stage 1: OOF / holdout probabilities from 4 base models ----
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds_base = list(skf_base.split(X_train, y_train))

    base_models = get_base_models()
    base_names = ["HistGradientBoosting", "LightGBM", "RandomForest", "XGBoost"]
    models_in_use = {name: base_models[name] for name in base_names}

    OOF_list, TEST_list = [], []
    for name in base_names:
        est = models_in_use[name]
        print("\n[Base] %s: unified 5-fold OOF & holdout ..." % name)
        oof, te = get_oof_and_holdout_with_folds(X_train, y_train, X_test, est, folds_base)
        OOF_list.append(oof)
        TEST_list.append(te)

    meta_train = np.vstack(OOF_list).T.astype(np.float32)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)
    print("[Meta] meta_train=%s, meta_test=%s" % (meta_train.shape, meta_test.shape))

    # ---- Standardize meta features ----
    scaler_meta = StandardScaler()
    META_TRAIN_S = scaler_meta.fit_transform(meta_train)
    META_TEST_S  = scaler_meta.transform(meta_test)

    # ---- Prepare for LNN: raw features -> standardize -> PCA(64) ----
    scaler_raw_lnn = StandardScaler().fit(X_train)
    X_train_s_lnn = scaler_raw_lnn.transform(X_train)
    X_test_s_lnn  = scaler_raw_lnn.transform(X_test)

    pca_lnn = PCA(n_components=64).fit(X_train_s_lnn)
    X_train_pca64 = pca_lnn.transform(X_train_s_lnn)
    X_test_pca64  = pca_lnn.transform(X_test_s_lnn)
    print("[LNN-PCA] X_train_pca64=%s, X_test_pca64=%s" % (X_train_pca64.shape, X_test_pca64.shape))

    # ---- Fuse meta + PCA(raw) as LNN input ----
    LNN_TRAIN = np.concatenate([META_TRAIN_S, X_train_pca64], axis=1)
    LNN_TEST  = np.concatenate([META_TEST_S,  X_test_pca64],  axis=1)
    print("[LNN-Input] LNN_TRAIN=%s, LNN_TEST=%s" % (LNN_TRAIN.shape, LNN_TEST.shape))

    # ---- Stage 2: LNN (CfC) 5-fold * n_repeats ensemble ----
    lnn_models, oof_train_proba = train_lnn_kfold_meta(
        LNN_TRAIN,
        y_train,
        n_splits=5,
        n_repeats=2,
        random_state=RANDOM_STATE,
        epochs=100,
        batch_size=32,
    )

    p_tr = oof_train_proba
    p_te = predict_ensemble_meta(lnn_models, LNN_TEST)

    # ---- Two-stage threshold search based on Train OOF probabilities ----
    thr_best, thr_stats_train = search_best_threshold_multi(y_train, p_tr)

    # ---- Compute metrics ----
    m_train = {"Set": "Train(OOF)", **metrics_from_proba(y_train, p_tr, thr_best)}
    m_test  = {"Set": "Test(Holdout)", **metrics_from_proba(y_test,  p_te, thr_best)}
    df_meta = pd.DataFrame([m_train, m_test])

    # ---- Output directory ----
    out_dir = "lnn_stack_outputs_optim_repeat_thrrefine"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Plot and save ROC/PRC ----
    plot_roc_prc(y_train, p_tr, "Train", out_dir)
    plot_roc_prc(
        y_test,
        p_te,
        "Test",
        out_dir,
        title_override=("LNN-ACP-Stack-Optim-Repeat-RefThr-Test ROC",
                        "LNN-ACP-Stack-Optim-Repeat-RefThr-Test PRC"),
    )

    # ================= Confusion matrices =================
    yb_train = (p_tr >= thr_best).astype(int)
    yb_test  = (p_te >= thr_best).astype(int)
    cm_train = confusion_matrix(y_train, yb_train, labels=[0, 1])
    cm_test  = confusion_matrix(y_test,  yb_test,  labels=[0, 1])

    plot_confmat(
        cm_train,
        title="Confusion Matrix — Train (LNN Stack Optim Repeat RefThr)",
        out_png=os.path.join(out_dir, "%s_CM_Train_red.png" % ts),
        main_color="#d32f2f",
    )
    plot_confmat(
        cm_test,
        title="Confusion Matrix — Test (LNN Stack Optim Repeat RefThr)",
        out_png=os.path.join(out_dir, "%s_CM_Test_deepblue.png" % ts),
        main_color="#0b3d91",
    )

    # ================= PCA / UMAP / t-SNE (raw features + LNN representations) =================
    # ---- Raw features standardize for visualization ----
    scaler_raw = StandardScaler().fit(X_train)
    X_train_s_raw = scaler_raw.transform(X_train)
    X_test_s_raw  = scaler_raw.transform(X_test)

    # ---------- PCA (Raw) ----------
    pca_raw_2d = PCA(n_components=2).fit(X_train_s_raw)
    raw_train_2d = pca_raw_2d.transform(X_train_s_raw)
    raw_test_2d  = pca_raw_2d.transform(X_test_s_raw)

    _scatter_pca(
        raw_train_2d,
        y_train,
        title="Raw Features PCA - Train",
        save_png=os.path.join(out_dir, "%s_PCA_Raw_Train.png" % ts),
    )
    _scatter_pca(
        raw_test_2d,
        y_test,
        title="Raw Features PCA - Test",
        save_png=os.path.join(out_dir, "%s_PCA_Raw_Test.png" % ts),
    )

    # ---------- Learned Representation extraction ----------
    rep_extractor = tf.keras.Model(
        inputs=lnn_models[0].input,
        outputs=lnn_models[0].get_layer("rep_dense").output,
    )
    rep_train = rep_extractor.predict(LNN_TRAIN, verbose=0)
    rep_test  = rep_extractor.predict(LNN_TEST,  verbose=0)

    scaler_rep = StandardScaler().fit(rep_train)
    rep_train_s = scaler_rep.transform(rep_train)
    rep_test_s  = scaler_rep.transform(rep_test)

    # ---------- PCA (Learned Rep) ----------
    pca_rep = PCA(n_components=2).fit(rep_train_s)
    rep_test_2d = pca_rep.transform(rep_test_s)

    _scatter_pca(
        rep_test_2d,
        y_test,
        title="Learned Representation PCA - Test",
        save_png=os.path.join(out_dir, "%s_PCA_LearnedRep_Test.png" % ts),
    )

    # ===================== UMAP =====================
    umap_raw = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
    ).fit(X_train_s_raw)

    raw_train_umap2d = umap_raw.transform(X_train_s_raw)
    raw_test_umap2d  = umap_raw.transform(X_test_s_raw)

    _scatter_umap(
        raw_train_umap2d,
        y_train,
        title="Raw Features UMAP - Train",
        save_png=os.path.join(out_dir, "%s_UMAP_Raw_Train.png" % ts),
    )
    _scatter_umap(
        raw_test_umap2d,
        y_test,
        title="Raw Features UMAP - Test",
        save_png=os.path.join(out_dir, "%s_UMAP_Raw_Test.png" % ts),
    )

    umap_rep = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
    ).fit(rep_train_s)

    rep_test_umap2d = umap_rep.transform(rep_test_s)

    _scatter_umap(
        rep_test_umap2d,
        y_test,
        title="Learned Representation UMAP - Test",
        save_png=os.path.join(out_dir, "%s_UMAP_LearnedRep_Test.png" % ts),
    )

    # ===================== t-SNE =====================
    p_train_tsne = _safe_tsne_perplexity(X_train_s_raw.shape[0])
    tsne_raw_train = TSNE(
        n_components=2,
        perplexity=p_train_tsne,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )
    raw_train_tsne2d = tsne_raw_train.fit_transform(X_train_s_raw)

    _scatter_tsne(
        raw_train_tsne2d,
        y_train,
        title="Raw Features t-SNE - Train",
        save_png=os.path.join(out_dir, "%s_tSNE_Raw_Train.png" % ts),
    )

    p_test_tsne = _safe_tsne_perplexity(X_test_s_raw.shape[0])
    tsne_raw_test = TSNE(
        n_components=2,
        perplexity=p_test_tsne,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )
    raw_test_tsne2d = tsne_raw_test.fit_transform(X_test_s_raw)

    _scatter_tsne(
        raw_test_tsne2d,
        y_test,
        title="Raw Features t-SNE - Test",
        save_png=os.path.join(out_dir, "%s_tSNE_Raw_Test.png" % ts),
    )

    p_rep_tsne = _safe_tsne_perplexity(rep_test_s.shape[0])
    tsne_rep_test = TSNE(
        n_components=2,
        perplexity=p_rep_tsne,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )
    rep_test_tsne2d = tsne_rep_test.fit_transform(rep_test_s)

    _scatter_tsne(
        rep_test_tsne2d,
        y_test,
        title="Learned Representation t-SNE - Test",
        save_png=os.path.join(out_dir, "%s_tSNE_LearnedRep_Test.png" % ts),
    )

    # ===================== SHAP =====================
    run_shap_analysis(X_train, y_train, out_dir, ts)

    # ---- Save Excel (including AP metric and threshold info) ----
    out_xlsx = os.path.join(out_dir, "LNN_ACP_Stack_AAC_CKSAAP_PCP16_Optim_Repeat_RefThr_%s.xlsx" % ts)
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_meta.to_excel(w, sheet_name="LNN_Train_vs_Test", index=False)
            pd.DataFrame(meta_train, columns=base_names).to_excel(
                w, sheet_name="MetaTrain_OOF", index=False
            )
            pd.DataFrame(meta_test, columns=base_names).to_excel(
                w, sheet_name="MetaTest_Preds", index=False
            )
            thr_info = {
                "Best_Threshold_from_Train_OOF": thr_best,
                "Train_OOF_ACC": thr_stats_train["ACC"],
                "Train_OOF_Precision": thr_stats_train["Precision"],
                "Train_OOF_Recall": thr_stats_train["Recall"],
                "Train_OOF_F1": thr_stats_train["F1"],
                "Train_OOF_MCC": thr_stats_train["MCC"],
                "Train_OOF_ThrScore": thr_stats_train["Score"],
            }
            pd.DataFrame([thr_info]).to_excel(w, sheet_name="LNN_Threshold", index=False)
        print("\nSaved to: %s" % out_xlsx)
    except Exception as e:
        print("[Warn] openpyxl not available, writing CSVs.", e)
        df_meta.to_csv(out_xlsx.replace(".xlsx", "_meta.csv"), index=False)

    print("\n=== LNN (CfC) ACP Stack model (AAC+CKSAAP+PCP16) — Optimized + Repeat Ensemble + Refined Threshold ===")
    print(df_meta)

    del lnn_models
    gc.collect()

if __name__ == "__main__":
    np.random.seed(42)
    main()
