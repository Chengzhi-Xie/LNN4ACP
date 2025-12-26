#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ================= Matplotlib backend fixed to Agg (no GUI, no tkinter) =================
import os
os.environ["MPLBACKEND"] = "Agg"   # Must be set before importing pyplot
import matplotlib
matplotlib.use("Agg")

# ======= Global Matplotlib font & color: Arial / Black / Bold (applies to all figures) =======
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.weight": "bold",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.frameon": True,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
# ================================================================================

"""
LNN4ACP — Prediction kept intact; Generation judged purely by LNN
"""

import warnings, gc, math
from datetime import datetime
from itertools import product
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from ncps.tf import CfC  # Liquid Neural Network cell (LNN)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# =================== Unified styling: force Arial/Black/Bold for each figure (incl. legend/ticks) ===================
def _apply_arial_black_bold(ax):
    """Force Arial + black + bold for title/labels/ticks/legend without changing other content."""
    # title / axis labels
    try:
        ax.title.set_fontname("Arial")
        ax.title.set_fontweight("bold")
        ax.title.set_color("black")
    except Exception:
        pass

    try:
        ax.xaxis.label.set_fontname("Arial")
        ax.xaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_fontname("Arial")
        ax.yaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_color("black")
    except Exception:
        pass

    # ticks
    try:
        ax.tick_params(axis="both", colors="black")
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lab.set_fontname("Arial")
            lab.set_fontweight("bold")
            lab.set_color("black")
    except Exception:
        pass

    # spines
    try:
        for sp in ax.spines.values():
            sp.set_color("black")
    except Exception:
        pass

    # legend
    leg = ax.get_legend()
    if leg is not None:
        try:
            for t in leg.get_texts():
                t.set_fontname("Arial")
                t.set_fontweight("bold")
                t.set_color("black")
            if leg.get_title() is not None:
                leg.get_title().set_fontname("Arial")
                leg.get_title().set_fontweight("bold")
                leg.get_title().set_color("black")
        except Exception:
            pass

def _apply_arial_black_bold_colorbar(cbar):
    """Force Arial + black + bold for colorbar tick labels."""
    if cbar is None:
        return
    try:
        cbar.ax.tick_params(colors="black")
        for lab in cbar.ax.get_yticklabels():
            lab.set_fontname("Arial")
            lab.set_fontweight("bold")
            lab.set_color("black")
        for lab in cbar.ax.get_xticklabels():
            lab.set_fontname("Arial")
            lab.set_fontweight("bold")
            lab.set_color("black")
    except Exception:
        pass
# ================================================================================

# =================== Tunable config (final default: potency-first + moderate diversity) ===================
CFG = {
    # Length preference (sensitivity)
    "len_pref_min": 10,
    "len_pref_max": 30,

    # Diversity settings
    "diversity_use": True,
    "diversity_k": 30,              # Candidate pool size for elite re-ranking (picked from top-k_pool)
    "diversity_k_pool": 60,         # Top-K pool used for re-ranking each generation
    "diversity_alpha_start": 0.15,  # Annealing: weaker penalty in early stage
    "diversity_alpha_end": 0.60,    # Annealing: moderate penalty in late stage
    "diversity_lambda": 0.35,       # Softened penalty weight (smaller -> better preserves potency density)
    "diversity_use_fitness_penalty": True,  # Whether to include softened penalty in parent fitness

    # Ablation & plots
    "run_ablation": True,

    # External safety optional CSV
    "external_safety_csv": "extras/external_safety.csv",

    # Post-hoc balanced top export
    "posthoc_balance": True,
    "posthoc_pool": 200,      # Final balanced re-ranking from overall top-200 candidates
    "posthoc_topN": 50,       # Export Balanced Top50 (also slices Top20)

    # GA config defaults
    "pop_size": 80,
    "n_generations": 30,
    "elite_size": 20,
    "mut_rate": 0.25,
}

# =================== TF GPU (on-demand memory growth) ===================
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

# =================== Data loading (two-line FASTA) ===================
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

# =================== Feature engineering (kept unchanged) ===================
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
DIPEPTIDES = ["".join(p) for p in product(AA20, repeat=2)]  # 400

PCP16_MAP = {
    "acidic": set("DE"), "aliphatic": set("ILV"), "aromatic": set("FYW"),
    "basic": set("KRH"), "charged": set("DEKRH"), "cyclic": set("P"),
    "hydrophilic": set("RNDQEHKST"), "hydrophobic": set("AILMFWV"),
    "hydroxylic": set("ST"), "neutral_pH": set("ACFGHILMNPQSTVWY"),
    "nonpolar": set("ACFGILMPVWY"), "polar": set("RNDQEHKTYS"),
    "small": set("ACDGNPSTV"), "large": set("EFHKLMQRWY"),
    "sulfur": set("CM"), "tiny": set("ACGST"),
}
PCP16_KEYS = list(PCP16_MAP.keys())

def _check_seq(seq: str) -> str:
    s = seq.strip().upper().replace(" ", "")
    if not s or any(ch not in AA_SET for ch in s):
        raise ValueError(f"Invalid residues in sequence: {seq}")
    return s

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
    return np.array([sum(ch in PCP16_MAP[key] for ch in s) / L for key in PCP16_KEYS], dtype=np.float32)  # 16

def extract_features(seqs: List[str], k_max_cksaap: int = 5) -> np.ndarray:
    feats = []
    for seq in seqs:
        feats.append(np.concatenate([aac_vector(seq),
                                     cksaap_vector(seq, k_max=k_max_cksaap),
                                     pcp16_vector(seq)], axis=0))
    X = np.vstack(feats).astype(np.float32)
    assert X.shape[1] == 2436, f"Unexpected feature dim {X.shape[1]} != 2436"
    return X

# =================== Basic/extended physicochemical properties ===================
KD_SCALE = { # Kyte-Doolittle
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,
    'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}
EISENBERG_SCALE = { # for μH (alpha 100°)
    'A':0.62,'C':0.29,'D':-0.90,'E':-0.74,'F':1.19,'G':0.48,'H':-0.40,'I':1.38,'K':-1.50,
    'L':1.06,'M':0.64,'N':-0.78,'P':0.12,'Q':-0.85,'R':-2.53,'S':-0.18,'T':-0.05,'V':1.08,'W':0.81,'Y':0.26
}

def hydrophobic_moment_alpha(seq: str, angle_deg: float = 100.0) -> float:
    s = _check_seq(seq)
    angle = math.radians(angle_deg)
    x = y = 0.0
    for i, aa in enumerate(s):
        h = EISENBERG_SCALE.get(aa, 0.0)
        x += h * math.cos(i * angle)
        y += h * math.sin(i * angle)
    muH = math.sqrt(x * x + y * y) / max(len(s), 1)
    return float(muH)

def kyte_doolittle_gravy(seq: str) -> float:
    s = _check_seq(seq)
    return float(sum(KD_SCALE.get(a, 0.0) for a in s) / max(1, len(s)))

def aliphatic_index(seq: str) -> float:
    s = _check_seq(seq)
    L = len(s)
    if L == 0: return 0.0
    ala = s.count('A'); val = s.count('V'); ile = s.count('I'); leu = s.count('L')
    return float((ala*1.0 + val*2.9 + (ile+leu)*3.9) * 100.0 / L)

def compute_basic_props(seq: str) -> Dict[str, float]:
    s = _check_seq(seq)
    L = len(s)
    pos_set, neg_set = set("KRH"), set("DE")
    hydrophobic_set, aromatic_set = set("AILMFWV"), set("FYW")
    pos = sum(ch in pos_set for ch in s)
    neg = sum(ch in neg_set for ch in s)
    net_charge = float(pos - neg)
    hydrophobic_count = sum(ch in hydrophobic_set for ch in s)
    aromatic_count = sum(ch in aromatic_set for ch in s)
    c_count = s.count("C")
    longest_h_run, cur = 0, 0
    for ch in s:
        if ch in hydrophobic_set:
            cur += 1; longest_h_run = max(longest_h_run, cur)
        else:
            cur = 0
    return {
        "Length": float(L),
        "NetCharge": net_charge,
        "HydrophobicFrac": hydrophobic_count / L if L else 0.0,
        "AromaticFrac": aromatic_count / L if L else 0.0,
        "CFrac": c_count / L if L else 0.0,
        "LongestHydrophobicRun": float(longest_h_run),
        "GRAVY": kyte_doolittle_gravy(s),
        "HydrophobicMoment": hydrophobic_moment_alpha(s),
        "AliphaticIndex": aliphatic_index(s),
    }

def compute_positive_props_stats(pos_seqs: List[str]) -> Dict[str, Dict[str, float]]:
    props_list = [compute_basic_props(s) for s in pos_seqs]
    if not props_list: return {}
    keys = props_list[0].keys()
    stats = {}
    for k in keys:
        vals = np.array([p[k] for p in props_list], dtype=np.float32)
        stats[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals) + 1e-6)}
    return stats

# =================== k-mer Jaccard (diversity) ===================
def kmer_set(seq: str, k: int = 3) -> set:
    s = _check_seq(seq)
    if len(s) < k: return set()
    return {s[i:i+k] for i in range(len(s)-k+1)}

def build_pos_kmer_sets(pos_seqs: List[str], k: int = 3) -> List[set]:
    return [kmer_set(s, k=k) for s in pos_seqs]

def max_kmer_jaccard_to_pos(seq: str, pos_kmer_sets: List[set], k: int = 3) -> float:
    ks = kmer_set(seq, k=k)
    if not ks or not pos_kmer_sets: return 0.0
    max_j = 0.0
    for ps in pos_kmer_sets:
        if not ps: continue
        inter, union = len(ks & ps), len(ks | ps)
        if union == 0: continue
        max_j = max(max_j, inter / union)
    return float(max_j)

def novelty_score_from_jaccard(max_j: float) -> float:
    if max_j >= 0.90: return 0.0
    if max_j <= 0.50: return 1.0
    return float((0.90 - max_j) / (0.90 - 0.50))

def jaccard_sets(a: set, b: set) -> float:
    if not a and not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return 0.0 if union == 0 else inter / union

# =================== Metrics & threshold search (kept) ===================
def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    yb = (proba >= thr).astype(int)
    return dict(ACC=accuracy_score(y_true, yb),
                Precision=precision_score(y_true, yb, zero_division=0),
                Recall=recall_score(y_true, yb),
                F1=f1_score(y_true, yb),
                AUC=roc_auc_score(y_true, proba),
                AP=average_precision_score(y_true, proba),
                MCC=matthews_corrcoef(y_true, yb),
                Threshold=float(thr))

def search_best_threshold_multi(y_true: np.ndarray, proba: np.ndarray, thr_grid=None):
    if thr_grid is None: thr_grid = np.linspace(0.1, 0.9, 81)
    best_thr, best_score, best_stats = 0.5, -1.0, None
    for thr in thr_grid:
        yb = (proba >= thr).astype(int)
        acc = accuracy_score(y_true, yb)
        prec = precision_score(y_true, yb, zero_division=0)
        rec = recall_score(y_true, yb)
        f1 = f1_score(y_true, yb)
        mcc = matthews_corrcoef(y_true, yb)
        score = f1 + 0.3*mcc + 0.1*(acc - 0.5)
        if score > best_score:
            best_thr, best_score = float(thr), score
            best_stats = dict(ACC=acc, Precision=prec, Recall=rec, F1=f1, MCC=mcc, Score=score)
    print("[ThresholdSearch-Multi] best_thr=%.4f | ACC=%.4f, P=%.4f, R=%.4f, F1=%.4f, MCC=%.4f, Score=%.4f"
          % (best_thr, best_stats["ACC"], best_stats["Precision"], best_stats["Recall"],
             best_stats["F1"], best_stats["MCC"], best_stats["Score"]))
    return best_thr, best_stats

# =================== Plotting (only font/bold/black changes; other logic kept) ===================
def _save_curve_points_roc(fpr, tpr, thr, save_csv):
    pd.DataFrame({"FPR":fpr,"TPR":tpr,"Threshold":thr}).to_csv(save_csv, index=False)

def _save_curve_points_prc(precision, recall, thr, save_csv):
    thr_full = np.concatenate([thr, [np.nan]]) if thr is not None else np.full_like(precision, np.nan)
    pd.DataFrame({"Recall":recall,"Precision":precision,"Threshold":thr_full}).to_csv(save_csv, index=False)

def plot_roc_prc(y_true, proba, set_name:str, out_dir:str,
                 title_prefix:str="LNN-Stack (AAC+CKSAAP+PCP16)", title_override=None):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ROC
    fpr, tpr, thr = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.4f})", linewidth=2)
    ax.plot([0,1], [0,1], "k--", label="Random", linewidth=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold", color="black", fontname="Arial")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold", color="black", fontname="Arial")
    roc_title = title_override[0] if title_override else f"{title_prefix} — {set_name} ROC"
    ax.set_title(roc_title, fontsize=14, fontweight="bold", color="black", fontname="Arial")
    ax.legend(loc="lower right", prop={"family":"Arial","weight":"bold"})
    _apply_arial_black_bold(ax)
    roc_png = os.path.join(out_dir, f"{ts}_{set_name}_ROC.png")
    fig.tight_layout()
    fig.savefig(roc_png, dpi=300)
    plt.close(fig)
    _save_curve_points_roc(fpr, tpr, thr, os.path.join(out_dir, f"{ts}_{set_name}_ROC_points.csv"))

    # PRC
    precision, recall, thr_pr = precision_recall_curve(y_true, proba)
    ap_val = average_precision_score(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PRC (AP = {ap_val:.4f})", linewidth=2)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold", color="black", fontname="Arial")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold", color="black", fontname="Arial")
    prc_title = title_override[1] if title_override else f"{title_prefix} — {set_name} PRC"
    ax.set_title(prc_title, fontsize=14, fontweight="bold", color="black", fontname="Arial")
    ax.legend(loc="lower left", prop={"family":"Arial","weight":"bold"})
    _apply_arial_black_bold(ax)
    prc_png = os.path.join(out_dir, f"{ts}_{set_name}_PRC.png")
    fig.tight_layout()
    fig.savefig(prc_png, dpi=300)
    plt.close(fig)
    _save_curve_points_prc(precision, recall, thr_pr, os.path.join(out_dir, f"{ts}_{set_name}_PRC_points.csv"))

    print(f"[Plot] {set_name}: ROC→{roc_png} | PRC→{prc_png}")

def _single_color_cmap(hex_color: str):
    return LinearSegmentedColormap.from_list("single_color", ["#FFFFFF", hex_color], N=256)

def plot_confmat(cm: np.ndarray, title: str, out_png: str, main_color: str = "#d32f2f"):
    cmap = _single_color_cmap(main_color)
    fig, ax = plt.subplots(figsize=(5.2,4.6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=14, fontweight="bold", color="black", fontname="Arial")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _apply_arial_black_bold_colorbar(cbar)

    ticks = ["Negative","Positive"]
    ax.set_xticks(np.arange(2)); ax.set_yticks(np.arange(2))
    ax.set_xticklabels(ticks, fontsize=11, fontweight="bold", color="black", fontname="Arial")
    ax.set_yticklabels(ticks, fontsize=11, fontweight="bold", color="black", fontname="Arial")
    ax.set_ylabel("True label", fontsize=12, fontweight="bold", color="black", fontname="Arial")
    ax.set_xlabel("Predicted label", fontsize=12, fontweight="bold", color="black", fontname="Arial")

    thresh = cm.max()/2.0 if cm.size else 0
    for i in range(2):
        for j in range(2):
            val = cm[i,j]
            ax.text(j, i, f"{val:d}", ha="center", va="center",
                    color="white" if val>thresh else "black",
                    fontsize=12, fontweight="bold", fontname="Arial")
    _apply_arial_black_bold(ax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("[CM] Saved:", out_png)

# =================== Stage 1: 4 base models (kept unchanged) ===================
def get_base_models() -> Dict[str, Any]:
    models={}
    try:
        models["LightGBM"]=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,learning_rate=0.05,
                                              subsample=0.9,colsample_bytree=0.9,random_state=42,
                                              n_jobs=-1,device="gpu")
        _=models["LightGBM"].get_params()
    except Exception:
        try:
            models["LightGBM"]=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,learning_rate=0.05,
                                                  subsample=0.9,colsample_bytree=0.9,random_state=42,
                                                  n_jobs=-1,device_type="gpu")
        except Exception:
            models["LightGBM"]=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,learning_rate=0.05,
                                                  subsample=0.9,colsample_bytree=0.9,random_state=42,
                                                  n_jobs=-1)
    models["RandomForest"]=RandomForestClassifier(n_estimators=300,max_depth=None,n_jobs=-1,random_state=42)
    models["XGBoost"]=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.9,
                                    colsample_bytree=0.9,eval_metric="logloss",random_state=42,
                                    n_jobs=-1,tree_method="hist")
    models["HistGradientBoosting"]=HistGradientBoostingClassifier(random_state=42)
    return models

def fit_base_model_full(estimator, X, y):
    if isinstance(estimator, XGBClassifier):
        gpu_params = estimator.get_params()
        gpu_params.update({"tree_method":"gpu_hist","predictor":"gpu_predictor"})
        try:
            clf = XGBClassifier(**gpu_params)
            clf.fit(X,y)
            print("[XGB][Full] Using GPU (gpu_hist).")
            return clf
        except XGBoostError as e:
            print("[XGB][Full] GPU not available, fallback to CPU hist.", e)
            cpu_params = estimator.get_params()
            cpu_params.pop("tree_method",None)
            cpu_params.pop("predictor",None)
            cpu_params.update({"tree_method":"hist"})
            clf = XGBClassifier(**cpu_params)
            clf.fit(X,y)
            return clf
    else:
        clf = estimator.__class__(**estimator.get_params())
        clf.fit(X,y)
        return clf

def get_oof_and_holdout_with_folds(X_tr, y_tr, X_ho, estimator, folds):
    oof = np.zeros(X_tr.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_ho.shape[0], len(folds)), dtype=np.float32)
    for k,(tr_idx,va_idx) in enumerate(folds):
        base_params = estimator.get_params()
        clf = estimator.__class__(**base_params)
        X_tr_fold,y_tr_fold = X_tr[tr_idx], y_tr[tr_idx]
        X_va_fold = X_tr[va_idx]

        if isinstance(clf, XGBClassifier):
            gpu_params = clf.get_params()
            gpu_params.update({"tree_method":"gpu_hist","predictor":"gpu_predictor"})
            try:
                clf = XGBClassifier(**gpu_params)
                clf.fit(X_tr_fold,y_tr_fold)
                print(f"[XGB][Fold {k+1}] Using GPU.")
            except XGBoostError as e:
                print(f"[XGB][Fold {k+1}] GPU NA, fallback to CPU. Error: {e}")
                cpu_params = clf.get_params()
                cpu_params.pop("tree_method",None)
                cpu_params.pop("predictor",None)
                cpu_params.update({"tree_method":"hist"})
                clf = XGBClassifier(**cpu_params)
                clf.fit(X_tr_fold,y_tr_fold)
        else:
            clf.fit(X_tr_fold,y_tr_fold)

        if hasattr(clf,"predict_proba"):
            oof[va_idx]=clf.predict_proba(X_va_fold)[:,1]
            hold_mat[:,k]=clf.predict_proba(X_ho)[:,1]
        else:
            s_va=clf.decision_function(X_va_fold)
            s_ho=clf.decision_function(X_ho)
            s_va=(s_va-s_va.min())/(s_va.max()-s_va.min()+1e-12)
            s_ho=(s_ho-s_ho.min())/(s_ho.max()-s_ho.min()+1e-12)
            oof[va_idx]=s_va
            hold_mat[:,k]=s_ho
    return oof, hold_mat.mean(axis=1).astype(np.float32)

# =================== Stage 2: LNN (CfC) (kept unchanged) ===================
def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(y.astype(int)); total=float(len(y))
    return {cls: total/(2.0*count) for cls,count in enumerate(counts) if count>0}

def build_lnn_meta(input_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,), name="meta_plus_pca_input")
    x = layers.Dense(128, activation="relu")(inp); x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x); x = layers.Dropout(0.3)(x)
    x = layers.Reshape((1,64))(x)
    x = CfC(64, mixed_memory=True, backbone_units=32, backbone_layers=1,
            backbone_dropout=0.1, return_sequences=True, name="cfc1")(x)
    x = CfC(64, mixed_memory=True, backbone_units=32, backbone_layers=1,
            backbone_dropout=0.1, return_sequences=False, name="cfc2")(x)
    x = layers.Dense(64, activation="relu", name="rep_dense")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = models.Model(inp, out, name="ACP_LNN_Meta_Fused")
    model.compile(optimizer=optimizers.Adam(3e-4),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_lnn_kfold_meta(X_full: np.ndarray, y: np.ndarray,
                         n_splits:int=5, n_repeats:int=2, random_state:int=42,
                         epochs:int=100, batch_size:int=32):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    models_list=[]
    for fold,(tr_idx,va_idx) in enumerate(skf.split(X_full,y)):
        print(f"\n[LNN Meta] Fold {fold+1}/{n_splits}")
        X_tr,X_va = X_full[tr_idx],X_full[va_idx]
        y_tr,y_va = y[tr_idx],y[va_idx]
        class_weight=get_class_weights(y_tr)
        preds_va_all=[]
        for rep in range(n_repeats):
            print(f"[LNN Meta] Fold {fold+1} Repeat {rep+1}/{n_repeats}")
            tf.keras.utils.set_random_seed(random_state + fold*100 + rep)
            model=build_lnn_meta(X_full.shape[1])
            es=callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)
            rl=callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=7,
                                           min_lr=1e-6,verbose=1)
            model.fit(X_tr,y_tr,epochs=epochs,batch_size=batch_size,verbose=1,
                      validation_data=(X_va,y_va),
                      callbacks=[es,rl],class_weight=class_weight)
            preds_va = model.predict(X_va,verbose=0).ravel()
            preds_va_all.append(preds_va)
            models_list.append(model)
        oof_pred[va_idx]=np.mean(np.vstack(preds_va_all),axis=0)
    return models_list, oof_pred

def predict_ensemble_meta(models_list, X_full: np.ndarray) -> np.ndarray:
    return np.mean(np.vstack([m.predict(X_full,verbose=0).ravel() for m in models_list]), axis=0)

# =================== Sequence-level LNN (kept unchanged) ===================
AA_TO_INDEX = {aa:i for i,aa in enumerate(AA20)}

def encode_seq_onehot(seq: str, max_len: int) -> np.ndarray:
    s=_check_seq(seq)
    L=len(s)
    arr=np.zeros((max_len,len(AA20)),dtype=np.float32)
    for i in range(min(L,max_len)):
        aa=s[i]
        if aa in AA_TO_INDEX:
            arr[i,AA_TO_INDEX[aa]]=1.0
    return arr

def build_seq_lnn(max_len: int) -> tf.keras.Model:
    inp = layers.Input(shape=(max_len, len(AA20)), name="seq_input")
    x = CfC(64, mixed_memory=True, backbone_units=32, backbone_layers=1,
            backbone_dropout=0.1, return_sequences=False, name="seq_cfc")(inp)
    x = layers.Dense(64, activation="relu", name="seq_dense")(x)
    x = layers.Dropout(0.4, name="seq_dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="seq_output")(x)
    model = models.Model(inp, out, name="ACP_LNN_Seq")
    model.compile(optimizer=optimizers.Adam(5e-4),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_seq_lnn(tr_seqs: List[str], y_train: np.ndarray,
                  epochs:int=80, batch_size:int=32, random_state:int=42):
    max_len = max(len(s) for s in tr_seqs)
    print(f"[Seq-LNN] MaxLen={max_len}")
    X = np.stack([encode_seq_onehot(s,max_len) for s in tr_seqs])
    y = np.asarray(y_train).astype(np.float32)
    class_weight=get_class_weights(y_train)
    tf.keras.utils.set_random_seed(random_state)
    model = build_seq_lnn(max_len)
    es=callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
    rl=callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=5,
                                   min_lr=1e-6,verbose=1)
    model.fit(X,y,validation_split=0.2,epochs=epochs,batch_size=batch_size,verbose=1,
              callbacks=[es,rl],class_weight=class_weight)
    y_pred=model.predict(X,verbose=0).ravel()
    print(f"[Seq-LNN] Train AUC={roc_auc_score(y,y_pred):.4f}, AP={average_precision_score(y,y_pred):.4f}")
    return model,max_len

def predict_seq_lnn(model: tf.keras.Model, seqs: List[str], max_len: int) -> np.ndarray:
    X=np.stack([encode_seq_onehot(s,max_len) for s in seqs])
    return model.predict(X,verbose=0).ravel().astype(np.float32)

# =================== Hemolysis/cytotoxicity proxy risk + external fusion ===================
def _logistic(x: float) -> float:
    return float(1.0/(1.0+math.exp(-x)))

def hemolysis_risk_from_props(p: Dict[str,float]) -> float:
    x = 0.0
    x += 4.0*(p["HydrophobicFrac"] - 0.45)
    x += 3.0*(p["HydrophobicMoment"] - 0.35)
    x += 0.35*(p["NetCharge"] - 6.0)
    x += 0.50*max(0.0, p["LongestHydrophobicRun"] - 4.0)
    x += 0.05*max(0.0, p["Length"] - CFG["len_pref_max"])
    return float(np.clip(_logistic(x), 0.0, 1.0))

def cytotox_risk_from_props(p: Dict[str,float]) -> float:
    x = 0.0
    x += 3.2*(p["HydrophobicMoment"] - 0.32)
    x += 2.5*(p["HydrophobicFrac"] - 0.42)
    x += 0.30*(p["NetCharge"] - 6.0)
    x += 1.5*(p["AromaticFrac"] - 0.10)
    return float(np.clip(_logistic(x), 0.0, 1.0))

def merge_external_safety_predictions(df: pd.DataFrame, csv_path: str, w_ext: float = 0.5) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return df
    ext = pd.read_csv(csv_path).drop_duplicates("Sequence")
    df = df.merge(ext, on="Sequence", how="left")
    if "HemolysisProb" in df.columns:
        df["HemolysisRisk"] = np.where(df["HemolysisProb"].notna(),
                                       w_ext*df["HemolysisProb"] + (1-w_ext)*df["HemolysisRisk"],
                                       df["HemolysisRisk"])
    if "ToxProb" in df.columns:
        df["CytotoxRisk"] = np.where(df["ToxProb"].notna(),
                                     w_ext*df["ToxProb"] + (1-w_ext)*df["CytotoxRisk"],
                                     df["CytotoxRisk"])
    if "CellLysisProb" in df.columns:
        df["CytotoxRisk"] = np.where(df["CellLysisProb"].notna(),
                                     0.5*df["CellLysisProb"] + 0.5*df["CytotoxRisk"],
                                     df["CytotoxRisk"])
    return df

# =================== Safety/Developability (incl. length 10–30 preference) ===================
def score_safety(props: Dict[str,float], pos_stats: Dict[str,Dict[str,float]]) -> float:
    L, net_charge, hyd_frac, long_run, c_frac = (
        props["Length"], props["NetCharge"], props["HydrophobicFrac"],
        props["LongestHydrophobicRun"], props["CFrac"]
    )
    charge_risk = 0.3 if net_charge<=2 else (0.4 if net_charge<=6 else (0.7 if net_charge<=8 else 0.9))
    hydro_risk  = 0.3 if hyd_frac<=0.30 else (0.4 if hyd_frac<=0.50 else (0.7 if hyd_frac<=0.60 else 0.9))
    run_risk    = 0.2 if long_run<3 else (0.4 if long_run<4 else (0.6 if long_run<6 else 0.9))
    cys_risk    = 0.2 if c_frac<0.08 else (0.5 if c_frac<0.15 else 0.8)

    if L < CFG["len_pref_min"]:
        len_pen = min(1.0, (CFG["len_pref_min"] - L) / 10.0)
    elif L > CFG["len_pref_max"]:
        len_pen = min(1.0, (L - CFG["len_pref_max"]) / 10.0)
    else:
        len_pen = 0.0

    combined_risk = 0.35*hydro_risk + 0.25*charge_risk + 0.20*run_risk + 0.10*cys_risk + 0.10*len_pen
    safety_base = 1.0 - combined_risk

    z_list=[]
    for key in ["Length","NetCharge","HydrophobicFrac","AromaticFrac","HydrophobicMoment","GRAVY"]:
        mu = pos_stats.get(key,{}).get("mean",None)
        sd = pos_stats.get(key,{}).get("std",None)
        if mu is None: continue
        z_list.append(abs(props[key]-mu)/(sd+1e-6))
    ood_penalty = 1.0 if not z_list else (
        1.0 if max(z_list)<=1.0 else (0.0 if max(z_list)>=3.0 else 1.0-(max(z_list)-1.0)/2.0)
    )
    return float(np.clip(safety_base*ood_penalty, 0.0, 1.0))

def score_developability(props: Dict[str,float]) -> float:
    L, net_charge, hyd_frac, c_frac = (
        props["Length"], props["NetCharge"], props["HydrophobicFrac"], props["CFrac"]
    )
    len_score  = 1.0 if CFG["len_pref_min"]<=L<=CFG["len_pref_max"] else (0.7 if (8<=L<=35) else 0.3)
    hyd_score  = 1.0 if 0.35<=hyd_frac<=0.55 else (0.7 if 0.25<=hyd_frac<=0.65 else 0.3)
    chg_score  = 1.0 if 2<=net_charge<=7 else (0.7 if 0<=net_charge<=9 else 0.3)
    cys_score  = 1.0 if c_frac<=0.05 else (0.7 if c_frac<=0.10 else 0.4)
    return float(np.clip(0.40*len_score + 0.25*hyd_score + 0.25*chg_score + 0.10*cys_score, 0.0, 1.0))

# =================== GA mutation/crossover (kept) ===================
def mutate_seq(seq: str, mut_rate: float = 0.2) -> str:
    s=list(_check_seq(seq)); L=len(s)
    for i in range(L):
        if np.random.rand()<mut_rate:
            s[i]=AA20[np.random.randint(0,len(AA20))]
    if L>5 and np.random.rand()<0.1:
        del s[np.random.randint(0,len(s))]
    if L<40 and np.random.rand()<0.1:
        s.insert(np.random.randint(0,len(s)+1), AA20[np.random.randint(0,len(AA20))])
    return "".join(s)

def crossover_seq(p1: str, p2: str) -> str:
    s1=_check_seq(p1); s2=_check_seq(p2); L=min(len(s1),len(s2))
    if L<=2: return s1
    cut=np.random.randint(1,L-1)
    return s1[:cut]+s2[cut:]

# =================== Candidate evaluation (LNN as the core discriminator) ===================
def evaluate_candidates_with_lnn(
    seqs: List[str],
    lnn_models_meta: List[tf.keras.Model],
    base_models_full: Dict[str, Any],
    scaler_meta: StandardScaler,
    scaler_raw_lnn: StandardScaler,
    pca_lnn: PCA,
    seq_lnn_model: tf.keras.Model,
    seq_lnn_max_len: int,
    pos_prop_stats: Dict[str, Dict[str, float]],
    pos_kmer_sets: List[set],
    train_neg_set: set,
    external_safety_csv: str,
) -> pd.DataFrame:

    unique_seqs = sorted(set([_check_seq(s) for s in seqs]))
    if not unique_seqs:
        return pd.DataFrame()

    # --- LNN_meta input (base learners only provide input features)
    X_raw = extract_features(unique_seqs, k_max_cksaap=5)
    X_raw_s = scaler_raw_lnn.transform(X_raw)
    X_pca64 = pca_lnn.transform(X_raw_s)

    meta_cols=[]
    for _, mdl in base_models_full.items():
        if hasattr(mdl,"predict_proba"):
            proba=mdl.predict_proba(X_raw)[:,1]
        else:
            s = mdl.decision_function(X_raw)
            proba=(s-s.min())/(s.max()-s.min()+1e-12)
        meta_cols.append(proba.astype(np.float32))
    meta_mat = np.vstack(meta_cols).T
    META_S = scaler_meta.transform(meta_mat)
    LNN_INPUT = np.concatenate([META_S, X_pca64], axis=1)

    # --- LNN scoring (the only activity source)
    acp_meta = predict_ensemble_meta(lnn_models_meta, LNN_INPUT)
    acp_seq  = predict_seq_lnn(seq_lnn_model, unique_seqs, seq_lnn_max_len)

    is_train_neg = np.array([s in train_neg_set for s in unique_seqs], dtype=bool)
    acp_meta_adj = acp_meta.copy()
    acp_seq_adj  = acp_seq.copy()
    acp_meta_adj[is_train_neg] = np.minimum(acp_meta_adj[is_train_neg], 0.1)
    acp_seq_adj[is_train_neg]  = np.minimum(acp_seq_adj[is_train_neg], 0.1)

    potency = (0.6*acp_meta_adj + 0.4*acp_seq_adj).astype(np.float32)

    # --- Safety/developability/diversity
    props_list = [compute_basic_props(s) for s in unique_seqs]
    safety_local = np.array([score_safety(p, pos_prop_stats) for p in props_list], dtype=np.float32)
    dev_arr      = np.array([score_developability(p) for p in props_list], dtype=np.float32)
    max_jacc_list = [max_kmer_jaccard_to_pos(s, pos_kmer_sets, k=3) for s in unique_seqs]
    novelty_arr   = np.array([novelty_score_from_jaccard(j) for j in max_jacc_list], dtype=np.float32)

    hemo_risk = np.array([hemolysis_risk_from_props(p) for p in props_list], dtype=np.float32)
    cyto_risk = np.array([cytotox_risk_from_props(p) for p in props_list], dtype=np.float32)

    df_tmp = pd.DataFrame({"Sequence": unique_seqs,"HemolysisRisk": hemo_risk,"CytotoxRisk": cyto_risk})
    df_tmp = merge_external_safety_predictions(df_tmp, external_safety_csv, w_ext=0.5)
    hemo_risk = df_tmp["HemolysisRisk"].values.astype(np.float32)
    cyto_risk = df_tmp["CytotoxRisk"].values.astype(np.float32)

    non_hemolytic_flag = (hemo_risk < 0.30)
    low_cytotox_flag   = (cyto_risk < 0.30)

    penalty = (1.0 - np.maximum(hemo_risk, cyto_risk))
    safety_composite = np.clip(safety_local * penalty, 0.0, 1.0)
    safety_composite = np.where((hemo_risk>=0.60)|(cyto_risk>=0.60),
                                np.minimum(safety_composite, 0.20), safety_composite)

    # --- GlobalScore (for ranking/export; Potency still comes only from LNN)
    global_score = (0.35*potency + 0.25*safety_composite + 0.20*dev_arr +
                    0.15*novelty_arr + 0.05*penalty).astype(np.float32)

    rows=[]
    for i,s in enumerate(unique_seqs):
        p = props_list[i]
        rows.append({
            "Sequence": s,
            "ACP_LNN_Meta": float(acp_meta_adj[i]),
            "ACP_LNN_Seq": float(acp_seq_adj[i]),
            "PotencyScore": float(potency[i]),
            "SafetyScore_Local": float(safety_local[i]),
            "DevelopabilityScore": float(dev_arr[i]),
            "MaxJaccard3Mer_Pos": float(max_jacc_list[i]),
            "NoveltyScoreKmer": float(novelty_arr[i]),
            "HemolysisRisk": float(hemo_risk[i]),
            "CytotoxRisk": float(cyto_risk[i]),
            "NonHemolyticFlag": bool(non_hemolytic_flag[i]),
            "LowCytotoxFlag": bool(low_cytotox_flag[i]),
            "SafetyComposite": float(safety_composite[i]),
            "GlobalScore": float(global_score[i]),
            "Length": p["Length"],
            "NetCharge": p["NetCharge"],
            "HydrophobicFrac": p["HydrophobicFrac"],
            "AromaticFrac": p["AromaticFrac"],
            "CFrac": p["CFrac"],
            "LongestHydrophobicRun": p["LongestHydrophobicRun"],
            "GRAVY": p["GRAVY"],
            "HydrophobicMoment": p["HydrophobicMoment"],
            "AliphaticIndex": p["AliphaticIndex"],
        })
    df = pd.DataFrame(rows)

    # Hard filter: remove high-risk candidates
    df = df[(df["NonHemolyticFlag"]) & (df["LowCytotoxFlag"])].copy()
    return df.sort_values("GlobalScore", ascending=False).reset_index(drop=True)

# =================== Diversity re-ranking (elites) ===================
def diversify_greedy(df_gen: pd.DataFrame, k: int, alpha: float) -> List[str]:
    """
    From df_gen (already sorted by GlobalScore descending), select k sequences by greedily maximizing:
      adjusted = GlobalScore * (1 - max_sim_to_selected)^alpha
    """
    if df_gen is None or df_gen.empty:
        return []
    cand = df_gen.copy()
    selected = []
    selected_kmer = []

    first = cand.iloc[0]
    selected.append(first["Sequence"])
    selected_kmer.append(kmer_set(first["Sequence"], 3))

    while len(selected) < min(k, len(cand)):
        best_idx, best_val = None, -1.0
        for idx, row in cand.iterrows():
            seq = row["Sequence"]
            if seq in selected:
                continue
            ks = kmer_set(seq, 3)
            sim = 0.0
            for ks_sel in selected_kmer:
                sim = max(sim, jaccard_sets(ks, ks_sel))
            penal = (1.0 - sim) ** alpha
            val = float(row["GlobalScore"]) * penal
            if val > best_val:
                best_val, best_idx = val, idx
        if best_idx is None:
            break
        row = cand.loc[best_idx]
        selected.append(row["Sequence"])
        selected_kmer.append(kmer_set(row["Sequence"], 3))
    return selected

def get_alpha_for_generation(gen: int, n_generations: int) -> float:
    if n_generations <= 1:
        return float(CFG["diversity_alpha_end"])
    a0 = float(CFG["diversity_alpha_start"])
    a1 = float(CFG["diversity_alpha_end"])
    t = float(gen) / float(n_generations - 1)
    return a0 + (a1 - a0) * t

def compute_soft_diversity_fitness(df_gen: pd.DataFrame, elites: List[str], alpha: float, lam: float) -> np.ndarray:
    """
    Softened diversity penalty:
      fitness = GlobalScore * ((1-lam) + lam*(1-sim)^alpha)
    sim is the maximum 3-mer Jaccard similarity from candidate to elites
    """
    if df_gen.empty:
        return np.array([], dtype=np.float32)
    elite_kmers = [kmer_set(s,3) for s in elites] if elites else []
    fitness = []
    for _, row in df_gen.iterrows():
        seq = row["Sequence"]
        base = float(row["GlobalScore"])
        if not elite_kmers:
            fitness.append(max(1e-6, base))
            continue
        ks = kmer_set(seq,3)
        sim = 0.0
        for eks in elite_kmers:
            sim = max(sim, jaccard_sets(ks, eks))
        penal = (1.0 - sim) ** alpha
        mixed = base * ((1.0 - lam) + lam * penal)
        fitness.append(max(1e-6, mixed))
    return np.array(fitness, dtype=np.float32)

# =================== Post-hoc balanced Top export ===================
def posthoc_balanced_select(df_all: pd.DataFrame, top_pool: int, top_n: int, alpha: float, lam: float) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return df_all
    pool = df_all.head(min(top_pool, len(df_all))).copy()
    pool = pool.sort_values("GlobalScore", ascending=False).reset_index(drop=True)

    selected_rows = []
    selected_kmers = []

    for _ in range(min(top_n, len(pool))):
        best_i, best_val = None, -1.0
        for i, row in pool.iterrows():
            if row.get("_picked", False):
                continue
            seq = row["Sequence"]
            ks = kmer_set(seq, 3)
            sim = 0.0
            for sk in selected_kmers:
                sim = max(sim, jaccard_sets(ks, sk))
            penal = (1.0 - sim) ** alpha
            base = float(row["GlobalScore"])
            score = base * ((1.0 - lam) + lam * penal)
            if score > best_val:
                best_val = score
                best_i = i
        if best_i is None:
            break
        pool.loc[best_i, "_picked"] = True
        row = pool.loc[best_i]
        selected_rows.append(row)
        selected_kmers.append(kmer_set(row["Sequence"], 3))

    if not selected_rows:
        return pool.head(top_n)

    out = pd.DataFrame(selected_rows).drop(columns=[c for c in ["_picked"] if c in pd.DataFrame(selected_rows).columns], errors="ignore")
    out = out.sort_values("GlobalScore", ascending=False).reset_index(drop=True)
    return out

# =================== Ablation/control plots (Top candidate property distributions) ===================
def summarize_and_plot_ablation(df_top_a: pd.DataFrame, df_top_b: pd.DataFrame,
                                out_dir: str, ts: str, label_a: str, label_b: str):
    if df_top_a is None or df_top_a.empty or df_top_b is None or df_top_b.empty:
        return
    os.makedirs(out_dir, exist_ok=True)

    def _plot_hist(col, title, fname):
        fig, ax = plt.subplots(figsize=(6.4,4.8))
        ax.hist(df_top_a[col].values, bins=20, alpha=0.6, label=label_a, density=True)
        ax.hist(df_top_b[col].values, bins=20, alpha=0.6, label=label_b, density=True)
        ax.set_title(title, fontsize=14, fontweight="bold", color="black", fontname="Arial")
        ax.set_xlabel(col, fontsize=12, fontweight="bold", color="black", fontname="Arial")
        ax.set_ylabel("Density", fontsize=12, fontweight="bold", color="black", fontname="Arial")
        ax.legend(prop={"family":"Arial","weight":"bold"})
        _apply_arial_black_bold(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{ts}_{fname}.png"), dpi=300)
        plt.close(fig)

    for col, title in [
        ("PotencyScore", "Potency (Top set)"),
        ("SafetyComposite", "Safety (Top set)"),
        ("DevelopabilityScore", "Developability (Top set)"),
        ("NoveltyScoreKmer", "Novelty (Top set)"),
        ("Length", "Length (Top set)")
    ]:
        if col in df_top_a.columns and col in df_top_b.columns:
            _plot_hist(col, title, f"Ablation_{col}")

# =================== GA main loop (final balanced version) ===================
def run_lnn4acp_design_ga(
    train_seqs: List[str], y_train: np.ndarray,
    lnn_models_meta: List[tf.keras.Model], base_models_full: Dict[str, Any],
    scaler_meta: StandardScaler, scaler_raw_lnn: StandardScaler, pca_lnn: PCA,
    seq_lnn_model: tf.keras.Model, seq_lnn_max_len: int,
    pos_prop_stats: Dict[str, Dict[str, float]], pos_kmer_sets: List[set],
    out_dir: str, ts: str,
    pop_size:int=None, n_generations:int=None, elite_size:int=None, mut_rate:float=None,
) -> pd.DataFrame:

    pop_size = pop_size or CFG["pop_size"]
    n_generations = n_generations or CFG["n_generations"]
    elite_size = elite_size or CFG["elite_size"]
    mut_rate = mut_rate or CFG["mut_rate"]

    os.makedirs(out_dir, exist_ok=True)
    pos_train = [s for s,y in zip(train_seqs,y_train) if y==1]
    neg_train = [s for s,y in zip(train_seqs,y_train) if y==0]
    train_neg_set = set(neg_train)
    if len(pos_train)==0:
        print("[GA] No positive training sequences.")
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    init_parents = rng.choice(pos_train, size=min(pop_size,len(pos_train)), replace=True)
    population=[]
    for s in init_parents:
        population.append(mutate_seq(s,mut_rate=mut_rate) if rng.random()<0.5 else s)

    archive_scores: Dict[str,Dict[str,Any]] = {}
    first_seen_gen: Dict[str,int] = {}

    def get_scores(pop: List[str], gen_idx:int)->List[Dict[str,Any]]:
        nonlocal archive_scores, first_seen_gen
        clean_pop=[]
        for s in pop:
            try:
                clean_pop.append(_check_seq(s))
            except ValueError:
                continue
        if not clean_pop:
            return []
        new_seqs=[s for s in set(clean_pop) if s not in archive_scores]
        if new_seqs:
            df_new = evaluate_candidates_with_lnn(
                new_seqs, lnn_models_meta, base_models_full, scaler_meta, scaler_raw_lnn, pca_lnn,
                seq_lnn_model, seq_lnn_max_len, pos_prop_stats, pos_kmer_sets, train_neg_set,
                external_safety_csv=CFG["external_safety_csv"]
            )
            for _,row in df_new.iterrows():
                archive_scores[row["Sequence"]] = row.to_dict()
                if row["Sequence"] not in first_seen_gen:
                    first_seen_gen[row["Sequence"]] = gen_idx
        return [archive_scores[s] for s in clean_pop if s in archive_scores]

    for gen in range(n_generations):
        print(f"\n[GA] Generation {gen+1}/{n_generations} — pop={len(population)}")
        scores = get_scores(population, gen)
        if not scores:
            print("[GA] Empty population, stopping.")
            break

        df_gen = pd.DataFrame(scores).sort_values("GlobalScore", ascending=False).reset_index(drop=True)
        print(df_gen[["GlobalScore","PotencyScore","SafetyComposite","DevelopabilityScore",
                      "NoveltyScoreKmer","HemolysisRisk","CytotoxRisk","Length"]].head(5))

        # ===== Diversity parameter annealing =====
        alpha_gen = get_alpha_for_generation(gen, n_generations)
        lam = float(CFG["diversity_lambda"])

        # ===== Elite selection =====
        if CFG["diversity_use"]:
            top_pool = min(int(CFG["diversity_k_pool"]), len(df_gen))
            elite_list = diversify_greedy(df_gen.iloc[:top_pool], elite_size, alpha_gen)
            if len(elite_list) < elite_size:
                fill = [s for s in df_gen["Sequence"].tolist() if s not in elite_list]
                elite_list.extend(fill[:max(0, elite_size - len(elite_list))])
            elites = elite_list[:elite_size]
        else:
            elites = df_gen["Sequence"].head(elite_size).tolist()

        # ===== Parent selection fitness (softened penalty to protect potency density) =====
        if CFG["diversity_use"] and CFG["diversity_use_fitness_penalty"]:
            fitness = compute_soft_diversity_fitness(df_gen, elites, alpha_gen, lam)
        else:
            fitness = np.maximum(df_gen["GlobalScore"].values.astype(np.float32), 1e-6)

        probs = fitness / max(1e-12, fitness.sum())

        # ===== Generate next generation =====
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            parents_idx = rng.choice(len(df_gen), size=2, replace=False, p=probs)
            p1 = df_gen["Sequence"].iloc[parents_idx[0]]
            p2 = df_gen["Sequence"].iloc[parents_idx[1]]
            child = mutate_seq(crossover_seq(p1,p2), mut_rate=mut_rate)
            new_pop.append(child)
        population = new_pop

    # ===== Archive summary =====
    all_rows=[]
    for seq,row in archive_scores.items():
        r=dict(row)
        r["GenerationFirstSeen"]=first_seen_gen.get(seq,None)
        all_rows.append(r)

    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        print("[GA] No candidates generated.")
        return df_all

    df_all = df_all.sort_values("GlobalScore", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(out_dir, f"LNN4ACP_Design_GA_LNNEnhanced_DiversitySafety_Candidates_{ts}.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"[GA] Saved candidates to: {out_csv}")

    df_all.head(20).to_csv(os.path.join(out_dir, f"LNN4ACP_Top20_HighConfidence_{ts}.csv"), index=False)
    df_all.head(50).to_csv(os.path.join(out_dir, f"LNN4ACP_Top50_HighConfidence_{ts}.csv"), index=False)

    try:
        df_strong = df_all[df_all["PotencyScore"] >= 0.80].copy()
        df_strong.to_csv(os.path.join(out_dir, f"LNN4ACP_StrongPotency_Subset_{ts}.csv"), index=False)
    except Exception:
        pass

    if CFG["posthoc_balance"]:
        alpha_final = float(CFG["diversity_alpha_end"]) if CFG["diversity_use"] else 0.0
        balanced = posthoc_balanced_select(
            df_all, top_pool=int(CFG["posthoc_pool"]),
            top_n=int(CFG["posthoc_topN"]),
            alpha=alpha_final, lam=lam
        )
        if balanced is not None and not balanced.empty:
            balanced.to_csv(os.path.join(out_dir, f"LNN4ACP_Top{CFG['posthoc_topN']}_Balanced_{ts}.csv"), index=False)
            balanced.head(20).to_csv(os.path.join(out_dir, f"LNN4ACP_Top20_Balanced_{ts}.csv"), index=False)

    return df_all

# =================== Main pipeline ===================
def main():
    DATA_DIR = "data"
    fp_all = os.path.join(DATA_DIR, "antiCP2.txt")
    if not os.path.exists(fp_all):
        alt_all = os.path.join(os.getcwd(), "antiCP2.txt")
        if os.path.exists(alt_all):
            fp_all = alt_all
    if not os.path.exists(fp_all):
        raise FileNotFoundError(f"File not found in ./data or CWD: {fp_all}")

    all_seq, all_y = read_fasta_pair_lines(fp_all)
    TEST_RATIO=1.0/6.0
    RANDOM_STATE=42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO,
        shuffle=True, stratify=all_y, random_state=RANDOM_STATE
    )
    print("[Loaded] total=%d | TRAIN=%d | TEST=%d" % (len(all_seq), len(tr_seq), len(te_seq)))

    print("[Feature] Extracting TRAIN ...")
    X_train = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test  = extract_features(te_seq, k_max_cksaap=5)
    print("[Shapes] Train=%s, Test=%s" % (X_train.shape, X_test.shape))

    # ===== base learners OOF =====
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds_base = list(skf_base.split(X_train, y_train))
    base_models = get_base_models()
    base_names = ["HistGradientBoosting","LightGBM","RandomForest","XGBoost"]
    models_in_use = {name: base_models[name] for name in base_names}

    OOF_list, TEST_list = [], []
    for name in base_names:
        est = models_in_use[name]
        print(f"\n[Base] {name}: 5-fold OOF & holdout ...")
        oof, te = get_oof_and_holdout_with_folds(X_train, y_train, X_test, est, folds_base)
        OOF_list.append(oof); TEST_list.append(te)

    meta_train = np.vstack(OOF_list).T.astype(np.float32)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)

    scaler_meta = StandardScaler()
    META_TRAIN_S = scaler_meta.fit_transform(meta_train)
    META_TEST_S  = scaler_meta.transform(meta_test)

    scaler_raw_lnn = StandardScaler().fit(X_train)
    X_train_s_lnn = scaler_raw_lnn.transform(X_train)
    X_test_s_lnn  = scaler_raw_lnn.transform(X_test)

    pca_lnn = PCA(n_components=64).fit(X_train_s_lnn)
    X_train_pca64 = pca_lnn.transform(X_train_s_lnn)
    X_test_pca64  = pca_lnn.transform(X_test_s_lnn)

    LNN_TRAIN = np.concatenate([META_TRAIN_S, X_train_pca64], axis=1)
    LNN_TEST  = np.concatenate([META_TEST_S,  X_test_pca64], axis=1)

    # ===== LNN meta-learner (kept unchanged) =====
    lnn_models, oof_train_proba = train_lnn_kfold_meta(
        LNN_TRAIN, y_train,
        n_splits=5, n_repeats=2, random_state=RANDOM_STATE,
        epochs=100, batch_size=32
    )
    p_tr = oof_train_proba
    p_te = predict_ensemble_meta(lnn_models, LNN_TEST)

    thr_best, thr_stats_train = search_best_threshold_multi(
        y_train, p_tr, np.linspace(0.1,0.9,81)
    )

    m_train = {"Set":"Train(OOF)", **metrics_from_proba(y_train, p_tr, thr_best)}
    m_test  = {"Set":"Test(Holdout)", **metrics_from_proba(y_test,  p_te, thr_best)}
    df_meta = pd.DataFrame([m_train, m_test])

    out_dir = "lnn_stack_outputs_optim_repeat"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_roc_prc(y_train, p_tr, "Train", out_dir)
    plot_roc_prc(y_test,  p_te, "Test", out_dir,
                 title_override=("LNN-ACP-Stack-Optim-Repeat-Test ROC",
                                 "LNN-ACP-Stack-Optim-Repeat-Test PRC"))

    yb_train=(p_tr>=thr_best).astype(int)
    yb_test =(p_te>=thr_best).astype(int)
    cm_train=confusion_matrix(y_train,yb_train,labels=[0,1])
    cm_test =confusion_matrix(y_test, yb_test, labels=[0,1])

    plot_confmat(cm_train, "Confusion Matrix — Train (LNN Stack Optim Repeat)",
                 os.path.join(out_dir,f"{ts}_CM_Train_red.png"), "#d32f2f")
    plot_confmat(cm_test,  "Confusion Matrix — Test (LNN Stack Optim Repeat)",
                 os.path.join(out_dir,f"{ts}_CM_Test_deepblue.png"), "#0b3d91")

    # ===== Sequence-level LNN (kept unchanged) =====
    print("\n[Seq-LNN] Training ...")
    seq_lnn_model, seq_lnn_max_len = train_seq_lnn(
        tr_seq, y_train, epochs=80, batch_size=32, random_state=RANDOM_STATE
    )

    # ===== Positive statistics / diversity priors =====
    pos_train_seqs = [s for s,y in zip(tr_seq,y_train) if y==1]
    pos_prop_stats = compute_positive_props_stats(pos_train_seqs)
    pos_kmer_sets  = build_pos_kmer_sets(pos_train_seqs, k=3)

    # ===== Fit base models on full training set (only to generate inputs for LNN_meta) =====
    base_models_full={}
    for name in base_names:
        print(f"[Base-Full] Fitting {name} on full training set for design stage ...")
        base_models_full[name] = fit_base_model_full(base_models[name], X_train, y_train)

    # ===== Run GA (final balanced strategy) =====
    print("\n[GA] Running LNN4ACP design (balanced diversity vs potency) ...")
    df_candidates_div = run_lnn4acp_design_ga(
        tr_seq, y_train,
        lnn_models, base_models_full,
        scaler_meta, scaler_raw_lnn, pca_lnn,
        seq_lnn_model, seq_lnn_max_len,
        pos_prop_stats, pos_kmer_sets,
        out_dir, ts
    )

    # ===== Ablation: control without diversity penalty =====
    df_candidates_no_div = None
    if CFG["run_ablation"]:
        old_use = CFG["diversity_use"]
        old_fit = CFG["diversity_use_fitness_penalty"]
        CFG["diversity_use"] = False
        CFG["diversity_use_fitness_penalty"] = False

        df_candidates_no_div = run_lnn4acp_design_ga(
            tr_seq, y_train,
            lnn_models, base_models_full,
            scaler_meta, scaler_raw_lnn, pca_lnn,
            seq_lnn_model, seq_lnn_max_len,
            pos_prop_stats, pos_kmer_sets,
            out_dir, ts+"_NoDiv"
        )

        CFG["diversity_use"] = old_use
        CFG["diversity_use_fitness_penalty"] = old_fit

    # ===== Save metrics and intermediate results =====
    out_xlsx = os.path.join(out_dir, f"LNN_ACP_Stack_AAC_CKSAAP_PCP16_Optim_Repeat_{ts}.xlsx")
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_meta.to_excel(w, sheet_name="LNN_Train_vs_Test", index=False)
            pd.DataFrame(meta_train, columns=base_names).to_excel(w, sheet_name="MetaTrain_OOF", index=False)
            pd.DataFrame(meta_test,  columns=base_names).to_excel(w, sheet_name="MetaTest_Preds", index=False)
            thr_info = {"Best_Threshold_from_Train_OOF": thr_best, **thr_stats_train}
            pd.DataFrame([thr_info]).to_excel(w, sheet_name="LNN_Threshold", index=False)

            if df_candidates_div is not None and not df_candidates_div.empty:
                df_candidates_div.head(2000).to_excel(w, sheet_name="Design_Top_withDivBalanced", index=False)
            if df_candidates_no_div is not None and not df_candidates_no_div.empty:
                df_candidates_no_div.head(2000).to_excel(w, sheet_name="Design_Top_noDiv", index=False)

        print("\nSaved to:", out_xlsx)
    except Exception as e:
        print("[Warn] openpyxl not available, write CSV instead.", e)
        df_meta.to_csv(out_xlsx.replace(".xlsx","_meta.csv"), index=False)

    # ===== Control/ablation plot: compare Top50 distributions =====
    if CFG["run_ablation"] and \
       df_candidates_div is not None and not df_candidates_div.empty and \
       df_candidates_no_div is not None and not df_candidates_no_div.empty:
        summarize_and_plot_ablation(
            df_candidates_div.head(50),
            df_candidates_no_div.head(50),
            out_dir, ts,
            "WithDivBalanced", "NoDivPenalty"
        )

    print("\n=== LNN4ACP — Prediction intact; Generation judged purely by LNN ===")
    print(df_meta)

    if df_candidates_div is not None and not df_candidates_div.empty:
        print("\n[GA] Top 10 (With balanced diversity, after safety gating):")
        print(df_candidates_div.loc[:9, ["Sequence","GlobalScore","PotencyScore","SafetyComposite",
                                         "DevelopabilityScore","NoveltyScoreKmer",
                                         "HemolysisRisk","CytotoxRisk","Length"]])

    del lnn_models
    gc.collect()

if __name__ == "__main__":
    np.random.seed(42)
    main()
