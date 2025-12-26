#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust PAMP/ACP Classification with ESM2
✔ Batch inference
✔ Illegal amino acid cleaning
✔ GPU/CPU both supported
"""

import os
import torch
import numpy as np
import pandas as pd
import esm
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# -------------------- Config --------------------
SEQ_LEN = 256
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 4  # safe for <12GB GPU
PAMP_PATH = "./ACP20.txt"   # ✅ set to current path
SAVE_PATH = "./esm2_output"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# ✅ Optimize GPU (if available)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# -------------------- Valid Amino Acids --------------------
VALID_AA = "ACDEFGHIKLMNPQRSTVWY"

def clean_sequence(seq: str) -> str:
    """Keep only valid amino acids and uppercase."""
    return ''.join([aa for aa in seq.upper() if aa in VALID_AA])

# ============== Data reading (FASTA two-line: odd header, even sequence) ==============
def read_fasta_pair_lines(file_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Read two-line FASTA: odd lines are headers (if contains 'positive' -> label 1, otherwise 0), even lines are sequences.
    Returns: list of sequences (only valid amino acids, truncated to SEQ_LEN) and label array (np.int32).
    """
    sequences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"[ERROR] Invalid FASTA format: number of lines ({len(lines)}) is not even. Please check {file_path}")

    for i in range(0, len(lines), 2):
        header = lines[i]
        raw_seq = lines[i + 1].replace(" ", "")
        label = 1 if "positive" in header.lower() else 0
        seq = clean_sequence(raw_seq)[:SEQ_LEN]
        if seq:  # keep only non-empty sequences after cleaning
            sequences.append(seq)
            labels.append(label)

    return sequences, np.array(labels, dtype=np.int32)

# -------------------- Data Loader --------------------
def load_data():
    seqs, labels = read_fasta_pair_lines(PAMP_PATH)
    if len(seqs) == 0:
        raise ValueError("[ERROR] No valid sequences were read; please check the contents and format of antiCP2.txt.")

    df = pd.DataFrame({'sequence': seqs, 'label': labels})
    # Stratified split by label: 5:1 train:test
    train_df, test_df = train_test_split(
        df, test_size=1/6, stratify=df.label, random_state=SEED
    )
    print(f"[DATA] Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)} "
          f"(Pos rate total/train/test: "
          f"{df.label.mean():.3f}/{train_df.label.mean():.3f}/{test_df.label.mean():.3f})")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# -------------------- ESM2 Embedding with Batching --------------------
def get_esm2_embeddings(seqs, model, alphabet, batch_size=4):
    model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    all_embeddings = []

    for i in range(0, len(seqs), batch_size):
        batch = [(f"seq{i+j}", s) for j, s in enumerate(seqs[i:i+batch_size])]
        _, _, toks = batch_converter(batch)
        toks = toks.to(DEVICE)

        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        token_reps = out["representations"][33]
        # Average over token dimension (including special tokens); to exclude <cls>/<eos> use token_reps[:, 1:-1, :].mean(dim=1)
        emb = token_reps.mean(dim=1).detach().cpu().numpy()
        all_embeddings.append(emb)

        del toks, out, token_reps
        torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)

# -------------------- Cross Validation --------------------
def cross_val_evaluate(train_df, model, alphabet):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    metrics = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"[INFO] Fold {fold} running...")

        X_tr = train_df.sequence.iloc[tr_idx].tolist()
        y_tr = train_df.label.iloc[tr_idx].tolist()
        X_vl = train_df.sequence.iloc[vl_idx].tolist()
        y_vl = train_df.label.iloc[vl_idx].tolist()

        X_tr_emb = get_esm2_embeddings(X_tr, model, alphabet, batch_size=BATCH_SIZE)
        X_vl_emb = get_esm2_embeddings(X_vl, model, alphabet, batch_size=BATCH_SIZE)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr_emb, y_tr)
        prob = clf.predict_proba(X_vl_emb)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics.append({
            'Fold': fold,
            'AUC': roc_auc_score(y_vl, prob),
            'ACC': accuracy_score(y_vl, pred),
            'Precision': precision_score(y_vl, pred, zero_division=0),
            'Recall': recall_score(y_vl, pred, zero_division=0),
            'F1': f1_score(y_vl, pred, zero_division=0),
            'MCC': matthews_corrcoef(y_vl, pred)
        })

    df = pd.DataFrame(metrics)
    df.to_csv(f"{SAVE_PATH}/CV_metrics.csv", index=False)
    print("\n[✔] CV Results saved to CV_metrics.csv")
    print(df.mean(numeric_only=True).round(4))
    return df

# -------------------- Final Test Evaluation --------------------
def final_test_evaluation(train_df, test_df, model, alphabet):
    X_tr = train_df.sequence.tolist()
    y_tr = train_df.label.tolist()
    X_te = test_df.sequence.tolist()
    y_te = test_df.label.tolist()

    X_tr_emb = get_esm2_embeddings(X_tr, model, alphabet, batch_size=BATCH_SIZE)
    X_te_emb = get_esm2_embeddings(X_te, model, alphabet, batch_size=BATCH_SIZE)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr_emb, y_tr)
    prob = clf.predict_proba(X_te_emb)[:, 1]
    pred = (prob >= 0.5).astype(int)

    sorted_df = pd.DataFrame({
        'sequence': X_te,
        'true_label': y_te,
        'pred_prob': prob,
        'pred_label': pred
    }).sort_values(by="pred_prob", ascending=False)
    sorted_df.to_csv(f"{SAVE_PATH}/sorted_predictions.csv", index=False)

    metrics = {
        'AUC': roc_auc_score(y_te, prob),
        'ACC': accuracy_score(y_te, pred),
        'Precision': precision_score(y_te, pred, zero_division=0),
        'Recall': recall_score(y_te, pred, zero_division=0),
        'F1': f1_score(y_te, pred, zero_division=0),
        'MCC': matthews_corrcoef(y_te, pred)
    }
    pd.DataFrame([metrics]).to_csv(f"{SAVE_PATH}/final_test_metrics.csv", index=False)

    print("\n[✔] Final test metrics saved to final_test_metrics.csv")
    print(metrics)

# -------------------- Main --------------------
def run_pipeline():
    print("[INFO] Loading ESM2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    train_df, test_df = load_data()

    print("[INFO] Running 5-Fold Cross-Validation...")
    cross_val_evaluate(train_df, model, alphabet)

    print("[INFO] Running Final Test Evaluation...")
    final_test_evaluation(train_df, test_df, model, alphabet)

if __name__ == "__main__":
    run_pipeline()
