#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast ProtT5 CV Pipeline (CPU/GPU adaptive)
- FASTA two-line reading: odd lines are headers, even lines are sequences; header containing 'positive' -> 1, otherwise 0
- Keep only 20 standard amino acids; ProtT5 expects space-separated inputs like 'A C D ...'
- Precompute all sequence embeddings once and reuse across CV folds (significant speedup)
- On CPU, apply dynamic quantization to Linear layers (int8) for further acceleration
- Length bucketing + batching to reduce padding waste
- Safe loading: prefer safetensors (enforced for torch<2.6), allow .bin when torch>=2.6
"""

import os, sys, math, hashlib
import torch
import numpy as np
import pandas as pd
from packaging import version
from typing import List, Tuple, Dict

# Configurable parameters
SEQ_LEN = 256
SEED = 42
N_SPLITS = 5
MODEL_ID = "Rostlab/prot_t5_xl_uniref50"   # For faster inference you can switch to: "Rostlab/prot_bert"
PAMP_PATH = "./ACP20.txt"
SAVE_PATH = "./prott5_output"
os.makedirs(SAVE_PATH, exist_ok=True)

# Device/threads/batch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH = 4 if torch.cuda.is_available() else 2
BATCH_SIZE = int(os.environ.get("PROTT5_BATCH", DEFAULT_BATCH))
THREADS = int(os.environ.get("PROTT5_THREADS", max(1, (os.cpu_count() or 4)//1)))
torch.set_num_threads(THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print(f"[ENV] device={DEVICE}, batch={BATCH_SIZE}, threads={THREADS}")

# ====== Dependency self-check (protobuf / sentencepiece) ======
def ensure_deps():
    try:
        import sentencepiece  # noqa
    except Exception as e:
        raise SystemExit("Missing sentencepiece. Please install: pip install -U 'sentencepiece>=0.1.99'") from e
    try:
        import google.protobuf  # noqa
    except Exception as e:
        raise SystemExit(
            "google.protobuf not found (common cause: accidentally installed the 'google' package).\n"
            "Fix: pip uninstall -y google && pip install -U 'protobuf>=4.25.0'"
        ) from e
ensure_deps()

# ====== HF model loading (safe + general) ======
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def is_t5_model(model_id: str) -> bool:
    return "prot_t5" in model_id.lower()

def load_model_and_tokenizer(model_id: str):
    torch_ver = version.parse(torch.__version__.split("+")[0])
    print(f"[INFO] torch={torch.__version__}")
    if is_t5_model(model_id):
        tokenizer = T5Tokenizer.from_pretrained(model_id, do_lower_case=False, legacy=False)
        # For torch<2.6 enforce safetensors
        if torch_ver < version.parse("2.6"):
            print("[SAFE] torch<2.6: enforce safetensors")
            model = T5EncoderModel.from_pretrained(model_id, use_safetensors=True, low_cpu_mem_usage=True)
        else:
            try:
                print("[SAFE] Prefer safetensors")
                model = T5EncoderModel.from_pretrained(model_id, use_safetensors=True, low_cpu_mem_usage=True)
            except Exception:
                print("[WARN] No safetensors available, falling back to .bin (safe for torch>=2.6)")
                model = T5EncoderModel.from_pretrained(model_id, use_safetensors=False, low_cpu_mem_usage=True)
    else:
        # Generic branch for ProtBERT etc. (faster)
        tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=False)
        if torch_ver < version.parse("2.6"):
            print("[SAFE] torch<2.6: prefer safetensors")
            try:
                model = AutoModel.from_pretrained(model_id, use_safetensors=True, low_cpu_mem_usage=True)
            except Exception as e:
                raise SystemExit("This model has no safetensors, and torch<2.6 cannot safely load .bin. Please upgrade to torch>=2.6.") from e
        else:
            try:
                model = AutoModel.from_pretrained(model_id, use_safetensors=True, low_cpu_mem_usage=True)
            except Exception:
                model = AutoModel.from_pretrained(model_id, use_safetensors=False, low_cpu_mem_usage=True)

    # CPU dynamic quantization (Linear layers only) for noticeable CPU inference speedup
    if DEVICE.type == "cpu":
        try:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print("[OPT] CPU dynamic quantization enabled (Linear->int8)")
        except Exception as e:
            print("[WARN] Dynamic quantization failed:", e)

    model.to(DEVICE)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return tokenizer, model

# ====== Data reading / cleaning ======
VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
def clean_sequence(seq: str) -> str:
    return ' '.join([aa for aa in seq.upper() if aa in VALID_AA])

def read_fasta_pair_lines(file_path: str) -> Tuple[List[str], np.ndarray]:
    seqs, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"FASTA line count ({len(lines)}) is not even: {file_path}")

    for i in range(0, len(lines), 2):
        header = lines[i]
        raw = lines[i+1].strip().upper().replace(" ", "")[:SEQ_LEN]
        lab = 1 if "positive" in header.lower() else 0
        spaced = clean_sequence(raw)
        if spaced:
            seqs.append(spaced)
            labels.append(lab)
    return seqs, np.array(labels, dtype=np.int32)

def load_data():
    seqs, labels = read_fasta_pair_lines(PAMP_PATH)
    if not seqs:
        raise ValueError("No valid sequences were read; please check antiCP2.txt.")
    df = pd.DataFrame({"sequence": seqs, "label": labels})
    tr, te = train_test_split(df, test_size=1/6, stratify=df.label, random_state=SEED)
    print(f"[DATA] total={len(df)} train={len(tr)} test={len(te)} | pos-rate T/{df.label.mean():.3f} "
          f"tr/{tr.label.mean():.3f} te/{te.label.mean():.3f}")
    return tr.reset_index(drop=True), te.reset_index(drop=True)

# ====== Efficient embedding: length bucketing + one-shot precompute ======
def seq_len_from_spaced(spaced_seq: str) -> int:
    # 'A C D' -> 3
    return spaced_seq.count(' ') + 1

def batched_embeddings(seqs: List[str], tokenizer, model, batch_size: int) -> np.ndarray:
    model.eval()
    embs = []
    total = len(seqs)
    for i in range(0, total, batch_size):
        batch = seqs[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
            last = out.last_hidden_state  # (B, L, D)
            mean_emb = last.mean(dim=1).detach().cpu().numpy()
        embs.append(mean_emb)
        # Lightweight progress indicator
        if (i // batch_size) % 20 == 0 or (i + batch_size) >= total:
            done = min(i+batch_size, total)
            print(f"  [embed] {done}/{total}")
        del tokens, out, last, mean_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(embs, axis=0)

def compute_all_embeddings_in_order(spaced_seqs: List[str], tokenizer, model, batch_size: int) -> np.ndarray:
    """
    Return embeddings in the original input order; internally bucket/sort by length to reduce padding, then restore order.
    """
    # Record original indices and lengths
    items = [(idx, s, seq_len_from_spaced(s)) for idx, s in enumerate(spaced_seqs)]
    # Simple bucketing: sort by length, then take adjacent sequences per batch
    items.sort(key=lambda x: x[2])
    sorted_seqs = [s for _, s, _ in items]

    sorted_embs = batched_embeddings(sorted_seqs, tokenizer, model, batch_size)
    # Restore to original order
    embs = np.zeros_like(sorted_embs)
    for sorted_pos, (orig_idx, _, _) in enumerate(items):
        embs[orig_idx] = sorted_embs[sorted_pos]
    return embs

# ====== Main flow: one-shot precompute -> CV by slicing ======
def run_cv_fast():
    tokenizer, model = load_model_and_tokenizer(MODEL_ID)
    from sklearn.model_selection import StratifiedKFold
    train_df, _ = load_data()

    # 1) Precompute all embeddings for train_df (big speedup)
    print("[STEP] Precomputing embeddings for all training sequences...")
    all_seqs = train_df.sequence.tolist()
    all_embs = compute_all_embeddings_in_order(all_seqs, tokenizer, model, BATCH_SIZE)
    np.save(os.path.join(SAVE_PATH, "train_all_embs.npy"), all_embs)
    print("[OK] Cached to train_all_embs.npy")

    # 2) CV: slice directly
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    metrics = []
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df.label), 1):
        print(f"[Fold {fold}]")
        y_tr = train_df.label.iloc[tr_idx].to_numpy()
        y_vl = train_df.label.iloc[vl_idx].to_numpy()
        X_tr_emb = all_embs[tr_idx]
        X_vl_emb = all_embs[vl_idx]

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr_emb, y_tr)
        prob = clf.predict_proba(X_vl_emb)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics.append({
            "Fold": fold,
            "AUC": roc_auc_score(y_vl, prob),
            "ACC": accuracy_score(y_vl, pred),
            "Precision": precision_score(y_vl, pred, zero_division=0),
            "Recall": recall_score(y_vl, pred, zero_division=0),
            "F1": f1_score(y_vl, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_vl, pred),
        })

    df = pd.DataFrame(metrics)
    out_csv = os.path.join(SAVE_PATH, "ProtT5_cv_fast_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✔ Results saved to {out_csv}")
    print(df.mean(numeric_only=True).round(4))

if __name__ == "__main__":
    try:
        run_cv_fast()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", file=sys.stderr)
        sys.exit(130)
