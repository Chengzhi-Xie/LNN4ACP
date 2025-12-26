# -*- coding: utf-8 -*-
"""
Mirror bar chart (Positive right / Negative left) for ACP length distributions.
Only reads the current directory file:
  - antiCP2.txt  (FASTA-like: header on odd lines, sequence on even lines)
Bins: <15 / 15–25 / 25–35 / 35–45 / >45 AA (kept range: 5..55 AA)
Output:
  - antiCP2_len_mirror.(png|tiff)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ===================== Binning and range =====================
KEEP_LEN_MIN = 5
KEEP_LEN_MAX = 55
# Bin edges correspond to: <15, [15,25), [25,35), [35,45), >45 (up to 55)
BIN_EDGES  = [5, 15, 25, 35, 45, 56]   # use 56 to cover up to 55
BIN_LABELS = ["<15AA", "15-25AA", "25-35AA", "35-45AA", ">45AA"]

COLOR_POS = 'salmon'     # Positive color (to the right)
COLOR_NEG = 'lightblue'  # Negative color (to the left)

# ===================== Read and parse =====================
def parse_label_seq_pairs(filepath, keep_len_min=5, keep_len_max=55):
    """
    Read a fasta-like text (odd lines are headers, even lines are sequences),
    keeping only sequences whose length falls within the specified range.
    Headers containing 'positive' are grouped as positives; 'negative' as negatives; others are ignored.
    """
    pos_lengths, neg_lengths = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    # Read in pairs: header -> sequence
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        header = lines[i].lstrip(">").lower()
        seq = lines[i + 1].replace(" ", "")
        L = len(seq)
        if not (keep_len_min <= L <= keep_len_max):
            continue
        if "positive" in header:
            pos_lengths.append(L)
        elif "negative" in header:
            neg_lengths.append(L)
        # ignore other labels
    return pos_lengths, neg_lengths

# ===================== Plot (mirrored horizontal bars) =====================
def plot_mirror(pos_lengths, neg_lengths, title, out_prefix):
    # Count frequencies for each bin
    pos_counts, _ = np.histogram(pos_lengths, bins=BIN_EDGES)
    neg_counts, _ = np.histogram(neg_lengths, bins=BIN_EDGES)
    neg_counts_signed = -neg_counts  # negative class to the left

    y = np.arange(len(BIN_LABELS))

    plt.figure(figsize=(10, 6.4))
    # Negative samples (to the left)
    plt.barh(y, neg_counts_signed, color=COLOR_NEG, edgecolor='black', label='Negative')
    # Positive samples (to the right)
    plt.barh(y, pos_counts,        color=COLOR_POS, edgecolor='black', label='Positive')

    # Annotate counts on the bars
    for yi, cnt in enumerate(pos_counts):
        if cnt > 0:
            plt.text(cnt + max(2, cnt*0.01), yi, f"{cnt}",
                     va='center', ha='left', fontsize=9, fontweight='bold')
    for yi, cnt in enumerate(neg_counts_signed):
        if cnt < 0:
            plt.text(cnt - max(2, abs(cnt)*0.01), yi, f"{abs(cnt)}",
                     va='center', ha='right', fontsize=9, fontweight='bold')

    # Central zero axis
    plt.axvline(0, color='black', linewidth=1)

    # Axes and title
    plt.xlabel("Number of Peptides", fontsize=12, fontweight='bold')
    plt.yticks(y, BIN_LABELS, fontsize=10, fontweight='bold')
    plt.title(f"{title}", fontsize=14, fontweight='bold')

    # Symmetric x-limits
    max_side = max(pos_counts.max()+20 if pos_counts.size else 0,
                   neg_counts.max() if neg_counts.size else 0)
    pad = max(10, int(max_side * 0.05))
    plt.xlim(-max_side - pad, max_side + pad)

    plt.legend(prop={'weight': 'bold'}, edgecolor='black', loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.tiff", dpi=600, bbox_inches="tight")
    plt.show()

# ===================== Main =====================
def main():
    fp = "antiCP2.txt"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found in current directory: {fp}")
    pos_len, neg_len = parse_label_seq_pairs(fp, KEEP_LEN_MIN, KEEP_LEN_MAX)
    print(f"[antiCP2] kept within [{KEEP_LEN_MIN}, {KEEP_LEN_MAX}] — Pos {len(pos_len)} / Neg {len(neg_len)}")
    plot_mirror(
        pos_len, neg_len,
        title="ACP Sequence Length Distribution (antiCP2.txt)",
        out_prefix="antiCP2_len_mirror"
    )

if __name__ == "__main__":
    main()
