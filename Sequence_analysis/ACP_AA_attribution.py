#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib import transforms
import matplotlib.colors as mcolors

# -----------------------------
# Global configuration: English font
# -----------------------------
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 20 amino acids and their colors
# -----------------------------
amino_acid_colors = {
    'A':'#e6194b','C':'#3cb44b','D':'#ffe119','E':'#0082c8',
    'F':'#f58231','G':'#911eb4','H':'#46f0f0','I':'#d2f53c',
    'K':'#f032e6','L':'#fabebe','M':'#008080','N':'#e6beff',
    'P':'#aa6e28','Q':'#fffac8','R':'#800000','S':'#aaffc3',
    'T':'#808000','V':'#ffd8b1','W':'#808080','Y':'#000080'
}
standard_alphabet = list(amino_acid_colors.keys())

# -----------------------------
# Read labels and sequences from a single ACP file
# File format: odd line is label (>ACP_positive_* / >ACP_negative_*),
#              even line below is the sequence.
# -----------------------------
def read_acp_single_file(fp="antiCP2.txt"):
    if not os.path.exists(fp):
        raise FileNotFoundError(f"[ERR] File not found: {fp} (expect it in current directory)")
    labels, seqs = [], []
    with open(fp, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        labline = lines[i].lstrip(">").lower()
        seq = lines[i + 1].strip().upper().replace(" ", "")
        if "positive" in labline:
            labels.append(1)   # ACP
            seqs.append(seq)
        elif "negative" in labline:
            labels.append(0)   # non-ACP
            seqs.append(seq)
        # other headers (if any) are ignored
    if len(labels) == 0:
        raise ValueError("[ERR] No sequences parsed from antiCP2.txt. "
                         "Check that headers contain 'positive'/'negative' and lines are paired.")
    return labels, seqs

labels, seqs = read_acp_single_file("antiCP2.txt")
acp_seqs     = [s for l, s in zip(labels, seqs) if l == 1]
non_acp_seqs = [s for l, s in zip(labels, seqs) if l == 0]

print(f"[INFO] Loaded sequences from antiCP2.txt: ACP={len(acp_seqs)}  non-ACP={len(non_acp_seqs)}  total={len(seqs)}")

# -----------------------------
# Compute positional residue frequency matrix
# -----------------------------
def positional_freq_matrix(seq_list, alphabet):
    if len(seq_list) == 0:
        return np.zeros((len(alphabet), 1), dtype=float), 1
    max_len = max(len(s) for s in seq_list)
    mat = np.zeros((len(alphabet), max_len), dtype=float)
    for seq in seq_list:
        for i, aa in enumerate(seq):
            if aa in alphabet:
                j = alphabet.index(aa)
                mat[j, i] += 1
    # normalize to frequencies
    mat /= max(1, len(seq_list))
    return mat, max_len

acp_mat, L = positional_freq_matrix(acp_seqs, standard_alphabet)
nonacp_mat, _ = positional_freq_matrix(non_acp_seqs, standard_alphabet)

# -----------------------------
# Custom gradient color maps
# -----------------------------
cmap_positive = mcolors.LinearSegmentedColormap.from_list('pos', ['#FFFFFF','#D13D5D'])
cmap_negative = mcolors.LinearSegmentedColormap.from_list('neg', ['#FFFFFF','#3A7FBF'])

# -----------------------------
# Plot positional residue frequency heatmap
# -----------------------------
def plot_position_heatmap(matrix, alphabet, title, cmap):
    plt.figure(figsize=(12, 6))
    im = plt.imshow(matrix, aspect='auto', interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Amino Acid', fontsize=14, fontweight='bold')
    plt.xlabel('Sequence Position', fontsize=14, fontweight='bold')
    plt.yticks(range(len(alphabet)), alphabet, fontsize=12)
    cbar = plt.colorbar(im, label='Frequency')
    plt.setp(plt.gca().get_xticklabels(), fontweight='bold', color='black')
    plt.setp(plt.gca().get_yticklabels(), fontweight='bold', color='black')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold', color='black')
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_color('black')
    cbar.ax.yaxis.label.set_fontsize(14)
    plt.tight_layout()

# (1) ACP frequency heatmap
plot_position_heatmap(acp_mat, standard_alphabet,
                      'ACP: Positional Amino Acid Frequency (antiCP2.txt)',
                      cmap_positive)
plt.savefig('acp_freq.png', dpi=600, format='png')
plt.savefig('acp_freq.tiff', dpi=600, format='tiff')
plt.show()

# (2) non-ACP frequency heatmap
plot_position_heatmap(nonacp_mat, standard_alphabet,
                      'non-ACP: Positional Amino Acid Frequency (antiCP2.txt)',
                      cmap_negative)
plt.savefig('nonacp_freq.png', dpi=600, format='png')
plt.savefig('nonacp_freq.tiff', dpi=600, format='tiff')
plt.show()

# -----------------------------
# Motif-like logo plot
# -----------------------------
def plot_amino_acid_logo(count_dict, order, title,
                         color_map=amino_acid_colors, gap=0.1):
    if len(order) == 0:
        order = standard_alphabet[:]  # fallback to standard order
    max_count = max(count_dict.values()) if count_dict else 1e-9
    fig, ax = plt.subplots(figsize=(max(8, len(order)), 4))
    x_pos = 0
    for aa in order:
        cnt = count_dict.get(aa, 0)
        height = cnt / max_count * 5  # normalized height
        tp = TextPath((0,0), aa, size=1,
                      prop=matplotlib.font_manager.FontProperties(family='Arial'))
        bb = tp.get_extents()
        sx = 1.0 / (bb.width if bb.width != 0 else 1.0)
        sy = height / (bb.height if bb.height != 0 else 1.0)
        path = transforms.Affine2D().scale(sx, sy).translate(x_pos, 0).transform_path(tp)
        ax.add_patch(PathPatch(path, color=color_map.get(aa,'black'), lw=0))
        x_pos += 1 + gap
    ax.set_xlim(-0.5, x_pos-gap+0.5)
    ax.set_ylim(0,6)
    ax.set_xticks(np.arange(0, len(order)*(1+gap), (1+gap)))
    ax.set_xticklabels(order, rotation=90)
    plt.setp(ax.get_xticklabels(), fontweight='bold', color='black')
    plt.setp(ax.get_yticklabels(), fontweight='bold', color='black')
    ax.set_ylabel('Normalized Count', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

# -----------------------------
# Compute amino acid counts
# -----------------------------
def compute_amino_acid_count(seq_list, order):
    return {aa: sum(seq.count(aa) for seq in seq_list) for aa in order}

order_acp = list(dict.fromkeys([ch for seq in acp_seqs for ch in seq if ch in standard_alphabet]))
order_non_acp = list(dict.fromkeys([ch for seq in non_acp_seqs for ch in seq if ch in standard_alphabet]))
global_order = list(dict.fromkeys([ch for seq in seqs for ch in seq if ch in standard_alphabet]))

# (3) ACP Motif-like Logo
plot_amino_acid_logo(compute_amino_acid_count(acp_seqs, order_acp),
                     order_acp, 'ACP Motif-like Amino Acid Logo (antiCP2.txt)')
plt.savefig('acp_logo.png', dpi=600, format='png')
plt.savefig('acp_logo.tiff', dpi=600, format='tiff')
plt.show()

# (4) non-ACP Motif-like Logo
plot_amino_acid_logo(compute_amino_acid_count(non_acp_seqs, order_non_acp),
                     order_non_acp, 'non-ACP Motif-like Amino Acid Logo (antiCP2.txt)')
plt.savefig('nonacp_logo.png', dpi=600, format='png')
plt.savefig('nonacp_logo.tiff', dpi=600, format='tiff')
plt.show()

# (5) Count comparison bar chart
plt.figure(figsize=(max(8, len(global_order)*0.4), 4))
x = np.arange(len(global_order))
counts_acp = [compute_amino_acid_count(acp_seqs, global_order).get(aa, 0) for aa in global_order]
counts_non = [compute_amino_acid_count(non_acp_seqs, global_order).get(aa, 0) for aa in global_order]
width = 0.4
plt.bar(x-0.2, counts_acp, width, color='#EA6C84', label='ACP', alpha=0.8)
plt.bar(x+0.2, counts_non, width, color='#5AA3CB', label='non-ACP', alpha=0.8)
plt.xticks(x, global_order, rotation=90, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('Amino Acid', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.title('Amino Acid Count Comparison (ACP vs non-ACP, antiCP2.txt)', fontsize=16, fontweight='bold')
plt.legend(prop={'weight':'bold'}, edgecolor='black')
plt.tight_layout()
plt.savefig('count_comparison_acp.png', dpi=600, format='png')
plt.savefig('count_comparison_acp.tiff', dpi=600, format='tiff')
plt.show()
