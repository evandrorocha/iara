"""
Plots the averaged confusion matrix (across all 10 folds) for the Golden MEL model.
Saves the figure to scratch/confusion_matrix_golden_mel.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

BASE = Path("results/trainings/tests/svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0/eval")
CLASS_NAMES = ["BACKGROUND", "SMALL", "MEDIUM", "LARGE"]
N = len(CLASS_NAMES)

cm_sum = np.zeros((N, N), dtype=float)
fold_count = 0

for fold_i in range(10):
    csv_path = BASE / f"fold_{fold_i}" / "svm_nystroem_4000_log_melgram_elasticnet_l1r0.15_C2.0_multiclass_test.csv"
    if not csv_path.exists():
        print(f"fold_{fold_i}: not found, skipping")
        continue

    df = pd.read_csv(csv_path)
    targets     = df["Target"].values.astype(int)
    predictions = df["Prediction"].values.astype(int)

    # Build confusion matrix for this fold (row=true, col=pred)
    cm = np.zeros((N, N), dtype=float)
    for t, p in zip(targets, predictions):
        cm[t, p] += 1

    # Normalize by row (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100
    cm_sum += cm_norm
    fold_count += 1

cm_avg = cm_sum / fold_count

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

im = ax.imshow(cm_avg, cmap="Blues", vmin=0, vmax=100)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color="white")
cbar.outline.set_edgecolor("white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=10)
cbar.set_label("Recall médio por fold (%)", color="white", fontsize=10)

# Annotate cells
for i in range(N):
    for j in range(N):
        val = cm_avg[i, j]
        color = "white" if val < 50 else "#0f1117"
        weight = "bold" if i == j else "normal"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=13, color=color, fontweight=weight)

ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

ax.set_xlabel("Classe Predita", color="white", fontsize=12, labelpad=10)
ax.set_ylabel("Classe Real", color="white", fontsize=12, labelpad=10)
ax.set_title("Matriz de Confusão — Golden MEL (SVM m=4000, C=2.0, ElasticNet)\nMédia 10 folds — Conjunto de Teste (by_window)",
             color="white", fontsize=11, pad=14)

plt.tight_layout()
out = Path("scratch/confusion_matrix_golden_mel.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")

# Print per-class recall summary
print("\nRecall médio por classe:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls}: {cm_avg[i, i]:.1f}%")
