"""Aggregate run JSONs into a results table + figures for the report."""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import CLASSES, RESULTS_DIR, ROOT, load_cifar10, simclr_view

FIG_DIR = os.path.join(ROOT, "report", "figures")
LABELS = {
    ("scratch", "finetune"): "Scratch (baseline)",
    ("imagenet_random", "finetune"): "Scratch, ImageNet arch @128px",
    ("simclr", "linear"): "SimCLR + linear probe",
    ("simclr", "finetune"): "SimCLR + fine-tune",
    ("imagenet", "linear"): "ImageNet + linear probe",
    ("imagenet", "finetune"): "ImageNet + fine-tune",
    ("scratch", "linear"): "Random init + linear probe",
}
ORDER = [("scratch", "finetune"), ("imagenet_random", "finetune"), ("scratch", "linear"),
         ("simclr", "linear"), ("simclr", "finetune"),
         ("imagenet", "linear"), ("imagenet", "finetune")]
FRACS = [0.01, 0.1, 1.0]


def load_runs():
    runs = defaultdict(list)
    for p in glob.glob(os.path.join(RESULTS_DIR, "runs", "*.json")):
        r = json.load(open(p))
        runs[(r["init"], r["protocol"], r["fraction"])].append(r)
    return runs


def summarize(runs):
    rows = []
    for key in ORDER:
        row = {"method": LABELS[key]}
        for f in FRACS:
            rs = runs.get((*key, f), [])
            if rs:
                acc = np.array([r["accuracy"] for r in rs]) * 100
                f1 = np.array([r["macro_f1"] for r in rs]) * 100
                row[f] = (acc.mean(), acc.std(), f1.mean(), f1.std(), len(rs))
        if len(row) > 1:
            rows.append(row)
    return rows


def write_tables(rows):
    lines = ["| Method | " + " | ".join(f"{int(f*100)}% labels Acc / F1" for f in FRACS) + " |",
             "|---|" + "---|" * len(FRACS)]
    tex = []
    for row in rows:
        cells, tcells = [], []
        for f in FRACS:
            if f in row:
                a, sa, f1, sf, n = row[f]
                pm = f" ± {sa:.1f}" if n > 1 else ""
                cells.append(f"{a:.2f}{pm} / {f1:.2f}")
                tpm = f"$\\pm${sa:.1f}" if n > 1 else ""
                tcells.append(f"{a:.2f}{tpm} & {f1:.2f}")
            else:
                cells.append("–")
                tcells.append("-- & --")
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")
        tex.append(f"{row['method']} & " + " & ".join(tcells) + r" \\")
    md = "\n".join(lines)
    open(os.path.join(RESULTS_DIR, "results_table.md"), "w").write(md + "\n")
    open(os.path.join(RESULTS_DIR, "results_table.tex"), "w").write("\n".join(tex) + "\n")
    print(md)


def plot_label_efficiency(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = {"Scratch": "#555555", "Random": "#aaaaaa", "SimCLR": "#2a7ab9", "ImageNet": "#d9622b"}
    for row in rows:
        xs = [f * 100 for f in FRACS if f in row]
        if len(xs) < 2:
            continue
        ys = [row[f][0] for f in FRACS if f in row]
        es = [row[f][1] for f in FRACS if f in row]
        color = colors[row["method"].split()[0]]
        ls = "--" if "linear" in row["method"] else "-"
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3, color=color, ls=ls,
                    label=row["method"])
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1%\n(500)", "10%\n(5k)", "100%\n(50k)"])
    ax.set_xlabel("Labelled training images")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "label_efficiency.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pretrain_curve():
    p = os.path.join(RESULTS_DIR, "simclr_pretrain_log.json")
    if not os.path.exists(p):
        return
    log = json.load(open(p))["epochs"]
    fig, ax1 = plt.subplots(figsize=(5.2, 2.8))
    ax1.plot([e["epoch"] for e in log], [e["loss"] for e in log], color="tab:blue")
    ax1.set_xlabel("Pretraining epoch")
    ax1.set_ylabel("NT-Xent loss", color="tab:blue")
    ax2 = ax1.twinx()
    k = [e for e in log if "knn_acc" in e]
    ax2.plot([e["epoch"] for e in k], [e["knn_acc"] * 100 for e in k], "o-", color="tab:red")
    ax2.set_ylabel("kNN accuracy (%)", color="tab:red")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "simclr_pretrain.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_confusions(runs):
    keys = [("scratch", "finetune", 0.1), ("simclr", "finetune", 0.1), ("imagenet", "finetune", 0.1)]
    keys = [k for k in keys if runs.get(k)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.1))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        cm = np.array(runs[k][0]["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{LABELS[k[:2]]}\n(10% labels)", fontsize=8)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels([c[:4] for c in CLASSES], rotation=90, fontsize=6)
        ax.set_yticklabels([c[:4] for c in CLASSES], fontsize=6)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]*100:.0f}", ha="center", va="center", fontsize=5,
                            color="white" if cm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_10pct.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_augmentations():
    torch.manual_seed(0)
    x, _ = load_cifar10(torch.device("cuda"), train=True)
    xb = x[:6]
    views = [xb.float() / 255] + [simclr_view(xb) for _ in range(5)]
    fig, axes = plt.subplots(6, 6, figsize=(4.0, 4.0))
    for r in range(6):
        for c in range(6):
            axes[r, c].imshow(views[c][r].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            axes[r, c].axis("off")
    axes[0, 0].set_title("orig", fontsize=7)
    for c in range(1, 6):
        axes[0, c].set_title(f"view {c}", fontsize=7)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(FIG_DIR, "simclr_views.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "simclr_views.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    os.makedirs(FIG_DIR, exist_ok=True)
    if "--aug" in sys.argv:
        plot_augmentations()
    runs = load_runs()
    rows = summarize(runs)
    write_tables(rows)
    plot_label_efficiency(rows)
    plot_pretrain_curve()
    plot_confusions(runs)
