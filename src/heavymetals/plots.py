from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray, labels: list[str], outpath: str | Path, title: str
) -> None:
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_learning_curve(history, outpath: str | Path, title: str = "Learning curve") -> None:
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    if hasattr(history, "history"):
        h = history.history
        if "loss" in h:
            ax.plot(h["loss"], label="train_loss")
        if "val_loss" in h:
            ax.plot(h["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
