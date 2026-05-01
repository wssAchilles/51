from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_segmentation_plot(time: pd.Series | np.ndarray, displacement: np.ndarray, velocity: np.ndarray, breaks: list[int], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, displacement, lw=1.2, color="#1f77b4")
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title(title)
    axes[1].plot(time, velocity, lw=1.0, color="#d62728")
    axes[1].set_ylabel("6h velocity (mm/h)")
    axes[1].set_xlabel("Time")
    for ax in axes:
        for b in breaks:
            ax.axvline(time.iloc[b] if hasattr(time, "iloc") else time[b], ls="--", lw=1.0, color="black", alpha=0.7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_prediction_plot(time: pd.Series | np.ndarray, predicted: np.ndarray, path: Path, title: str, observed: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    if observed is not None:
        ax.plot(time, observed, lw=1.0, label="Observed", color="#1f77b4")
    ax.plot(time, predicted, lw=1.3, label="Predicted", color="#d62728")
    ax.set_title(title)
    ax.set_ylabel("Surface displacement (mm)")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_scatter_plot(x: np.ndarray, y: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=8, alpha=0.55, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

