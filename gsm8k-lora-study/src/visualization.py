from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_random_matrix_tradeoff(results: pd.DataFrame, dimension: int) -> tuple[plt.Figure, np.ndarray]:
    """Plot reconstruction error against the LoRA parameter budget.

    Args:
        results: Table containing rank, error, and parameter-count columns.
        dimension: Square matrix dimension used for the full-weight baseline.

    Returns:
        The Matplotlib figure and its two axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(results["rank"], results["relative_error"], marker="o", linewidth=2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Relative Frobenius error")
    axes[0].set_title("Random matrix reconstruction error")
    axes[0].grid(alpha=0.3)

    axes[1].plot(results["rank"], results["LoRA_params"], marker="o", linewidth=2, color="#c44e52")
    axes[1].axhline(dimension * dimension, linestyle="--", color="black", label="Full matrix params")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Trainable parameters")
    axes[1].set_title("LoRA parameter budget")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_structure_comparison(curves: dict[str, pd.DataFrame]) -> tuple[plt.Figure, plt.Axes]:
    """Plot rank-approximation curves for multiple matrix structures.

    Args:
        curves: Mapping from curve labels to rank and error tables.

    Returns:
        The Matplotlib figure and comparison axis.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, frame in curves.items():
        ax.plot(frame["rank"], frame["relative_error"], marker="o", linewidth=2, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Reconstruction error depends on matrix structure")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_loss_curves(log_history: pd.DataFrame, title: str) -> pd.DataFrame:
    """Plot available training and validation loss curves.

    Args:
        log_history: Trainer log records or an equivalent table.
        title: Plot title.

    Returns:
        The normalized log-history DataFrame.
    """
    history = pd.DataFrame(log_history)
    if history.empty:
        print("No log history found.")
        return history

    train_curve = history.loc[history["loss"].notna(), ["step", "loss"]].copy()
    eval_curve = history.loc[history["eval_loss"].notna(), ["step", "eval_loss"]].copy()

    plt.figure(figsize=(8, 4))
    if not train_curve.empty:
        plt.plot(train_curve["step"], train_curve["loss"], label="train loss")
    if not eval_curve.empty:
        plt.plot(eval_curve["step"], eval_curve["eval_loss"], label="val loss")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    return history
