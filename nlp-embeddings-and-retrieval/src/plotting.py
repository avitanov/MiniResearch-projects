"""Plotting helpers for saved notebook figures."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_line_plot(
    x_values: List[int],
    y_values: List[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """
    Save a simple line plot to disk.

    Args:
        x_values (List[int]): X-axis values.
        y_values (List[float]): Y-axis values.
        title (str): Plot title.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        output_path (Path): Destination image path.

    Returns:
        None: This function writes a plot file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_dual_axis_line_plot(
    x_values: List[int],
    left_y_values: List[float],
    right_y_values: List[float],
    title: str,
    x_label: str,
    left_y_label: str,
    right_y_label: str,
    left_series_label: str,
    right_series_label: str,
    output_path: Path,
) -> None:
    """
    Save two line series with separate y-axes to disk.

    This is useful when two curves share the same x-axis but have different
    scales, such as training loss and validation accuracy by epoch.

    Args:
        x_values (List[int]): X-axis values.
        left_y_values (List[float]): Left-axis values.
        right_y_values (List[float]): Right-axis values.
        title (str): Plot title.
        x_label (str): X-axis label.
        left_y_label (str): Left y-axis label.
        right_y_label (str): Right y-axis label.
        left_series_label (str): Legend label for the left-axis series.
        right_series_label (str): Legend label for the right-axis series.
        output_path (Path): Destination image path.

    Returns:
        None: This function writes a plot file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, left_axis = plt.subplots(figsize=(7, 4))
    right_axis = left_axis.twinx()

    left_axis.plot(
        x_values,
        left_y_values,
        marker="o",
        color="tab:blue",
        label=left_series_label,
    )
    right_axis.plot(
        x_values,
        right_y_values,
        marker="s",
        color="tab:green",
        label=right_series_label,
    )

    left_axis.set_title(title)
    left_axis.set_xlabel(x_label)
    left_axis.set_ylabel(left_y_label, color="tab:blue")
    right_axis.set_ylabel(right_y_label, color="tab:green")
    left_axis.tick_params(axis="y", labelcolor="tab:blue")
    right_axis.tick_params(axis="y", labelcolor="tab:green")
    left_axis.grid(alpha=0.3)

    lines = left_axis.get_lines() + right_axis.get_lines()
    labels = [line.get_label() for line in lines]
    left_axis.legend(lines, labels, loc="center right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_embedding_projection_plot(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    label_column: str,
    text_column: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Save a labeled 2D embedding projection plot.

    Args:
        dataframe (pd.DataFrame): Table with projection coordinates and labels.
        x_column (str): Column name for x-axis values.
        y_column (str): Column name for y-axis values.
        label_column (str): Column name for point colors.
        text_column (str): Column name for point text labels.
        title (str): Plot title.
        output_path (Path): Destination image path.

    Returns:
        None: This function writes a plot file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    labels = sorted(dataframe[label_column].unique())

    for label in labels:
        label_frame = dataframe.loc[dataframe[label_column].eq(label)]
        plt.scatter(
            label_frame[x_column],
            label_frame[y_column],
            label=label,
            s=55,
        )

    for _, row in dataframe.iterrows():
        plt.text(
            row[x_column] + 0.01,
            row[y_column] + 0.01,
            str(row[text_column]),
            fontsize=8,
        )

    plt.title(title)
    plt.xlabel("PCA dimension 1")
    plt.ylabel("PCA dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """
    Save a normalized confusion matrix heatmap.

    Args:
        matrix (pd.DataFrame): Confusion matrix with row and column labels.
        title (str): Plot title.
        output_path (Path): Destination image path.

    Returns:
        None: This function writes a plot file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix.values, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.colorbar(label="Share of true class")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                f"{matrix.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Save an attention matrix heatmap.

    Args:
        attention_matrix (np.ndarray): Attention scores with shape
            (n_tokens, n_tokens).
        tokens (List[str]): Token labels with length n_tokens.
        title (str): Plot title.
        output_path (Path): Destination image path.

    Returns:
        None: This function writes a plot file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 8))
    plt.imshow(attention_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Tokens attended to")
    plt.ylabel("Tokens attending")
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
    plt.yticks(range(len(tokens)), tokens, fontsize=7)
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_precision_comparison_plot(
    comparison_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a grouped bar chart comparing semantic and keyword precision.

    Args:
        comparison_frame (pd.DataFrame): Table with query,
            semantic_precision_at_5, and keyword_precision_at_5 columns.
        output_path (Path): Destination image path.

    Returns:
        None: This function writes a plot file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_labels = [
        f"Q{index + 1}"
        for index in range(len(comparison_frame))
    ]
    x_positions = np.arange(len(comparison_frame))
    bar_width = 0.36

    plt.figure(figsize=(9, 4.5))
    plt.bar(
        x_positions - bar_width / 2,
        comparison_frame["semantic_precision_at_5"],
        width=bar_width,
        label="Semantic",
    )
    plt.bar(
        x_positions + bar_width / 2,
        comparison_frame["keyword_precision_at_5"],
        width=bar_width,
        label="Keyword",
    )
    plt.title("Task 6 Semantic vs Keyword Precision@5")
    plt.xlabel("Query")
    plt.ylabel("Precision@5")
    plt.xticks(x_positions, query_labels)
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
