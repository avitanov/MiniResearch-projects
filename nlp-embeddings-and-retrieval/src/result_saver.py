"""Save project results to stable files."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(
    data: Any,
    output_path: Path,
) -> None:
    """
    Save JSON-serializable data to disk.

    The parent directory is created automatically so notebook sections can write
    stable result files without repeating directory setup.

    Args:
        data (Any): JSON-serializable object to save.
        output_path (Path): Destination file path.

    Returns:
        None: This function writes a file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a dataframe to a CSV file.

    The parent directory is created automatically so notebook tasks can write
    stable table outputs.

    Args:
        dataframe (pd.DataFrame): Table to save.
        output_path (Path): Destination CSV file path.

    Returns:
        None: This function writes a file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def save_text(
    text: str,
    output_path: Path,
) -> None:
    """
    Save plain text or markdown content to disk.

    The parent directory is created automatically so notebook interpretation
    sections can be reused by the frontend without rerunning notebook cells.

    Args:
        text (str): Text content to save.
        output_path (Path): Destination file path.

    Returns:
        None: This function writes a text file and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
