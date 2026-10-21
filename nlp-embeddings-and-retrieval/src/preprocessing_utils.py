"""Text preprocessing helpers for Task 0."""

import html
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
from datasets import Dataset, DatasetDict

# Matches simple HTML tags such as <br>, <strong>, </div>, or <a href="...">.
HTML_TAG_PATTERN: str = r"</?[A-Za-z][A-Za-z0-9]*(?:\s+[^<>]*)?/?>"

# Matches HTML entities such as &amp;, &#39;, or &#x27; before unescaping text.
HTML_ENTITY_PATTERN: str = r"&(?:[A-Za-z][A-Za-z0-9]+|#[0-9]+|#x[0-9A-Fa-f]+);"

# Matches non-printable ASCII control characters while keeping tab/newline valid.
CONTROL_CHARACTER_PATTERN: str = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"


def _build_preprocessing_frame(
    dataset: Mapping[str, Any],
    dataset_name: str,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """
    Build one pandas frame from HuggingFace dataset splits.

    Args:
        dataset (Mapping[str, Any]): Dataset splits keyed by split name.
        dataset_name (str): Human-readable dataset name.
        text_column (str): Column containing raw text.
        label_column (str): Column containing labels.

    Returns:
        pd.DataFrame: Frame with raw text, cleaned text, split, label, and
        duplicate-check key.
    """
    frames = []
    for split_name, dataset_split in dataset.items():
        frame = dataset_split.to_pandas()[[text_column, label_column]].copy()
        frame = frame.rename(columns={text_column: "raw_text", label_column: "label"})
        frame["dataset"] = dataset_name
        frame["split"] = split_name
        frame["raw_text"] = frame["raw_text"].fillna("").astype(str)
        frames.append(frame)

    full_frame = pd.concat(frames, ignore_index=True)
    full_frame["cleaned_text"] = (
        full_frame["raw_text"]
        .map(html.unescape)
        .str.replace(HTML_TAG_PATTERN, " ", regex=True)
        .str.replace(CONTROL_CHARACTER_PATTERN, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    full_frame["text_key"] = full_frame["cleaned_text"].str.casefold()
    return full_frame


def _summarize_frame(
    frame: pd.DataFrame,
    dataset_name: str,
    text_column: str,
) -> Dict[str, Any]:
    """
    Summarize required preprocessing checks for a frame.

    Args:
        frame (pd.DataFrame): Preprocessing frame to summarize.
        dataset_name (str): Human-readable dataset name.
        text_column (str): Text column used for artifact checks.

    Returns:
        Dict[str, Any]: Counts for blanks, artifacts, duplicates, label
        conflicts, split overlap, and row totals.
    """
    grouped = frame.groupby("text_key", dropna=False)
    duplicate_sizes = grouped.size()

    return {
        "dataset": dataset_name,
        "row_count": int(len(frame)),
        "blank_text_count": int(frame["cleaned_text"].eq("").sum()),
        "html_tag_count": int(frame[text_column].str.contains(HTML_TAG_PATTERN, regex=True, na=False).sum()),
        "html_entity_count": int(frame[text_column].str.contains(HTML_ENTITY_PATTERN, regex=True, na=False).sum()),
        "control_character_count": int(
            frame[text_column].str.contains(CONTROL_CHARACTER_PATTERN, regex=True, na=False).sum()
        ),
        "repeated_unique_text_count": int(duplicate_sizes.gt(1).sum()),
        "rows_with_repeated_text_count": int(duplicate_sizes[duplicate_sizes.gt(1)].sum()),
        "conflicting_label_group_count": int(grouped["label"].nunique().gt(1).sum()),
        "split_overlap_group_count": int(grouped["split"].nunique().gt(1).sum()),
        "split_counts": {
            str(split): int(count)
            for split, count in frame["split"].value_counts().sort_index().items()
        },
    }


def _to_dataset_dict(frame: pd.DataFrame) -> DatasetDict:
    """
    Convert a cleaned pandas frame to a HuggingFace DatasetDict.

    Args:
        frame (pd.DataFrame): Cleaned frame with split, cleaned text, and label.

    Returns:
        DatasetDict: Cleaned dataset splits with text and label columns.
    """
    split_datasets = {}
    for split_name, split_frame in frame.groupby("split", sort=False):
        output_frame = (
            split_frame[["cleaned_text", "label"]]
            .rename(columns={"cleaned_text": "text"})
            .reset_index(drop=True)
        )
        split_datasets[str(split_name)] = Dataset.from_pandas(
            output_frame,
            preserve_index=False,
        )
    return DatasetDict(split_datasets)


def preprocess_dataset_splits(
    dataset: Mapping[str, Any],
    dataset_name: str,
    output_path: Path,
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Clean, deduplicate, audit, and save train/test text classification splits.

    Args:
        dataset (Mapping[str, Any]): Dataset splits keyed by split name.
        dataset_name (str): Human-readable dataset name.
        output_path (Path): Directory where the cleaned DatasetDict is saved.
        text_column (str): Column containing raw text.
            Defaults to "text".
        label_column (str): Column containing labels.
            Defaults to "label".

    Returns:
        Dict[str, Any]: Raw summary, clean summary, removal counts, two-row
        explainability examples, label distribution, and saved path.
    """
    raw_frame = _build_preprocessing_frame(
        dataset=dataset,
        dataset_name=dataset_name,
        text_column=text_column,
        label_column=label_column,
    )
    raw_summary = _summarize_frame(
        frame=raw_frame,
        dataset_name=dataset_name,
        text_column="raw_text",
    )

    non_blank_frame = raw_frame.loc[raw_frame["cleaned_text"].ne("")].copy()

    labels_per_text = non_blank_frame.groupby("text_key", dropna=False)["label"].nunique()
    conflicting_keys = labels_per_text[labels_per_text.gt(1)].index
    no_conflict_frame = non_blank_frame.loc[~non_blank_frame["text_key"].isin(conflicting_keys)].copy()

    duplicate_mask = no_conflict_frame.duplicated(
        subset=["split", "text_key"],
        keep="first",
    )
    no_duplicate_frame = no_conflict_frame.drop_duplicates(
        subset=["split", "text_key"],
        keep="first",
    ).copy()

    train_keys = set(no_duplicate_frame.loc[no_duplicate_frame["split"].eq("train"), "text_key"])
    test_overlap_mask = no_duplicate_frame["split"].eq("test") & no_duplicate_frame["text_key"].isin(train_keys)
    clean_frame = no_duplicate_frame.loc[~test_overlap_mask].copy()

    artifact_mask = (
        raw_frame["raw_text"].str.contains(HTML_TAG_PATTERN, regex=True, na=False)
        | raw_frame["raw_text"].str.contains(HTML_ENTITY_PATTERN, regex=True, na=False)
        | raw_frame["raw_text"].str.contains(CONTROL_CHARACTER_PATTERN, regex=True, na=False)
    )
    example_columns = ["dataset", "split", "label", "raw_text", "cleaned_text"]

    clean_summary = _summarize_frame(
        frame=clean_frame,
        dataset_name=dataset_name,
        text_column="cleaned_text",
    )
    clean_summary["saved_path"] = str(output_path)

    cleaned_dataset = _to_dataset_dict(clean_frame)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataset.save_to_disk(str(output_path))

    return {
        "dataset": dataset_name,
        "saved_path": str(output_path),
        "raw_summary": raw_summary,
        "clean_summary": clean_summary,
        "removal_summary": {
            "blank_text_rows": int(len(raw_frame) - len(non_blank_frame)),
            "conflicting_label_rows": int(len(non_blank_frame) - len(no_conflict_frame)),
            "duplicate_rows_within_splits": int(len(no_conflict_frame) - len(no_duplicate_frame)),
            "test_overlap_rows": int(len(no_duplicate_frame) - len(clean_frame)),
            "total_removed_rows": int(len(raw_frame) - len(clean_frame)),
        },
        "examples": {
            "blank_text": raw_frame.loc[
                raw_frame["cleaned_text"].eq(""),
                example_columns,
            ]
            .head(2)
            .to_dict("records"),
            "text_artifacts": raw_frame.loc[
                artifact_mask,
                example_columns,
            ]
            .head(2)
            .to_dict("records"),
            "conflicting_labels": non_blank_frame.loc[
                non_blank_frame["text_key"].isin(conflicting_keys),
                example_columns,
            ]
            .head(2)
            .to_dict("records"),
            "duplicate_rows_within_splits": no_conflict_frame.loc[
                duplicate_mask,
                example_columns,
            ]
            .head(2)
            .to_dict("records"),
            "test_overlaps": no_duplicate_frame.loc[
                test_overlap_mask,
                example_columns,
            ]
            .head(2)
            .to_dict("records"),
        },
        "label_distribution": {
            split_name: split_frame["label"].astype(str).value_counts().sort_index().to_dict()
            for split_name, split_frame in clean_frame.groupby("split", sort=False)
        },
    }
