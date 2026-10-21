"""Classification helpers for TF-IDF and DistilBERT sentiment models."""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


LABEL_NAMES: Dict[int, str] = {
    0: "negative",
    1: "positive",
}


class ReviewDataset(Dataset):
    """PyTorch dataset for tokenized review classification examples."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        """
        Initialize the review dataset.

        Args:
            texts (List[str]): Review texts.
            labels (List[int]): Integer labels for the reviews.
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer.
            max_length (int): Maximum token length after truncation.

        Returns:
            None: This method does not return a value.
        """
        self.texts: List[str] = texts
        self.labels: List[int] = labels
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        """
        Return the number of reviews in the dataset.

        Returns:
            int: Number of reviews.
        """
        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize one review and return tensors for training.

        Args:
            index (int): Row index to fetch.

        Returns:
            Dict[str, torch.Tensor]: Token tensors and one label tensor.
        """
        encoding = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def dataset_split_to_frame(
    dataset_split: Any,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Convert one HuggingFace dataset split to a stratified pandas frame.

    Args:
        dataset_split (Any): HuggingFace dataset split with text and label.
        sample_size (int): Number of rows to keep while matching the full split
            label distribution as closely as possible.
        seed (int): Random seed for shuffling.

    Returns:
        pd.DataFrame: Stratified frame with text and label columns.
    """
    full_frame = dataset_split.to_pandas()[["text", "label"]].copy()
    full_frame["label"] = full_frame["label"].astype(int)

    label_counts = full_frame["label"].value_counts().sort_index()
    label_shares = label_counts / len(full_frame)
    target_counts = (label_shares * sample_size).round().astype(int)

    count_difference = sample_size - int(target_counts.sum())
    if count_difference != 0:
        largest_label = int(label_counts.idxmax())
        target_counts.loc[largest_label] += count_difference

    sampled_frames = []
    for label, target_count in target_counts.items():
        label_frame = full_frame.loc[full_frame["label"].eq(label)]
        sampled_frame = label_frame.sample(
            n=int(target_count),
            random_state=seed,
        )
        sampled_frames.append(sampled_frame)

    output_frame = pd.concat(sampled_frames)
    output_frame = output_frame.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)
    return output_frame


def split_train_validation_frame(
    frame: pd.DataFrame,
    validation_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a sampled training frame into non-overlapping train and validation rows.

    The split is stratified by label so the validation subset keeps the same
    sentiment balance as the sampled training data.

    Args:
        frame (pd.DataFrame): Sampled training rows with text and label columns.
        validation_size (int): Number of rows to reserve for validation.
        seed (int): Random seed for deterministic splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation frames.
    """
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_size,
        random_state=seed,
        stratify=frame["label"],
    )
    return (
        train_frame.reset_index(drop=True),
        validation_frame.reset_index(drop=True),
    )


def train_tfidf_baseline(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    max_features: int,
) -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, float]]:
    """
    Train and evaluate a TF-IDF logistic regression baseline.

    Args:
        train_frame (pd.DataFrame): Training rows with text and label columns.
        test_frame (pd.DataFrame): Test rows with text and label columns.
        max_features (int): Maximum number of TF-IDF features.

    Returns:
        Tuple[LogisticRegression, TfidfVectorizer, Dict[str, float]]: Fitted
        classifier, fitted vectorizer, and metrics.
    """
    start_time = time.time()

    vectorizer = TfidfVectorizer(max_features=max_features)
    train_vectors = vectorizer.fit_transform(train_frame["text"])
    test_vectors = vectorizer.transform(test_frame["text"])

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_vectors, train_frame["label"])

    predictions = classifier.predict(test_vectors)
    training_time = time.time() - start_time

    metrics = {
        "accuracy": float(accuracy_score(test_frame["label"], predictions)),
        "macro_f1": float(f1_score(test_frame["label"], predictions, average="macro")),
        "training_time_seconds": float(training_time),
    }
    return classifier, vectorizer, metrics


def create_review_loader(
    frame: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Create a dataloader for DistilBERT training or evaluation.

    Args:
        frame (pd.DataFrame): Rows with text and label columns.
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer.
        max_length (int): Maximum token length after truncation.
        batch_size (int): Number of rows per batch.
        shuffle (bool): Whether to shuffle batches.

    Returns:
        DataLoader: PyTorch dataloader.
    """
    dataset = ReviewDataset(
        texts=frame["text"].tolist(),
        labels=frame["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_distilbert(
    model: PreTrainedModel,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate a DistilBERT classifier.

    Args:
        model (PreTrainedModel): Sequence classification model.
        data_loader (DataLoader): Evaluation dataloader.
        device (torch.device): Device used for model execution.

    Returns:
        Dict[str, Any]: Accuracy, macro F1, predictions, labels, and
        confidences.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_confidences.extend(confidences.cpu().numpy().tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(all_labels, all_predictions, average="macro")),
        "predictions": all_predictions,
        "labels": all_labels,
        "confidences": all_confidences,
    }


def train_distilbert(
    model: PreTrainedModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    device: torch.device,
) -> Tuple[PreTrainedModel, pd.DataFrame, float]:
    """
    Fine-tune a DistilBERT classifier with a simple PyTorch loop.

    Args:
        model (PreTrainedModel): Sequence classification model.
        train_loader (DataLoader): Training dataloader.
        validation_loader (DataLoader): Validation dataloader.
        learning_rate (float): AdamW learning rate.
        epochs (int): Number of training epochs.
        device (torch.device): Device used for model execution.

    Returns:
        Tuple[PreTrainedModel, pd.DataFrame, float]: Trained model, epoch
        history, and training time in seconds.
    """
    start_time = time.time()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    model.to(device)
    history_rows = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        average_loss = total_loss / len(train_loader)
        validation_result = evaluate_distilbert(
            model=model,
            data_loader=validation_loader,
            device=device,
        )
        history_rows.append(
            {
                "epoch": epoch + 1,
                "training_loss": average_loss,
                "validation_accuracy": validation_result["accuracy"],
            }
        )

    training_time = time.time() - start_time
    return model, pd.DataFrame(history_rows), float(training_time)


def build_confusion_matrix_frame(
    labels: List[int],
    predictions: List[int],
) -> pd.DataFrame:
    """
    Build a normalized confusion matrix table.

    Args:
        labels (List[int]): True labels.
        predictions (List[int]): Predicted labels.

    Returns:
        pd.DataFrame: Normalized confusion matrix with label names.
    """
    matrix = confusion_matrix(
        labels,
        predictions,
        normalize="true",
    )
    return pd.DataFrame(
        matrix,
        index=["true_negative", "true_positive"],
        columns=["pred_negative", "pred_positive"],
    )


def get_misclassified_examples(
    test_frame: pd.DataFrame,
    labels: List[int],
    predictions: List[int],
    confidences: List[float],
    sample_count: int = 5,
) -> pd.DataFrame:
    """
    Collect misclassified examples for review.

    Args:
        test_frame (pd.DataFrame): Test rows with text and label columns.
        labels (List[int]): True labels.
        predictions (List[int]): Predicted labels.
        confidences (List[float]): Prediction confidences.
        sample_count (int): Number of mistakes to return. Defaults to 5.

    Returns:
        pd.DataFrame: Misclassified examples with labels and confidence.
    """
    rows = []
    for index, true_label in enumerate(labels):
        predicted_label = predictions[index]
        if true_label == predicted_label:
            continue
        rows.append(
            {
                "text": test_frame.iloc[index]["text"],
                "true_label": LABEL_NAMES[int(true_label)],
                "predicted_label": LABEL_NAMES[int(predicted_label)],
                "confidence": float(confidences[index]),
            }
        )
        if len(rows) >= sample_count:
            break
    return pd.DataFrame(rows)
