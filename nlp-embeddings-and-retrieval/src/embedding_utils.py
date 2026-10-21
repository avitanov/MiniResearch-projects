"""Embedding helpers for static and contextual embedding comparisons."""

import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_static_neighbors(
    model: KeyedVectors,
    word: str,
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """
    Find nearest neighbors from a static embedding model.

    Args:
        model (KeyedVectors): Gensim keyed-vector model.
        word (str): Word to search around.
        top_n (int): Number of neighbors to return. Defaults to 5.

    Returns:
        List[Dict[str, float]]: Neighbor words and similarity scores.
    """
    neighbors = []
    for neighbor, score in model.most_similar(word, topn=top_n):
        neighbors.append(
            {
                "word": neighbor,
                "similarity": float(score),
            }
        )
    return neighbors


def run_static_analogy(
    model: KeyedVectors,
    positive_words: List[str],
    negative_words: List[str],
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """
    Run a vector analogy with a static embedding model.

    Args:
        model (KeyedVectors): Gensim keyed-vector model.
        positive_words (List[str]): Words added in the analogy.
        negative_words (List[str]): Words subtracted in the analogy.
        top_n (int): Number of candidates to return. Defaults to 5.

    Returns:
        List[Dict[str, float]]: Analogy candidates and similarity scores.
    """
    candidates = []
    results = model.most_similar(
        positive=positive_words,
        negative=negative_words,
        topn=top_n,
    )
    for word, score in results:
        candidates.append(
            {
                "word": word,
                "similarity": float(score),
            }
        )
    return candidates


def compute_cosine_similarity(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        first_vector (np.ndarray): First vector with shape (embedding_dim,).
        second_vector (np.ndarray): Second vector with shape (embedding_dim,).

    Returns:
        float: Cosine similarity score.
    """
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    return float(np.dot(first_vector, second_vector) / (denominator + 1e-12))


def find_word_span(
    sentence: str,
    word: str,
) -> Tuple[int, int]:
    """
    Find the first whole-word character span for a word in a sentence.

    Args:
        sentence (str): Sentence that contains the word.
        word (str): Whole target word to locate.

    Returns:
        Tuple[int, int]: Start and end character offsets.
    """
    pattern = rf"\b{re.escape(word)}\b"
    match = re.search(
        pattern,
        sentence,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Whole word '{word}' was not found in the sentence.")
    return match.start(), match.end()


def get_contextual_word_embedding(
    sentence: str,
    word: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> np.ndarray:
    """
    Extract a contextual embedding for a word in a sentence.

    The selected token vectors from the last hidden state are averaged when the
    word is split into multiple WordPiece tokens.

    Args:
        sentence (str): Sentence containing the target word.
        word (str): Target word to extract.
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer.
        model (PreTrainedModel): HuggingFace model with hidden states enabled.

    Returns:
        np.ndarray: Contextual word vector with shape (hidden_size,).
    """
    word_start, word_end = find_word_span(
        sentence=sentence,
        word=word,
    )
    inputs = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = inputs.pop("offset_mapping")[0].tolist()

    matching_token_ids = []
    for token_index, offset in enumerate(offsets):
        token_start, token_end = offset
        overlaps_word = token_start < word_end and token_end > word_start
        if overlaps_word:
            matching_token_ids.append(token_index)

    if not matching_token_ids:
        raise ValueError(f"No tokenizer tokens matched '{word}'.")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    vectors = outputs.last_hidden_state[0, matching_token_ids, :]
    return vectors.mean(dim=0).detach().cpu().numpy()


def project_embeddings_to_2d(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Project embeddings to two dimensions with PCA.

    Args:
        embeddings (np.ndarray): Embedding matrix with shape
            (n_items, embedding_dim).

    Returns:
        np.ndarray: Projected coordinates with shape (n_items, 2).
    """
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(embeddings)
