"""Semantic and keyword search helpers for AG News retrieval."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


AG_NEWS_LABEL_NAMES: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_ag_news_corpus(
    dataset_path: Path,
    split: str,
    sample_size: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """
    Load an AG News split as a document corpus.

    The returned frame includes stable document ids and readable label names so
    retrieval outputs can be saved and inspected later.

    Args:
        dataset_path (Path): Path to a saved HuggingFace AG News dataset.
        split (str): Dataset split name, such as "test".
        sample_size (Optional[int]): Number of rows to sample. Defaults to
            None, which keeps the full split.
        seed (int): Random seed used when sampling rows.

    Returns:
        pd.DataFrame: Corpus with document ids, text, labels, and label names.
    """
    dataset = load_from_disk(str(dataset_path))
    frame = dataset[split].to_pandas()[["text", "label"]].copy()
    frame["label"] = frame["label"].astype(int)

    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(
            n=sample_size,
            random_state=seed,
        )

    frame = frame.reset_index(drop=True)
    frame.insert(0, "doc_id", range(len(frame)))
    frame["label_name"] = frame["label"].map(AG_NEWS_LABEL_NAMES)
    return frame


def build_sentence_transformer_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    show_progress_bar: bool,
) -> Tuple[SentenceTransformer, np.ndarray, float]:
    """
    Encode texts with a sentence-transformer model.

    Args:
        texts (List[str]): Corpus texts to embed.
        model_name (str): Sentence-transformer model name.
        batch_size (int): Number of texts to encode per batch.
        show_progress_bar (bool): Whether to show encoding progress.

    Returns:
        Tuple[SentenceTransformer, np.ndarray, float]: Encoder, embedding
        matrix with shape (n_texts, embedding_dim), and elapsed seconds.
    """
    start_time = time.time()
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    elapsed_seconds = time.time() - start_time
    return encoder, embeddings, float(elapsed_seconds)


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Normalize embeddings so inner product equals cosine similarity.

    Args:
        embeddings (np.ndarray): Embedding matrix with shape
            (n_items, embedding_dim).

    Returns:
        np.ndarray: L2-normalized embedding matrix with shape
        (n_items, embedding_dim).
    """
    normalized_embeddings = embeddings.astype("float32", copy=True)
    faiss.normalize_L2(normalized_embeddings)
    return normalized_embeddings


def build_faiss_index(
    normalized_embeddings: np.ndarray,
) -> Tuple[faiss.IndexFlatIP, float, int]:
    """
    Build a FAISS inner-product index from normalized embeddings.

    Args:
        normalized_embeddings (np.ndarray): Normalized embedding matrix with
            shape (n_items, embedding_dim).

    Returns:
        Tuple[faiss.IndexFlatIP, float, int]: FAISS index, build time in
        seconds, and embedding memory usage in bytes.
    """
    start_time = time.time()
    dimension = int(normalized_embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)
    build_time_seconds = time.time() - start_time
    memory_bytes = int(normalized_embeddings.nbytes)
    return index, float(build_time_seconds), memory_bytes


def run_semantic_search(
    query: str,
    encoder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    corpus_frame: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Search a FAISS index with a sentence-transformer query embedding.

    Args:
        query (str): Natural-language search query.
        encoder (SentenceTransformer): Fitted sentence-transformer encoder.
        index (faiss.IndexFlatIP): FAISS index over normalized corpus vectors.
        corpus_frame (pd.DataFrame): Corpus rows with document ids and labels.
        top_k (int): Number of nearest neighbors to return.

    Returns:
        pd.DataFrame: Ranked semantic results with scores and document fields.
    """
    query_embedding = encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    query_embedding = normalize_embeddings(query_embedding)
    scores, indices = index.search(query_embedding, top_k)

    rows = []
    for rank, (score, index_id) in enumerate(zip(scores[0], indices[0]), start=1):
        document = corpus_frame.iloc[int(index_id)]
        rows.append(
            {
                "rank": rank,
                "doc_id": int(document["doc_id"]),
                "score": float(score),
                "label": int(document["label"]),
                "label_name": str(document["label_name"]),
                "text": str(document["text"]),
            }
        )
    return pd.DataFrame(rows)


def fit_keyword_search(
    texts: List[str],
    max_features: int,
) -> Tuple[TfidfVectorizer, Any, float]:
    """
    Fit a TF-IDF keyword search baseline.

    Args:
        texts (List[str]): Corpus texts used to fit the vectorizer.
        max_features (int): Maximum number of TF-IDF features.

    Returns:
        Tuple[TfidfVectorizer, Any, float]: Fitted vectorizer, sparse TF-IDF
        matrix with shape (n_texts, n_features), and elapsed seconds.
    """
    start_time = time.time()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    elapsed_seconds = time.time() - start_time
    return vectorizer, matrix, float(elapsed_seconds)


def run_keyword_search(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    corpus_frame: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Search the corpus with TF-IDF cosine similarity.

    Args:
        query (str): Natural-language search query.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        matrix (Any): Sparse TF-IDF matrix with shape (n_texts, n_features).
        corpus_frame (pd.DataFrame): Corpus rows with document ids and labels.
        top_k (int): Number of highest-scoring documents to return.

    Returns:
        pd.DataFrame: Ranked keyword results with scores and document fields.
    """
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = np.argsort(scores)[::-1][:top_k]

    rows = []
    for rank, index_id in enumerate(top_indices, start=1):
        document = corpus_frame.iloc[int(index_id)]
        rows.append(
            {
                "rank": rank,
                "doc_id": int(document["doc_id"]),
                "score": float(scores[index_id]),
                "label": int(document["label"]),
                "label_name": str(document["label_name"]),
                "text": str(document["text"]),
            }
        )
    return pd.DataFrame(rows)


def add_relevance_labels(
    results_frame: pd.DataFrame,
    query: str,
    method: str,
    expected_label: int,
) -> pd.DataFrame:
    """
    Add query metadata and coarse relevance labels to search results.

    Args:
        results_frame (pd.DataFrame): Ranked search results.
        query (str): Query text that produced the results.
        method (str): Search method name.
        expected_label (int): AG News label treated as relevant for the query.

    Returns:
        pd.DataFrame: Results with query, method, and relevance columns.
    """
    output_frame = results_frame.copy()
    output_frame.insert(0, "query", query)
    output_frame.insert(1, "method", method)
    output_frame["expected_label"] = expected_label
    output_frame["expected_label_name"] = AG_NEWS_LABEL_NAMES[expected_label]
    output_frame["is_relevant"] = output_frame["label"].eq(expected_label)
    return output_frame


def compute_precision_at_k(
    results_frame: pd.DataFrame,
    k: int,
) -> float:
    """
    Compute precision@k from boolean relevance labels.

    Args:
        results_frame (pd.DataFrame): Search results with an is_relevant column.
        k (int): Number of top ranked results to evaluate.

    Returns:
        float: Share of top-k rows marked relevant.
    """
    top_results = results_frame.head(k)
    if top_results.empty:
        return 0.0
    return float(top_results["is_relevant"].mean())


def compare_search_methods(
    labeled_results_frame: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """
    Build a precision comparison table for semantic and keyword search.

    Args:
        labeled_results_frame (pd.DataFrame): Combined labeled search results.
        k (int): Number of results used for precision evaluation.

    Returns:
        pd.DataFrame: One row per query with semantic and keyword precision.
    """
    rows = []
    for query, query_frame in labeled_results_frame.groupby("query", sort=False):
        expected_label_name = str(query_frame["expected_label_name"].iloc[0])
        semantic_frame = query_frame.loc[query_frame["method"].eq("semantic")]
        keyword_frame = query_frame.loc[query_frame["method"].eq("keyword")]
        semantic_precision = compute_precision_at_k(
            results_frame=semantic_frame,
            k=k,
        )
        keyword_precision = compute_precision_at_k(
            results_frame=keyword_frame,
            k=k,
        )

        if semantic_precision > keyword_precision:
            better_method = "semantic"
        elif keyword_precision > semantic_precision:
            better_method = "keyword"
        else:
            better_method = "tie"

        rows.append(
            {
                "query": query,
                "expected_label_name": expected_label_name,
                "semantic_precision_at_5": semantic_precision,
                "keyword_precision_at_5": keyword_precision,
                "better_method": better_method,
            }
        )
    return pd.DataFrame(rows)
