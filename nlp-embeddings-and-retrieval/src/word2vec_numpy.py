"""NumPy helpers for training a small skip-gram Word2Vec model."""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

UNK_TOKEN: str = "<UNK>"


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenize text with lowercase regex word matching.

    This tokenizer is intentionally simple for the from-scratch Word2Vec task.
    It keeps common contractions as one token and removes punctuation.

    Args:
        text (str): Input text to tokenize.

    Returns:
        List[str]: Lowercase tokens from the input text.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def tokenize_texts(
    texts: List[str],
) -> List[List[str]]:
    """
    Tokenize a list of texts into word-token lists.

    Args:
        texts (List[str]): Raw text examples to tokenize.

    Returns:
        List[List[str]]: Tokenized texts, where each inner list contains tokens.
    """
    tokenized_texts = []
    for text in texts:
        tokens = simple_tokenize(text)
        if tokens:
            tokenized_texts.append(tokens)
    return tokenized_texts


def build_vocabulary(
    tokenized_texts: List[List[str]],
    vocab_size: int,
    min_count: int = 2,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    """
    Build a top-frequency vocabulary from tokenized texts.

    Args:
        tokenized_texts (List[List[str]]): Tokenized texts used for vocabulary
            counting.
        vocab_size (int): Maximum vocabulary size, including the unknown token.
        min_count (int): Minimum token frequency required for the vocabulary.
            Defaults to 1.

    Returns:
        Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]: Word-to-id map,
        id-to-word map, and raw token counts.
    """
    token_counts: Counter[str] = Counter()
    for tokens in tokenized_texts:
        token_counts.update(tokens)

    vocabulary = {UNK_TOKEN: 0}
    for word, count in token_counts.most_common():
        if len(vocabulary) >= vocab_size:
            break
        if count >= min_count:
            vocabulary[word] = len(vocabulary)

    id_to_word = {word_id: word for word, word_id in vocabulary.items()}
    return vocabulary, id_to_word, dict(token_counts)


def encode_sentences(
    tokenized_texts: List[List[str]],
    vocabulary: Dict[str, int],
) -> List[List[int]]:
    """
    Replace tokens with vocabulary ids.

    Args:
        tokenized_texts (List[List[str]]): Tokenized texts to encode.
        vocabulary (Dict[str, int]): Word-to-id lookup table.

    Returns:
        List[List[int]]: Encoded sentences containing integer word ids.
    """
    unknown_id = vocabulary[UNK_TOKEN]
    encoded_sentences = []
    for tokens in tokenized_texts:
        encoded_tokens = [
            vocabulary.get(token, unknown_id)
            for token in tokens
        ]
        encoded_sentences.append(encoded_tokens)
    return encoded_sentences


def build_word_counts(
    encoded_sentences: List[List[int]],
    vocab_size: int,
) -> np.ndarray:
    """
    Count encoded word ids for negative sampling.

    Args:
        encoded_sentences (List[List[int]]): Encoded sentences with word ids.
        vocab_size (int): Number of rows in the vocabulary.

    Returns:
        np.ndarray: Word counts with shape (vocab_size,).
    """
    counts = np.zeros(vocab_size, dtype=np.float64)
    for sentence in encoded_sentences:
        for word_id in sentence:
            counts[word_id] += 1.0
    return counts


def generate_skipgram_pairs(
    encoded_sentences: List[List[int]],
    window_size: int,
) -> np.ndarray:
    """
    Generate center-context training pairs with a sliding window.

    Args:
        encoded_sentences (List[List[int]]): Encoded sentences with word ids.
        window_size (int): Number of words to look left and right.

    Returns:
        np.ndarray: Training pairs with shape (n_pairs, 2). Column 0 contains
        center ids and column 1 contains context ids.
    """
    pairs = []
    for sentence in encoded_sentences:
        for center_index, center_id in enumerate(sentence):
            left_index = max(0, center_index - window_size)
            right_index = min(len(sentence), center_index + window_size + 1)

            for context_index in range(left_index, right_index):
                if context_index == center_index:
                    continue
                pairs.append((center_id, sentence[context_index]))

    return np.array(pairs, dtype=np.int64)


def compute_cosine_similarity(
    query_vector: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and a matrix.

    Args:
        query_vector (np.ndarray): Query vector with shape (embedding_dim,).
        matrix (np.ndarray): Matrix of vectors with shape
            (n_items, embedding_dim).

    Returns:
        np.ndarray: Similarity scores with shape (n_items,).
    """
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominator = (query_norm * matrix_norms) + 1e-12
    return matrix @ query_vector / denominator


def get_nearest_neighbors(
    word: str,
    embeddings: np.ndarray,
    vocabulary: Dict[str, int],
    id_to_word: Dict[int, str],
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """
    Find nearest words by cosine similarity.

    Args:
        word (str): Probe word to search around.
        embeddings (np.ndarray): Word embedding matrix with shape
            (vocab_size, embedding_dim).
        vocabulary (Dict[str, int]): Word-to-id lookup table.
        id_to_word (Dict[int, str]): Id-to-word lookup table.
        top_n (int): Number of neighbors to return. Defaults to 5.

    Returns:
        List[Dict[str, float]]: Nearest words and similarity scores.
    """
    if word not in vocabulary:
        return []

    word_id = vocabulary[word]
    similarities = compute_cosine_similarity(
        query_vector=embeddings[word_id],
        matrix=embeddings,
    )
    similarities[word_id] = -np.inf
    best_ids = np.argsort(similarities)[::-1][:top_n]

    neighbors = []
    for neighbor_id in best_ids:
        neighbors.append(
            {
                "word": id_to_word[int(neighbor_id)],
                "similarity": float(similarities[neighbor_id]),
            }
        )
    return neighbors


def run_analogy(
    positive_word: str,
    negative_word: str,
    target_word: str,
    embeddings: np.ndarray,
    vocabulary: Dict[str, int],
    id_to_word: Dict[int, str],
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """
    Solve a simple analogy with vector arithmetic.

    The query is positive_word - negative_word + target_word.

    Args:
        positive_word (str): First positive word in the analogy.
        negative_word (str): Word to subtract from the analogy.
        target_word (str): Second positive word in the analogy.
        embeddings (np.ndarray): Word embedding matrix with shape
            (vocab_size, embedding_dim).
        vocabulary (Dict[str, int]): Word-to-id lookup table.
        id_to_word (Dict[int, str]): Id-to-word lookup table.
        top_n (int): Number of candidates to return. Defaults to 5.

    Returns:
        List[Dict[str, float]]: Analogy candidates and similarity scores.
    """
    required_words = [positive_word, negative_word, target_word]
    if any(word not in vocabulary for word in required_words):
        return []

    query_vector = (
        embeddings[vocabulary[positive_word]]
        - embeddings[vocabulary[negative_word]]
        + embeddings[vocabulary[target_word]]
    )
    similarities = compute_cosine_similarity(
        query_vector=query_vector,
        matrix=embeddings,
    )

    for word in required_words:
        similarities[vocabulary[word]] = -np.inf

    best_ids = np.argsort(similarities)[::-1][:top_n]
    candidates = []
    for candidate_id in best_ids:
        candidates.append(
            {
                "word": id_to_word[int(candidate_id)],
                "similarity": float(similarities[candidate_id]),
            }
        )
    return candidates


class SkipGramWord2Vec:
    """Train a small skip-gram Word2Vec model with negative sampling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        negative_samples: int = 5,
        learning_rate: float = 0.01,
        n_epochs: int = 5,
        seed: int = 42,
    ) -> None:
        """
        Initialize the skip-gram model.

        Args:
            vocab_size (int): Number of vocabulary items.
            embedding_dim (int): Size of each word vector. Defaults to 100.
            negative_samples (int): Number of negative words sampled for each
                positive pair. Defaults to 5.
            learning_rate (float): Gradient descent step size. Defaults to 0.01.
            n_epochs (int): Number of full passes over the training pairs.
                Defaults to 5.
            seed (int): Random seed for initialization and sampling.
                Defaults to 42.

        Returns:
            None: This method does not return a value.
        """
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
        self.negative_samples: int = negative_samples
        self.learning_rate: float = learning_rate
        self.n_epochs: int = n_epochs
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.losses: List[float] = []
        self.noise_distribution: Optional[np.ndarray] = None

        limit = 0.5 / embedding_dim
        self.input_embeddings: np.ndarray = self.rng.uniform(
            low=-limit,
            high=limit,
            size=(vocab_size, embedding_dim),
        )
        self.output_embeddings: np.ndarray = self.rng.uniform(
            low=-limit,
            high=limit,
            size=(vocab_size, embedding_dim),
        )

    def _build_noise_distribution(
        self,
        word_counts: np.ndarray,
    ) -> np.ndarray:
        """
        Build the unigram distribution raised to the 0.75 power.

        Args:
            word_counts (np.ndarray): Word counts with shape (vocab_size,).

        Returns:
            np.ndarray: Negative-sampling probabilities with shape
            (vocab_size,).
        """
        adjusted_counts = np.power(word_counts, 0.75)
        total_count = adjusted_counts.sum()
        if total_count == 0:
            return np.ones(self.vocab_size) / self.vocab_size
        return adjusted_counts / total_count

    def _sigmoid(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a numerically stable sigmoid.

        Args:
            values (np.ndarray): Input values with any shape.

        Returns:
            np.ndarray: Sigmoid outputs with the same shape as values.
        """
        clipped_values = np.clip(values, -15, 15)
        return 1.0 / (1.0 + np.exp(-clipped_values))

    def _sample_negative_words_for_pairs(
        self,
        training_pairs: np.ndarray,
    ) -> np.ndarray:
        """
        Sample negative word ids for prepared training pairs.

        Args:
            training_pairs (np.ndarray): Center-context ids with shape
                (n_pairs, 2).

        Returns:
            np.ndarray: Negative word ids with shape
            (n_pairs, negative_samples).
        """
        if self.noise_distribution is None:
            raise ValueError("Noise distribution is not initialized.")

        negative_ids = self.rng.choice(
            self.vocab_size,
            size=(len(training_pairs), self.negative_samples),
            p=self.noise_distribution,
        )

        context_ids = training_pairs[:, 1][:, None]
        matching_mask = negative_ids == context_ids

        while np.any(matching_mask):
            replacement_count = int(matching_mask.sum())
            negative_ids[matching_mask] = self.rng.choice(
                self.vocab_size,
                size=replacement_count,
                p=self.noise_distribution,
            )
            matching_mask = negative_ids == context_ids

        return negative_ids.astype(np.int64)

    def _update_pair(
        self,
        center_id: int,
        context_id: int,
        negative_ids: np.ndarray,
    ) -> float:
        """
        Update embeddings for one center-context pair.

        Args:
            center_id (int): Center word id.
            context_id (int): True context word id.
            negative_ids (np.ndarray): Negative word ids with shape
                (negative_samples,).

        Returns:
            float: Negative-sampling loss for the pair.
        """
        center_vector = self.input_embeddings[center_id].copy()
        positive_vector = self.output_embeddings[context_id].copy()
        negative_vectors = self.output_embeddings[negative_ids].copy()

        positive_score = float(self._sigmoid(np.dot(center_vector, positive_vector)))
        negative_scores = self._sigmoid(negative_vectors @ center_vector)

        loss = -np.log(positive_score + 1e-12)
        loss -= float(np.log(1.0 - negative_scores + 1e-12).sum())

        center_gradient = (positive_score - 1.0) * positive_vector
        center_gradient += negative_scores @ negative_vectors
        positive_gradient = (positive_score - 1.0) * center_vector
        negative_gradients = negative_scores[:, None] * center_vector

        self.input_embeddings[center_id] -= self.learning_rate * center_gradient
        self.output_embeddings[context_id] -= self.learning_rate * positive_gradient

        for row_index, negative_id in enumerate(negative_ids):
            self.output_embeddings[negative_id] -= (
                self.learning_rate * negative_gradients[row_index]
            )

        return float(loss)

    def fit(
        self,
        training_pairs: np.ndarray,
        word_counts: np.ndarray,
    ) -> "SkipGramWord2Vec":
        """
        Train the model on prepared skip-gram pairs.

        Args:
            training_pairs (np.ndarray): Center-context ids with shape
                (n_pairs, 2).
            word_counts (np.ndarray): Word counts with shape (vocab_size,).

        Returns:
            SkipGramWord2Vec: Fitted model instance.
        """
        self.noise_distribution = self._build_noise_distribution(
            word_counts=word_counts,
        )
        self.losses = []

        for _ in range(self.n_epochs):
            pair_order = self.rng.permutation(len(training_pairs))
            negative_ids_by_pair = self._sample_negative_words_for_pairs(
                training_pairs=training_pairs,
            )
            epoch_loss = 0.0

            for pair_index in pair_order:
                center_id = int(training_pairs[pair_index, 0])
                context_id = int(training_pairs[pair_index, 1])
                epoch_loss += self._update_pair(
                    center_id=center_id,
                    context_id=context_id,
                    negative_ids=negative_ids_by_pair[pair_index],
                )

            self.losses.append(epoch_loss / len(training_pairs))

        return self

    def get_embeddings(self) -> np.ndarray:
        """
        Return the learned input embeddings.

        Returns:
            np.ndarray: Input embedding matrix with shape
            (vocab_size, embedding_dim).
        """
        return self.input_embeddings.copy()
