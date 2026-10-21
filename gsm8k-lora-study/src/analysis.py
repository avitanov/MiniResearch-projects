from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 512]


def rank_approximation_errors(matrix: np.ndarray, ranks: list[int]) -> pd.DataFrame:
    """Measure truncated-SVD reconstruction error at each rank.

    Args:
        matrix: Matrix to approximate.
        ranks: Candidate approximation ranks.

    Returns:
        A table containing each rank and its relative Frobenius error.
    """
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    baseline = np.linalg.norm(matrix, ord="fro")
    rows = []
    for rank in ranks:
        capped_rank = min(rank, s.shape[0])
        approximation = (u[:, :capped_rank] * s[:capped_rank]) @ vt[:capped_rank, :]
        residual = np.linalg.norm(matrix - approximation, ord="fro")
        rows.append(
            {
                "rank": rank,
                "relative_error": float(residual / baseline),
            }
        )
    return pd.DataFrame(rows)


def lora_parameter_stats(dimension: int, ranks: list[int]) -> pd.DataFrame:
    """Calculate LoRA parameter counts for square weight matrices.

    Args:
        dimension: Input and output dimension of the square weight matrix.
        ranks: LoRA ranks to evaluate.

    Returns:
        A table of parameter counts, percentages, and savings by rank.
    """
    full_weight_params = dimension * dimension
    rows = []
    for rank in ranks:
        a_params = dimension * rank
        b_params = rank * dimension
        total_lora_params = a_params + b_params
        rows.append(
            {
                "dimension": dimension,
                "rank": rank,
                "A_params": a_params,
                "B_params": b_params,
                "LoRA_params": total_lora_params,
                "full_weight_params": full_weight_params,
                "lora_pct_of_full": 100 * total_lora_params / full_weight_params,
                "parameter_savings_pct": 100 * (1 - total_lora_params / full_weight_params),
            }
        )
    return pd.DataFrame(rows)


def lora_budget_table(dimensions: list[int], ranks: list[int]) -> pd.DataFrame:
    """Combine LoRA parameter statistics across matrix dimensions.

    Args:
        dimensions: Square weight-matrix dimensions to evaluate.
        ranks: LoRA ranks to evaluate for every dimension.

    Returns:
        A combined parameter-budget table for all dimensions and ranks.
    """
    frames = [lora_parameter_stats(dimension, ranks) for dimension in dimensions]
    return pd.concat(frames, ignore_index=True)


def make_random_matrix(dimension: int, seed: int) -> np.ndarray:
    """Create a reproducible square matrix with standard-normal entries.

    Args:
        dimension: Number of rows and columns.
        seed: Random-number generator seed.

    Returns:
        A square random matrix.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dimension, dimension))


def make_exact_rank_matrix(dimension: int, intrinsic_rank: int, seed: int) -> np.ndarray:
    """Create a reproducible symmetric matrix with bounded intrinsic rank.

    Args:
        dimension: Number of rows and columns.
        intrinsic_rank: Number of latent basis columns.
        seed: Random-number generator seed.

    Returns:
        A symmetric matrix whose rank is at most ``intrinsic_rank``.
    """
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((dimension, intrinsic_rank))
    return basis @ basis.T


def make_diagonal_dominant_matrix(dimension: int, seed: int, noise_scale: float = 0.1) -> np.ndarray:
    """Create a descending diagonal matrix with random additive noise.

    Args:
        dimension: Number of rows and columns.
        seed: Random-number generator seed.
        noise_scale: Standard-normal noise multiplier.

    Returns:
        A noisy, diagonally dominant square matrix.
    """
    rng = np.random.default_rng(seed)
    diagonal = np.diag(np.arange(dimension, 0, -1).astype(float))
    noise = noise_scale * rng.standard_normal((dimension, dimension))
    return diagonal + noise
