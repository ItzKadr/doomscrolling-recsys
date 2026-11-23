"""Vectorized grouped ranking metrics.

All functions take *flat* arrays (one entry per (session, item) row, rows of a
session contiguous) plus ``sizes`` (number of rows per session). Internally the
data is scattered into a padded ``(n_groups, max_len)`` matrix, sorted once by
score with a single ``np.argsort`` and reduced along axis 1, so there is no
Python loop over sessions in the hot path.

Definitions (rank ``r`` starts at 1)::

    DCG@k   = sum_{r<=k} g(rel_r) / log2(r + 1),   g(rel) = 2^rel - 1  ("exp2")
                                                   or g(rel) = rel      ("linear")
    NDCG@k  = DCG@k / IDCG@k                       (IDCG = DCG of the ideal order)
    Recall@k = |relevant in top-k| / |relevant|    (relevant: rel >= threshold)

Sessions with ``IDCG = 0`` (or no relevant items, for recall) are undefined and
returned as ``NaN``; aggregate with :func:`numpy.nanmean`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Gain = Literal["exp2", "linear"]


def offsets_from_sizes(sizes: np.ndarray) -> np.ndarray:
    """Start index of every group in the flat arrays."""
    sizes = np.asarray(sizes, dtype=np.int64)
    return np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)


def pad_by_group(
    values: np.ndarray, sizes: np.ndarray, fill: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter flat grouped ``values`` into a padded matrix.

    Returns ``(padded, mask)`` with shape ``(n_groups, max(sizes))``; ``mask`` is
    True for real entries. ``padded[mask]`` restores the original flat order.
    """
    values = np.asarray(values)
    sizes = np.asarray(sizes, dtype=np.int64)
    if values.shape[0] != int(sizes.sum()):
        raise ValueError(f"len(values)={values.shape[0]} != sum(sizes)={int(sizes.sum())}")
    n_groups = sizes.size
    width = int(sizes.max()) if n_groups else 0
    rows = np.repeat(np.arange(n_groups), sizes)
    cols = np.arange(values.shape[0]) - np.repeat(offsets_from_sizes(sizes), sizes)
    dtype = np.result_type(values.dtype, np.asarray(fill).dtype)
    padded = np.full((n_groups, width), fill, dtype=dtype)
    padded[rows, cols] = values
    mask = np.zeros((n_groups, width), dtype=bool)
    mask[rows, cols] = True
    return padded, mask


def order_by_score(scores: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Within-group descending order of ``scores``.

    Returns ``(order, mask_sorted)``: ``order[g, r]`` is the column (within the
    padded matrix) of the item placed at rank ``r + 1`` in group ``g``; padding
    always sorts last. Ties keep the original row order (stable sort).
    """
    padded, mask = pad_by_group(np.asarray(scores, dtype=float), sizes, fill=-np.inf)
    key = np.where(mask, -padded, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    return order, np.take_along_axis(mask, order, axis=1)


def ranks_within_groups(scores: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """1-based rank of every row inside its group when sorted by descending score."""
    order, _ = order_by_score(scores, sizes)
    n_groups, width = order.shape
    rank_padded = np.empty_like(order)
    # inverse permutation: the item in column order[g, r] gets rank r + 1
    rank_padded[np.arange(n_groups)[:, None], order] = np.arange(1, width + 1)[None, :]
    _, mask = pad_by_group(np.zeros(int(np.sum(sizes))), sizes)
    return rank_padded[mask]


def discounts(width: int) -> np.ndarray:
    """``1 / log2(r + 1)`` for ranks ``r = 1..width``."""
    return 1.0 / np.log2(np.arange(2, width + 2, dtype=float))


def gains(relevance: np.ndarray, gain: Gain = "exp2") -> np.ndarray:
    """Gain transform of graded relevance."""
    relevance = np.asarray(relevance, dtype=float)
    if gain == "exp2":
        return np.exp2(relevance) - 1.0
    if gain == "linear":
        return relevance
    raise ValueError(f"unknown gain {gain!r}")


def _sorted_values(
    values: np.ndarray, scores: np.ndarray, sizes: np.ndarray, fill: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Values arranged in within-group score order, padded with ``fill``."""
    order, mask_sorted = order_by_score(scores, sizes)
    padded, _ = pad_by_group(np.asarray(values, dtype=float), sizes, fill=fill)
    return np.take_along_axis(padded, order, axis=1), mask_sorted


def dcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sizes: np.ndarray,
    k: int,
    gain: Gain = "exp2",
) -> np.ndarray:
    """Per-group DCG@k of the ranking induced by ``y_score``."""
    g_sorted, _ = _sorted_values(gains(y_true, gain), y_score, sizes)
    width = min(k, g_sorted.shape[1])
    return g_sorted[:, :width] @ discounts(width)


def ideal_dcg_at_k(y_true: np.ndarray, sizes: np.ndarray, k: int, gain: Gain = "exp2") -> np.ndarray:
    """Per-group DCG@k of the ideal ranking (sorted by true relevance)."""
    return dcg_at_k(y_true, y_true, sizes, k, gain)


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sizes: np.ndarray,
    k: int,
    gain: Gain = "exp2",
) -> np.ndarray:
    """Per-group NDCG@k (NaN where the ideal DCG is zero)."""
    dcg = dcg_at_k(y_true, y_score, sizes, k, gain)
    idcg = ideal_dcg_at_k(y_true, sizes, k, gain)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), np.nan)


def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sizes: np.ndarray,
    k: int,
    threshold: float = 3.0,
) -> np.ndarray:
    """Per-group Recall@k with ``relevant := y_true >= threshold`` (NaN if none)."""
    rel = (np.asarray(y_true, dtype=float) >= threshold).astype(float)
    rel_sorted, _ = _sorted_values(rel, y_score, sizes)
    width = min(k, rel_sorted.shape[1])
    hits = rel_sorted[:, :width].sum(axis=1)
    total = np.add.reduceat(rel, offsets_from_sizes(sizes)) if rel.size else np.zeros(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, hits / np.where(total > 0, total, 1.0), np.nan)


def topk_mean(values: np.ndarray, y_score: np.ndarray, sizes: np.ndarray, k: int) -> np.ndarray:
    """Per-group mean of ``values`` over the top-k items (e.g. Freshness@k)."""
    v_sorted, mask_sorted = _sorted_values(values, y_score, sizes)
    width = min(k, v_sorted.shape[1])
    m = mask_sorted[:, :width]
    total = np.where(m, v_sorted[:, :width], 0.0).sum(axis=1)
    return total / np.maximum(m.sum(axis=1), 1)


def group_mean(values: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Per-group mean of a flat array (``np.add.reduceat`` based)."""
    values = np.asarray(values, dtype=float)
    return np.add.reduceat(values, offsets_from_sizes(sizes)) / np.asarray(sizes, dtype=float)
