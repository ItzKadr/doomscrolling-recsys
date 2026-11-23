r"""Multi-objective re-ranking: relevance (NDCG@k) vs content freshness (Freshness@k).

Scalarization
-------------
Both signals are put on a common scale *within every session* with Z-scores
(the ranker's raw scores are only meaningful up to a monotone transform and
their spread varies by session)::

    z(x)_i = (x_i - mean_s(x)) / std_s(x)
    Score_MOO = (1 - alpha) * z(rel_hat) + alpha * z(freshness),  alpha in [0, 1]

Sweeping ``alpha`` traces a curve in the (Freshness@k, NDCG@k) plane.

Pareto front and knee
---------------------
A point dominates another if it is no worse in both objectives and strictly
better in one; the non-dominated points form the Pareto front. The operating
point is the *knee*: after min-max normalizing both axes over the front, the
point with the largest perpendicular distance to the chord joining the two
extremes (best NDCG, best freshness). A guard-rail restricts the search to
points whose relative NDCG@k drop vs ``alpha = 0`` is at most ``max_rel_drop``
(2 % by default).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.metrics import ndcg_at_k, offsets_from_sizes, topk_mean

logger = logging.getLogger(__name__)


@dataclass
class KneePoint:
    """Selected operating point of the relevance/freshness trade-off."""

    alpha: float
    ndcg: float
    freshness: float
    ndcg_base: float  # NDCG@k at alpha = 0
    freshness_base: float  # Freshness@k at alpha = 0
    ndcg_drop_rel: float  # 1 - ndcg / ndcg_base
    freshness_gain_rel: float  # freshness / freshness_base - 1
    chord_distance: float
    max_rel_drop: float
    unconstrained_alpha: float  # knee without the NDCG guard-rail

    def to_dict(self) -> dict[str, float]:
        return {key: float(val) for key, val in asdict(self).items()}


def zscore_within_groups(x: np.ndarray, sizes: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score of ``x`` inside each group (0 for constant groups)."""
    x = np.asarray(x, dtype=float)
    sizes = np.asarray(sizes, dtype=np.int64)
    offsets = offsets_from_sizes(sizes)
    n = sizes.astype(float)
    mean = np.add.reduceat(x, offsets) / n
    centred = x - np.repeat(mean, sizes)
    std = np.sqrt(np.add.reduceat(centred**2, offsets) / n)
    std_rep = np.repeat(std, sizes)
    return np.where(std_rep > eps, centred / np.where(std_rep > eps, std_rep, 1.0), 0.0)


def moo_scores(
    rel_hat: np.ndarray, freshness: np.ndarray, sizes: np.ndarray, alpha: float
) -> np.ndarray:
    """``(1 - alpha) * z(rel_hat) + alpha * z(freshness)`` with within-session Z-scores."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return (1.0 - alpha) * zscore_within_groups(rel_hat, sizes) + alpha * zscore_within_groups(
        freshness, sizes
    )


def pareto_mask(points: np.ndarray, maximize: tuple[bool, ...] = (True, True)) -> np.ndarray:
    """Boolean mask of non-dominated rows of ``points`` (n x m), vectorized O(n^2 m)."""
    pts = np.asarray(points, dtype=float)
    signs = np.where(np.asarray(maximize), 1.0, -1.0)
    p = pts * signs  # now maximize everything
    ge = (p[:, None, :] >= p[None, :, :]).all(axis=2)  # ge[j, i]: j >= i everywhere
    gt = (p[:, None, :] > p[None, :, :]).any(axis=2)
    dominated = (ge & gt).any(axis=0)  # some j dominates i
    return ~dominated


def chord_distances(ndcg: np.ndarray, freshness: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Signed distance to the chord between the front's extremes (normalized axes).

    Positive values bulge towards the utopia point (best freshness, best NDCG).
    Non-front points get ``NaN``.
    """
    ndcg = np.asarray(ndcg, dtype=float)
    fresh = np.asarray(freshness, dtype=float)
    idx = np.flatnonzero(front)
    out = np.full(ndcg.shape, np.nan)
    if idx.size < 3:
        out[idx] = 0.0
        return out
    n_rng = np.ptp(ndcg[idx])
    f_rng = np.ptp(fresh[idx])
    if n_rng <= 0 or f_rng <= 0:
        out[idx] = 0.0
        return out
    y = (ndcg - ndcg[idx].min()) / n_rng
    x = (fresh - fresh[idx].min()) / f_rng
    a = idx[np.argmax(ndcg[idx])]  # relevance extreme
    b = idx[np.argmax(fresh[idx])]  # freshness extreme
    bx, by = x[b] - x[a], y[b] - y[a]
    norm = np.hypot(bx, by)
    dist = (bx * (y - y[a]) - by * (x - x[a])) / norm
    out[idx] = dist[idx]
    return out


def sweep_alpha(
    rel_hat: np.ndarray,
    freshness: np.ndarray,
    y_true: np.ndarray,
    sizes: np.ndarray,
    alphas: np.ndarray | None = None,
    k: int = 5,
) -> pd.DataFrame:
    """Evaluate NDCG@k (true relevance) and Freshness@k for every ``alpha``.

    Returns one row per alpha with ``is_pareto``, ``chord_distance`` and the
    relative NDCG drop / freshness gain vs ``alpha = 0``.
    """
    alphas = np.linspace(0.0, 1.0, 41) if alphas is None else np.asarray(alphas, dtype=float)
    z_rel = zscore_within_groups(rel_hat, sizes)
    z_fresh = zscore_within_groups(freshness, sizes)
    rows = []
    for alpha in alphas:  # a handful of alphas; each step is fully vectorized
        score = (1.0 - alpha) * z_rel + alpha * z_fresh
        rows.append(
            {
                "alpha": float(alpha),
                f"ndcg@{k}": float(np.nanmean(ndcg_at_k(y_true, score, sizes, k))),
                f"freshness@{k}": float(np.mean(topk_mean(freshness, score, sizes, k))),
            }
        )
    frame = pd.DataFrame(rows)
    n_col, f_col = f"ndcg@{k}", f"freshness@{k}"
    base = frame.iloc[int(np.argmin(np.abs(frame["alpha"].to_numpy())))]
    frame["ndcg_drop_rel"] = 1.0 - frame[n_col] / base[n_col]
    frame["freshness_gain_rel"] = frame[f_col] / base[f_col] - 1.0
    front = pareto_mask(frame[[n_col, f_col]].to_numpy())
    frame["is_pareto"] = front
    frame["chord_distance"] = chord_distances(frame[n_col].to_numpy(), frame[f_col].to_numpy(), front)
    return frame


def select_knee(frame: pd.DataFrame, k: int = 5, max_rel_drop: float = 0.02) -> KneePoint:
    """Knee of the Pareto front subject to ``ndcg_drop_rel <= max_rel_drop``.

    Falls back to ``alpha = 0`` if no other front point satisfies the guard-rail.
    """
    n_col, f_col = f"ndcg@{k}", f"freshness@{k}"
    base_idx = int(np.argmin(np.abs(frame["alpha"].to_numpy())))
    front = frame["is_pareto"].to_numpy(dtype=bool)
    dist = np.nan_to_num(frame["chord_distance"].to_numpy(dtype=float), nan=-np.inf)
    unconstrained = int(np.argmax(np.where(front, dist, -np.inf)))
    feasible = front & (frame["ndcg_drop_rel"].to_numpy() <= max_rel_drop + 1e-12)
    if feasible.any():
        chosen = int(np.argmax(np.where(feasible, dist, -np.inf)))
        if not np.isfinite(dist[chosen]) or dist[chosen] <= 0:
            # no convex bulge inside the budget: take the freshest feasible point
            chosen = int(np.argmax(np.where(feasible, frame[f_col].to_numpy(), -np.inf)))
    else:
        chosen = base_idx
    row = frame.iloc[chosen]
    base = frame.iloc[base_idx]
    knee = KneePoint(
        alpha=float(row["alpha"]),
        ndcg=float(row[n_col]),
        freshness=float(row[f_col]),
        ndcg_base=float(base[n_col]),
        freshness_base=float(base[f_col]),
        ndcg_drop_rel=float(row["ndcg_drop_rel"]),
        freshness_gain_rel=float(row["freshness_gain_rel"]),
        chord_distance=float(np.nan_to_num(row["chord_distance"], nan=0.0)),
        max_rel_drop=float(max_rel_drop),
        unconstrained_alpha=float(frame.iloc[unconstrained]["alpha"]),
    )
    logger.info(
        "knee alpha=%.3f: NDCG drop %.2f%%, freshness gain %.2f%% (unconstrained knee alpha=%.3f)",
        knee.alpha,
        100 * knee.ndcg_drop_rel,
        100 * knee.freshness_gain_rel,
        knee.unconstrained_alpha,
    )
    return knee


def point_at_alpha(frame: pd.DataFrame, alpha: float) -> pd.Series:
    """Row of a sweep closest to ``alpha``."""
    return frame.iloc[int(np.argmin(np.abs(frame["alpha"].to_numpy() - alpha)))]
