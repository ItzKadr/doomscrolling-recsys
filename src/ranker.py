r"""LightGBM LambdaMART rankers trained on biased clicks: naive vs IPS-weighted.

How LightGBM uses ``sample_weight`` in ``lambdarank``
-----------------------------------------------------
For every query LightGBM first computes the LambdaRank gradient ``lambda_i`` and
hessian ``h_i`` of each row, then multiplies both by that row's weight
(``rank_objective.hpp``: ``gradients[i] *= weights[i]``). Weights are therefore
*per row*, while the unbiased pairwise IPS objective (Joachims et al., 2017)

.. math::  L = \sum_{i \in \text{clicked}} \frac{1}{p_{k_i}}
           \sum_{j \notin \text{clicked}} \ell(s_i - s_j)

needs a weight per *pair* that depends on the clicked item ``i`` only.

Weighting schemes (``scheme`` argument)
---------------------------------------
``"pair"`` (default, exact): *click splitting*. A session with ``m`` clicks is
    split into ``m`` pseudo-queries; each contains one clicked row (label 1) and
    all non-clicked rows (label 0) and every row of it carries the clicked
    item's clipped IPS weight ``w_i = 1/max(p_hat[k_i], tau)``. Every
    (clicked, non-clicked) pair then receives exactly the weight ``w_i`` and the
    sum over pseudo-queries reproduces the pairwise IPS loss above.
``"clicked"``: weight ``w_i`` on clicked rows, 1 on non-clicked rows. Only the
    "push up" half of every pair is reweighted (partial correction).
``"row"``: weight ``w_i`` on every row according to its own position. This is
    the literal per-row reading of IPS; it *amplifies* the push-down of
    non-clicked items at low positions and is kept as an ablation.

Sessions without clicks carry no pairwise signal and are dropped for all
models. Rows are shuffled within each query before training: at iteration 0
all scores tie and LightGBM's stable sort would otherwise rank rows in logged
order, leaking the logged position into the first trees' |Delta NDCG| terms.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Literal

import lightgbm as lgb
import numpy as np

from src.ips import DEFAULT_TAU, ips_weights
from src.metrics import offsets_from_sizes

logger = logging.getLogger(__name__)

Scheme = Literal["none", "pair", "clicked", "row"]
SCHEMES: tuple[str, ...] = ("none", "pair", "clicked", "row")


@dataclass(frozen=True)
class RankerConfig:
    """Hyper-parameters of ``lightgbm.LGBMRanker`` (objective = lambdarank)."""

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 300  # IPS weights inflate variance: keep leaves large
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    lambdarank_truncation_level: int = 10
    n_jobs: int = 4
    seed: int = 42

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainingSet:
    """Rows (indices into the source split) grouped into LightGBM queries."""

    row_index: np.ndarray  # indices into the source arrays, query-contiguous
    labels: np.ndarray
    sizes: np.ndarray  # query sizes
    weights: np.ndarray | None  # per-row sample weights (None = unweighted)
    scheme: str

    @property
    def n_queries(self) -> int:
        return int(self.sizes.size)


def make_ranker(cfg: RankerConfig) -> lgb.LGBMRanker:
    """Configured LambdaMART model (deterministic, thread-capped)."""
    return lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        lambdarank_truncation_level=cfg.lambdarank_truncation_level,
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


def _session_ids(sizes: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(sizes.size), sizes)


def _shuffle_within_groups(
    group_of_row: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permutation that keeps groups contiguous (in group order) but shuffles inside."""
    return np.lexsort((rng.random(group_of_row.size), group_of_row))


def build_training_set(
    clicks: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    p_hat: np.ndarray | None = None,
    tau: float = DEFAULT_TAU,
    scheme: Scheme = "pair",
    seed: int = 0,
) -> TrainingSet:
    """Turn a logged split into LightGBM queries, labels and IPS weights.

    ``scheme="none"`` is the naive model (clicks as labels, no weights).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    clicks = np.asarray(clicks, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    sess = _session_ids(sizes)
    n_clicks = np.add.reduceat(clicks, offsets_from_sizes(sizes))
    has_click = n_clicks[sess] > 0

    if scheme != "none" and p_hat is None:
        raise ValueError("p_hat is required for IPS schemes")
    w_all = ips_weights(positions, p_hat, tau) if p_hat is not None else None

    if scheme in ("none", "clicked", "row"):
        keep = np.flatnonzero(has_click)
        perm = _shuffle_within_groups(sess[keep], rng)
        row_index = keep[perm]
        _, new_sizes = np.unique(sess[row_index], return_counts=True)
        labels = clicks[row_index]
        if scheme == "none":
            weights = None
        elif scheme == "row":
            weights = w_all[row_index]
        else:  # clicked
            weights = np.where(labels == 1, w_all[row_index], 1.0)
        return TrainingSet(row_index, labels, new_sizes.astype(np.int64), weights, scheme)

    # ---- scheme == "pair": click splitting --------------------------------------
    click_rows = np.flatnonzero(clicks == 1)  # one pseudo-query per click event
    neg_rows = np.flatnonzero(clicks == 0)
    neg_sizes = np.bincount(sess[neg_rows], minlength=sizes.size)
    neg_offsets = offsets_from_sizes(neg_sizes) if neg_sizes.size else np.zeros(0, np.int64)
    ev_sess = sess[click_rows]
    ev_neg = neg_sizes[ev_sess]
    n_events = click_rows.size
    # negatives of every event: neg_rows[neg_offsets[s] + j], j < ev_neg
    total_neg = int(ev_neg.sum())
    ev_of_neg = np.repeat(np.arange(n_events), ev_neg)
    if total_neg:
        j = np.arange(total_neg) - np.repeat(offsets_from_sizes(ev_neg), ev_neg)  # rank inside event
        neg_part = neg_rows[neg_offsets[ev_sess][ev_of_neg] + j]
    else:
        neg_part = np.zeros(0, dtype=np.int64)

    event_of_row = np.concatenate((np.arange(n_events), ev_of_neg))
    row_index_unsorted = np.concatenate((click_rows, neg_part))
    perm = _shuffle_within_groups(event_of_row, rng)
    row_index = row_index_unsorted[perm]
    event_sorted = event_of_row[perm]
    labels = clicks[row_index]
    weights = w_all[click_rows][event_sorted]
    new_sizes = (ev_neg + 1).astype(np.int64)
    return TrainingSet(row_index, labels, new_sizes, weights, scheme)


def fit_ranker(
    X: np.ndarray,
    ts: TrainingSet,
    cfg: RankerConfig,
    feature_names: list[str] | None = None,
) -> lgb.LGBMRanker:
    """Fit LambdaMART on the rows/queries/weights described by ``ts``."""
    model = make_ranker(cfg)
    kwargs: dict[str, object] = {"group": ts.sizes}
    if ts.weights is not None:
        kwargs["sample_weight"] = ts.weights
    if feature_names is not None:
        kwargs["feature_name"] = feature_names
    model.fit(X[ts.row_index], ts.labels, **kwargs)
    logger.info(
        "fitted %s ranker: %d queries, %d rows, mean weight=%.3f",
        ts.scheme,
        ts.n_queries,
        ts.row_index.size,
        float(np.mean(ts.weights)) if ts.weights is not None else 1.0,
    )
    return model


def predict(model: lgb.LGBMRanker, X: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
    """Raw ranking scores (optionally from the first ``num_iteration`` trees).

    Goes through the underlying ``Booster`` (identical output for lambdarank,
    which has no output transform) to skip sklearn's feature-name validation.
    """
    return np.asarray(
        model.booster_.predict(X, num_iteration=num_iteration, raw_score=True), dtype=float
    )


def feature_importance(
    model: lgb.LGBMRanker, feature_names: list[str], iteration: int | None = None
) -> dict[str, float]:
    """Normalized gain importance (sums to 1) of the first ``iteration`` trees."""
    gain = model.booster_.feature_importance(importance_type="gain", iteration=iteration).astype(float)
    total = gain.sum()
    shares = gain / total if total > 0 else gain
    return dict(zip(feature_names, shares.tolist()))
