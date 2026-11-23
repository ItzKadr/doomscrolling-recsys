"""Training-set construction and the effect of sample weights on LambdaMART."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset import SimConfig, feature_matrix, group_sizes, simulate
from src.ips import ips_weights
from src.metrics import ndcg_at_k
from src.ranker import RankerConfig, TrainingSet, build_training_set, fit_ranker, predict

TINY = RankerConfig(n_estimators=40, learning_rate=0.1, num_leaves=7, min_child_samples=20, n_jobs=2)


@pytest.fixture(scope="module")
def train_split():
    logs = simulate(SimConfig(n_sessions=3000, seed=21))
    tr = logs.split("train")
    return tr, feature_matrix(tr), group_sizes(tr), logs.true_propensities


def test_pair_scheme_structure() -> None:
    # session 0: clicks at rows 1, 3 ; session 1: no click ; session 2: click at row 6
    clicks = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    positions = np.array([1, 2, 3, 4, 1, 2, 1, 2])
    sizes = np.array([4, 2, 2])
    p_hat = np.array([1.0, 0.5, 0.25, 0.2])
    ts = build_training_set(clicks, positions, sizes, p_hat, tau=0.15, scheme="pair", seed=0)
    # 3 click events -> 3 pseudo-queries; sizes = 1 click + negatives of its session
    np.testing.assert_array_equal(np.sort(ts.sizes), [2, 3, 3])
    assert ts.labels.sum() == 3
    starts = np.concatenate(([0], np.cumsum(ts.sizes)[:-1]))
    for start, size in zip(starts, ts.sizes):
        lab = ts.labels[start : start + size]
        w = ts.weights[start : start + size]
        rows = ts.row_index[start : start + size]
        assert lab.sum() == 1  # exactly one positive per pseudo-query
        assert np.allclose(w, w[0])  # constant weight inside the query
        pos_row = rows[lab == 1][0]
        assert w[0] == pytest.approx(ips_weights(positions[[pos_row]], p_hat, 0.15)[0])
        # the other clicked row of the same session is not used as a negative
        assert not np.any(clicks[rows[lab == 0]] == 1)


def test_naive_and_row_schemes_drop_clickless_sessions() -> None:
    clicks = np.array([0, 1, 0, 0, 0, 1])
    positions = np.array([1, 2, 1, 2, 1, 2])
    sizes = np.array([2, 2, 2])
    p_hat = np.array([1.0, 0.1])
    naive = build_training_set(clicks, positions, sizes, None, scheme="none")
    assert naive.weights is None and naive.n_queries == 2
    row = build_training_set(clicks, positions, sizes, p_hat, tau=0.15, scheme="row")
    np.testing.assert_allclose(np.sort(row.weights), [1.0, 1.0, 1 / 0.15, 1 / 0.15])
    clicked = build_training_set(clicks, positions, sizes, p_hat, tau=0.15, scheme="clicked")
    assert np.allclose(clicked.weights[clicked.labels == 0], 1.0)
    with pytest.raises(ValueError):
        build_training_set(clicks, positions, sizes, None, scheme="pair")


def test_unit_weights_equal_no_weights_and_ips_weights_change_model(train_split) -> None:
    tr, X, sizes, p_true = train_split
    clicks, positions = tr["click"].to_numpy(), tr["position"].to_numpy()
    naive_ts = build_training_set(clicks, positions, sizes, None, scheme="none", seed=1)
    unit_ts = TrainingSet(naive_ts.row_index, naive_ts.labels, naive_ts.sizes, np.ones(naive_ts.row_index.size), "unit")
    ips_ts = build_training_set(clicks, positions, sizes, p_true, tau=0.15, scheme="pair", seed=1)
    s_naive = predict(fit_ranker(X, naive_ts, TINY), X)
    s_unit = predict(fit_ranker(X, unit_ts, TINY), X)
    s_ips = predict(fit_ranker(X, ips_ts, TINY), X)
    np.testing.assert_allclose(s_unit, s_naive, atol=1e-10)
    assert np.max(np.abs(s_ips - s_naive)) > 1e-2


def test_training_improves_ranking_quality(train_split) -> None:
    """A tiny overfit check: more boosting rounds -> better train NDCG on clicks."""
    tr, X, sizes, p_true = train_split
    ts = build_training_set(tr["click"].to_numpy(), tr["position"].to_numpy(), sizes, p_true, scheme="pair")
    model = fit_ranker(X, ts, TINY)
    rel = tr["relevance"].to_numpy(dtype=float)
    early = np.nanmean(ndcg_at_k(rel, predict(model, X, num_iteration=1), sizes, 5))
    late = np.nanmean(ndcg_at_k(rel, predict(model, X), sizes, 5))
    assert late > early
    random = np.nanmean(ndcg_at_k(rel, np.random.default_rng(0).random(rel.size), sizes, 5))
    assert late > random + 0.15
