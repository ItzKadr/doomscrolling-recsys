"""Propensity estimation, clipped IPS weights and counterfactual estimators."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset import SimConfig, group_sizes, simulate
from src.ips import (
    click_ndcg_at_k,
    ctr_by_position,
    estimate_propensities,
    fit_power_law,
    ips_dcg_per_session,
    ips_ideal_dcg_per_session,
    ips_ndcg_at_k,
    ips_weights,
    snips_ndcg_at_k,
)
from src.metrics import dcg_at_k


def test_propensities_recovered_from_randomized_slice() -> None:
    cfg = SimConfig(n_sessions=20_000, rand_frac=0.5, val_frac=0.1, test_frac=0.1, eta=1.0, seed=3)
    rand = simulate(cfg).split("rand")
    est = estimate_propensities(
        rand["position"].to_numpy(), rand["click"].to_numpy(), group_sizes(rand), cfg.list_size, n_boot=100
    )
    p_true = cfg.true_propensities()
    assert est.p_hat[0] == pytest.approx(1.0)
    assert np.max(np.abs(est.p_hat - p_true)) < 0.05
    assert est.eta_hat == pytest.approx(cfg.eta, abs=0.1)
    # the bootstrap CI should cover the truth at most positions
    covered = (est.ci_low <= p_true + 1e-9) & (p_true - 1e-9 <= est.ci_high)
    assert covered.mean() >= 0.8


def test_logged_ctr_curve_is_confounded() -> None:
    """Logged CTR drops faster than p_k: the old ranker puts better items on top."""
    cfg = SimConfig(n_sessions=6000, seed=5)
    logs = simulate(cfg)
    tr = logs.split("train")
    ctr = ctr_by_position(tr["position"].to_numpy(), tr["click"].to_numpy(), cfg.list_size)
    naive_ratio = ctr / ctr[0]
    assert naive_ratio[-1] < cfg.true_propensities()[-1]


def test_fit_power_law_exact() -> None:
    k = np.arange(1, 11, dtype=float)
    assert fit_power_law(k**-1.7) == pytest.approx(1.7)


def test_ips_weights_clipping() -> None:
    p_hat = np.array([1.0, 0.5, 0.2, 0.1])
    w = ips_weights(np.array([1, 2, 3, 4]), p_hat, tau=0.15)
    np.testing.assert_allclose(w, [1.0, 2.0, 5.0, 1 / 0.15])
    assert np.all(w <= 1 / 0.15 + 1e-12)
    with pytest.raises(ValueError):
        ips_weights(np.array([1]), p_hat, tau=0.0)


def test_ips_dcg_hand_computed() -> None:
    # one session, logged positions 1..4, clicks at logged positions 2 and 4
    p_hat = np.array([1.0, 0.5, 0.25, 0.2])
    positions = np.array([1, 2, 3, 4])
    clicks = np.array([0, 1, 0, 1])
    sizes = np.array([4])
    # new ranking puts logged item 4 first, item 2 third
    scores = np.array([0.5, 0.3, 0.1, 0.9])
    w2, w4 = 1 / 0.5, 1 / 0.2
    expected = w4 / np.log2(2) + w2 / np.log2(4)
    got = ips_dcg_per_session(scores, clicks, positions, sizes, p_hat, k=5, tau=0.1)[0]
    assert got == pytest.approx(expected)
    ideal = ips_ideal_dcg_per_session(clicks, positions, sizes, p_hat, k=5, tau=0.1)[0]
    assert ideal == pytest.approx(w4 + w2 / np.log2(3))
    # k=1 keeps only the top item
    assert ips_dcg_per_session(scores, clicks, positions, sizes, p_hat, k=1, tau=0.1)[0] == pytest.approx(w4)


def test_ips_dcg_is_unbiased_monte_carlo() -> None:
    """E[IPS-DCG] equals the DCG with gains gamma(rel) when p_hat = p >= tau."""
    rng = np.random.default_rng(0)
    K, S = 6, 40_000
    p = np.arange(1, K + 1, dtype=float) ** -1.0
    gamma_one = np.array([0.9, 0.1, 0.6, 0.05, 0.7, 0.3])  # one fixed list, logged order
    gamma = np.tile(gamma_one, S)
    positions = np.tile(np.arange(1, K + 1), S)
    clicks = (rng.random(S * K) < np.repeat(p[None, :], S, axis=0).ravel() * gamma).astype(int)
    sizes = np.full(S, K)
    new_order_scores = np.tile(gamma_one + 0.01 * np.arange(K), S)  # some new ranking
    est = ips_dcg_per_session(new_order_scores, clicks, positions, sizes, p, k=3, tau=0.1).mean()
    truth = dcg_at_k(gamma_one, gamma_one + 0.01 * np.arange(K), np.array([K]), 3, gain="linear")[0]
    assert est == pytest.approx(truth, rel=0.03)
    # the naive (unweighted) version is clearly biased downwards
    naive = ips_dcg_per_session(new_order_scores, clicks, positions, sizes, np.ones(K), k=3, tau=0.1).mean()
    assert naive < 0.8 * truth


def test_snips_bounded_and_scale_invariant() -> None:
    rng = np.random.default_rng(1)
    sizes = np.full(300, 5)
    n = int(sizes.sum())
    positions = np.tile(np.arange(1, 6), 300)
    clicks = (rng.random(n) < 0.3).astype(int)
    scores = rng.normal(size=n)
    p_hat = np.array([1.0, 0.6, 0.4, 0.3, 0.2])
    sn = snips_ndcg_at_k(scores, clicks, positions, sizes, p_hat, 5, tau=0.05)
    assert 0.0 <= sn <= 1.0
    # rescaling all propensities by c (weights by 1/c) leaves SNIPS unchanged
    sn_scaled = snips_ndcg_at_k(scores, clicks, positions, sizes, p_hat * 0.5, 5, tau=0.01)
    assert sn_scaled == pytest.approx(sn)
    # plain IPS-NDCG is not scale invariant
    ips = ips_ndcg_at_k(scores, clicks, positions, sizes, p_hat, 5, tau=0.05)
    ips_scaled = ips_ndcg_at_k(scores, clicks, positions, sizes, p_hat * 0.5, 5, tau=0.01)
    assert ips_scaled == pytest.approx(2 * ips)


def test_click_ndcg_prefers_logger_order() -> None:
    """With strong position bias the logged order wins the naive click metric."""
    rng = np.random.default_rng(2)
    S, K = 3000, 5
    p = np.arange(1, K + 1, dtype=float) ** -1.5
    gamma = rng.random((S, K))
    clicks = (rng.random((S, K)) < p * gamma).astype(int).ravel()
    sizes = np.full(S, K)
    logger_scores = -np.tile(np.arange(1, K + 1), S).astype(float)
    oracle_scores = gamma.ravel()
    assert click_ndcg_at_k(logger_scores, clicks, sizes) > click_ndcg_at_k(oracle_scores, clicks, sizes)
    positions = np.tile(np.arange(1, K + 1), S)
    # IPS evaluation reverses the verdict
    assert snips_ndcg_at_k(oracle_scores, clicks, positions, sizes, p, 5, tau=0.01) > snips_ndcg_at_k(
        logger_scores, clicks, positions, sizes, p, 5, tau=0.01
    )
