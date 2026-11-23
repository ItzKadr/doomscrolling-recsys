r"""Position-bias propensities, clipped IPS weights and counterfactual estimators.

Propensity estimation: result randomization
-------------------------------------------
Under the examination hypothesis ``P(click | k, d) = p_k * gamma(d)``. On the
randomized slice of traffic every shown list is a uniform random permutation,
so the item shown at position ``k`` is a uniform draw from the session's list,
independently of ``k``. Hence

.. math::  \mathbb{E}[\text{CTR}_k] = p_k \, \mathbb{E}[\gamma(d)]
           \;\Rightarrow\; \hat p_k = \text{CTR}_k / \text{CTR}_1,

identifying ``p_k`` up to the convention ``p_1 = 1`` (the top slot is always
examined). Confidence intervals come from a session-level bootstrap
(multinomial resampling, fully vectorized).

Clipped IPS
-----------
``w_i = 1 / max(p_hat[k_i], tau)``. Without clipping IPS is unbiased but the
weight of a click at a rarely examined slot explodes (variance ~ 1/p_k); the
clip at ``tau`` caps the weight at ``1/tau`` and trades a small, known bias for
a large variance reduction.

Counterfactual offline evaluation from logged clicks
----------------------------------------------------
For a new ranking ``pi`` of a logged list (rank ``r_pi(i)``) and clicked set
``C_s``::

    IPS-DCG@k(pi; s) = sum_{i in C_s} w_i * 1[r_pi(i) <= k] / log2(1 + r_pi(i))

Because ``E[click_i] = p_{k_i} gamma(d_i)``, with ``p_hat = p >= tau`` its
expectation is the DCG@k of ``pi`` with gains ``gamma(d_i)``: an unbiased
estimate of ranking quality obtained from biased logs.

* ``IPS-NDCG@k`` (plain, Horvitz-Thompson style): per session divide by the
  ideal DCG of the session's clicks with *unit* gains, average over sessions
  with at least one click. The importance weights are not normalized, so a
  session can score above 1 and the estimator has the higher variance.
* ``SNIPS-NDCG@k`` (self-normalized, Hajek style): ratio of totals
  ``sum_s IPS-DCG@k(pi; s) / sum_s IPS-IDCG@k(s)``, where IPS-IDCG is the best
  achievable IPS-DCG (clicked items sorted by weight). Weights appear in both
  numerator and denominator, the estimate is bounded in [0, 1], invariant to a
  global rescaling of the weights and has lower variance; the price is a small
  ratio-estimator bias.

A naive click-NDCG (clicks as labels, no weights) is provided for contrast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import (
    dcg_at_k,
    discounts,
    ndcg_at_k,
    offsets_from_sizes,
    ranks_within_groups,
)

logger = logging.getLogger(__name__)

DEFAULT_TAU: float = 0.15


@dataclass
class PropensityEstimate:
    """Estimated examination probabilities per position (1-based positions)."""

    positions: np.ndarray
    p_hat: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ctr: np.ndarray  # raw CTR per position on the randomized slice
    n_sessions: int
    eta_hat: float  # power-law fit p_k ~ k^-eta (diagnostic)
    method: str = "result_randomization"

    def to_frame(self, true_propensities: np.ndarray | None = None) -> pd.DataFrame:
        """Tidy table (one row per position)."""
        frame = pd.DataFrame(
            {
                "position": self.positions,
                "ctr_randomized": self.ctr,
                "p_hat": self.p_hat,
                "p_hat_ci_low": self.ci_low,
                "p_hat_ci_high": self.ci_high,
            }
        )
        if true_propensities is not None:
            frame["p_true"] = np.asarray(true_propensities, dtype=float)[: len(frame)]
        return frame


def _click_matrix(
    positions: np.ndarray, clicks: np.ndarray, sizes: np.ndarray, n_positions: int
) -> tuple[np.ndarray, np.ndarray]:
    """(sessions x positions) click and impression count matrices."""
    sizes = np.asarray(sizes, dtype=np.int64)
    n_groups = sizes.size
    rows = np.repeat(np.arange(n_groups), sizes)
    cols = np.asarray(positions, dtype=np.int64) - 1
    if cols.min(initial=0) < 0 or cols.max(initial=0) >= n_positions:
        raise ValueError("positions must lie in 1..n_positions")
    clicks_m = np.zeros((n_groups, n_positions))
    shown_m = np.zeros((n_groups, n_positions))
    np.add.at(clicks_m, (rows, cols), np.asarray(clicks, dtype=float))
    np.add.at(shown_m, (rows, cols), 1.0)
    return clicks_m, shown_m


def fit_power_law(p_hat: np.ndarray) -> float:
    """Least-squares ``eta`` in ``log p_k = -eta log k`` (through the origin, k >= 2)."""
    k = np.arange(1, p_hat.size + 1, dtype=float)
    valid = (k >= 2) & (p_hat > 0)
    if not np.any(valid):
        return float("nan")
    x = np.log(k[valid])
    y = np.log(p_hat[valid])
    return float(-(x @ y) / (x @ x))


def estimate_propensities(
    positions: np.ndarray,
    clicks: np.ndarray,
    sizes: np.ndarray,
    n_positions: int,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 0,
) -> PropensityEstimate:
    r"""Estimate ``p_k`` from uniformly randomized sessions (``p_hat_1 = 1``).

    ``p_hat_k = CTR_k / CTR_1``. The bootstrap resamples *sessions* with a
    multinomial count vector per replicate, so ``B`` replicates are a single
    matrix product ``(B x S) @ (S x K)``.
    """
    clicks_m, shown_m = _click_matrix(positions, clicks, sizes, n_positions)
    n_sessions = clicks_m.shape[0]
    total_clicks = clicks_m.sum(axis=0)
    total_shown = np.maximum(shown_m.sum(axis=0), 1.0)
    ctr = total_clicks / total_shown
    if ctr[0] <= 0:
        raise ValueError("no clicks at position 1 in the randomized slice")
    p_hat = ctr / ctr[0]

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_sessions, np.full(n_sessions, 1.0 / n_sessions), size=n_boot)
    boot_ctr = (counts @ clicks_m) / np.maximum(counts @ shown_m, 1.0)
    boot_p = boot_ctr / np.maximum(boot_ctr[:, :1], 1e-12)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_p, [alpha, 1.0 - alpha], axis=0)
    eta_hat = fit_power_law(p_hat)
    logger.info("propensities from %d randomized sessions, eta_hat=%.3f", n_sessions, eta_hat)
    return PropensityEstimate(
        positions=np.arange(1, n_positions + 1),
        p_hat=p_hat,
        ci_low=lo,
        ci_high=hi,
        ctr=ctr,
        n_sessions=n_sessions,
        eta_hat=eta_hat,
    )


def ips_weights(positions: np.ndarray, p_hat: np.ndarray, tau: float = DEFAULT_TAU) -> np.ndarray:
    """Clipped inverse propensity weights ``1 / max(p_hat[k - 1], tau)``."""
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    p = np.asarray(p_hat, dtype=float)[np.asarray(positions, dtype=np.int64) - 1]
    return 1.0 / np.maximum(p, tau)


def ips_dcg_per_session(
    scores: np.ndarray,
    clicks: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    p_hat: np.ndarray,
    k: int = 5,
    tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """IPS-DCG@k of the ranking induced by ``scores`` for every logged session."""
    w = ips_weights(positions, p_hat, tau) * np.asarray(clicks, dtype=float)
    ranks = ranks_within_groups(scores, sizes)
    contrib = np.where(ranks <= k, w / np.log2(ranks + 1.0), 0.0)
    return np.add.reduceat(contrib, offsets_from_sizes(sizes))


def ips_ideal_dcg_per_session(
    clicks: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    p_hat: np.ndarray,
    k: int = 5,
    tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """Largest achievable IPS-DCG@k per session (clicked items sorted by weight)."""
    w = ips_weights(positions, p_hat, tau) * np.asarray(clicks, dtype=float)
    return dcg_at_k(w, w, sizes, k, gain="linear")


def _unit_click_idcg(clicks: np.ndarray, sizes: np.ndarray, k: int) -> np.ndarray:
    n_clicks = np.add.reduceat(np.asarray(clicks, dtype=float), offsets_from_sizes(sizes))
    width = int(np.max(sizes))
    cum = np.concatenate(([0.0], np.cumsum(discounts(width))))
    return cum[np.minimum(n_clicks.astype(np.int64), k)]


def ips_ndcg_per_session(
    scores: np.ndarray,
    clicks: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    p_hat: np.ndarray,
    k: int = 5,
    tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """Plain IPS-NDCG@k per session (NaN for sessions without clicks)."""
    num = ips_dcg_per_session(scores, clicks, positions, sizes, p_hat, k, tau)
    den = _unit_click_idcg(clicks, sizes, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def ips_ndcg_at_k(
    scores: np.ndarray,
    clicks: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    p_hat: np.ndarray,
    k: int = 5,
    tau: float = DEFAULT_TAU,
) -> float:
    """Plain IPS-NDCG@k: mean over sessions with >= 1 click (see module docstring)."""
    per = ips_ndcg_per_session(scores, clicks, positions, sizes, p_hat, k, tau)
    return float(np.nanmean(per)) if np.any(~np.isnan(per)) else float("nan")


def snips_ndcg_at_k(
    scores: np.ndarray,
    clicks: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    p_hat: np.ndarray,
    k: int = 5,
    tau: float = DEFAULT_TAU,
) -> float:
    """Self-normalized IPS-NDCG@k: ``sum_s IPS-DCG / sum_s IPS-IDCG`` in [0, 1]."""
    num = ips_dcg_per_session(scores, clicks, positions, sizes, p_hat, k, tau).sum()
    den = ips_ideal_dcg_per_session(clicks, positions, sizes, p_hat, k, tau).sum()
    return float(num / den) if den > 0 else float("nan")


def click_ndcg_at_k(scores: np.ndarray, clicks: np.ndarray, sizes: np.ndarray, k: int = 5) -> float:
    """Naive NDCG@k with raw clicks as binary labels (biased towards the logger)."""
    per = ndcg_at_k(clicks, scores, sizes, k, gain="linear")
    return float(np.nanmean(per)) if np.any(~np.isnan(per)) else float("nan")


def ctr_by_position(positions: np.ndarray, clicks: np.ndarray, n_positions: int) -> np.ndarray:
    """Empirical CTR per position (any traffic)."""
    pos = np.asarray(positions, dtype=np.int64) - 1
    shown = np.bincount(pos, minlength=n_positions).astype(float)
    clicked = np.bincount(pos, weights=np.asarray(clicks, dtype=float), minlength=n_positions)
    return clicked / np.maximum(shown, 1.0)


def clicks_per_position_matrix(
    positions: np.ndarray, clicks: np.ndarray, sizes: np.ndarray, n_positions: int
) -> np.ndarray:
    """(sessions x positions) click matrix; exposed for diagnostics and tests."""
    return _click_matrix(positions, clicks, sizes, n_positions)[0]


__all__ = [
    "DEFAULT_TAU",
    "PropensityEstimate",
    "estimate_propensities",
    "fit_power_law",
    "ips_weights",
    "ips_dcg_per_session",
    "ips_ideal_dcg_per_session",
    "ips_ndcg_per_session",
    "ips_ndcg_at_k",
    "snips_ndcg_at_k",
    "click_ndcg_at_k",
    "ctr_by_position",
    "clicks_per_position_matrix",
]
