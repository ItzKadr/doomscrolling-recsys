"""Z-scores, MOO scalarization, Pareto filtering and knee selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import ranks_within_groups
from src.pareto import chord_distances, moo_scores, pareto_mask, select_knee, sweep_alpha, zscore_within_groups


def test_zscore_within_groups() -> None:
    x = np.array([1.0, 2.0, 3.0, 10.0, 10.0, 5.0, 7.0])
    sizes = np.array([3, 2, 2])
    z = zscore_within_groups(x, sizes)
    np.testing.assert_allclose(z[:3], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    np.testing.assert_allclose(z[3:5], [0.0, 0.0])  # constant group -> 0
    np.testing.assert_allclose(z[5:], [-1.0, 1.0])
    rng = np.random.default_rng(0)
    big = rng.normal(5, 3, size=1000)
    zz = zscore_within_groups(big, np.full(100, 10)).reshape(100, 10)
    np.testing.assert_allclose(zz.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(zz.std(axis=1), 1.0, atol=1e-12)


def test_moo_extremes() -> None:
    rng = np.random.default_rng(1)
    sizes = np.full(20, 8)
    rel, fresh = rng.normal(size=160), rng.random(160)
    np.testing.assert_array_equal(
        ranks_within_groups(moo_scores(rel, fresh, sizes, 0.0), sizes), ranks_within_groups(rel, sizes)
    )
    np.testing.assert_array_equal(
        ranks_within_groups(moo_scores(rel, fresh, sizes, 1.0), sizes), ranks_within_groups(fresh, sizes)
    )
    with pytest.raises(ValueError):
        moo_scores(rel, fresh, sizes, 1.5)


def test_pareto_mask_hand_example() -> None:
    pts = np.array([[0.9, 0.1], [0.8, 0.5], [0.7, 0.4], [0.5, 0.9], [0.5, 0.8]])
    # (0.7, 0.4) is dominated by (0.8, 0.5); (0.5, 0.8) by (0.5, 0.9)
    np.testing.assert_array_equal(pareto_mask(pts), [True, True, False, True, False])
    # minimizing the second objective flips the answer
    np.testing.assert_array_equal(pareto_mask(pts, (True, False)), [True, False, False, False, False])


def test_chord_distance_hand_example() -> None:
    # extremes A=(fresh 0, ndcg 1) and B=(fresh 1, ndcg 0) in normalized space;
    # the middle point (0.5, 0.9) lies sqrt(2)*0.4/... above the chord x + y = 1
    ndcg = np.array([1.0, 0.9, 0.0])
    fresh = np.array([0.0, 0.5, 1.0])
    d = chord_distances(ndcg, fresh, np.array([True, True, True]))
    assert d[0] == pytest.approx(0.0) and d[2] == pytest.approx(0.0)
    assert d[1] == pytest.approx((0.5 + 0.9 - 1.0) / np.sqrt(2))


def _toy_front() -> pd.DataFrame:
    alphas = np.linspace(0, 1, 11)
    ndcg = 0.8 - 0.3 * alphas**2  # concave trade-off
    fresh = 0.4 + 0.3 * np.sqrt(alphas)
    frame = pd.DataFrame({"alpha": alphas, "ndcg@5": ndcg, "freshness@5": fresh})
    frame["ndcg_drop_rel"] = 1 - frame["ndcg@5"] / frame["ndcg@5"].iloc[0]
    frame["freshness_gain_rel"] = frame["freshness@5"] / frame["freshness@5"].iloc[0] - 1
    front = pareto_mask(frame[["ndcg@5", "freshness@5"]].to_numpy())
    frame["is_pareto"] = front
    frame["chord_distance"] = chord_distances(frame["ndcg@5"].to_numpy(), frame["freshness@5"].to_numpy(), front)
    return frame


def test_knee_respects_guard_rail() -> None:
    frame = _toy_front()
    loose = select_knee(frame, max_rel_drop=1.0)
    tight = select_knee(frame, max_rel_drop=0.02)
    assert tight.ndcg_drop_rel <= 0.02 + 1e-12
    assert tight.alpha <= loose.alpha
    assert loose.alpha == pytest.approx(loose.unconstrained_alpha)
    assert tight.freshness_gain_rel > 0
    none = select_knee(frame, max_rel_drop=0.0)
    assert none.alpha == 0.0


def test_sweep_alpha_endpoints() -> None:
    rng = np.random.default_rng(3)
    sizes = np.full(200, 10)
    rel = rng.integers(0, 5, size=2000).astype(float)
    fresh = rng.random(2000)
    rel_hat = rel + rng.normal(0, 0.5, size=2000)
    frame = sweep_alpha(rel_hat, fresh, rel, sizes, np.linspace(0, 1, 11), k=5)
    assert frame["ndcg@5"].iloc[0] > frame["ndcg@5"].iloc[-1]
    assert frame["freshness@5"].iloc[-1] > frame["freshness@5"].iloc[0]
    assert frame["ndcg_drop_rel"].iloc[0] == pytest.approx(0.0)
    assert frame["is_pareto"].iloc[0] and frame["is_pareto"].iloc[-1]
