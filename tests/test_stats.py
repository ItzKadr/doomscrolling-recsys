"""Offline A/B testing: bootstrap, t-test, Wilcoxon."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from src.stats import ABTestResult, ab_test, paired_bootstrap


def test_detects_known_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.70, 0.1, size=3000)
    b = a + 0.02 + rng.normal(0, 0.05, size=3000)
    res = ab_test(a, b, "ndcg@5", "naive", "ips", n_boot=2000, seed=1)
    assert isinstance(res, ABTestResult)
    assert res.ci_low < 0.02 < res.ci_high
    assert res.significant and res.p_bootstrap < 0.01
    assert res.t_pvalue < 1e-10 and res.wilcoxon_pvalue < 1e-10
    assert res.rel_lift == pytest.approx(b.mean() / a.mean() - 1)
    assert res.rel_lift_ci_low < res.rel_lift < res.rel_lift_ci_high
    assert res.boot_diffs.shape == (2000,)
    t = sps.ttest_rel(b, a)
    assert res.t_stat == pytest.approx(t.statistic)


def test_no_difference() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    res = ab_test(a, a.copy(), "m")
    assert res.mean_diff == 0.0 and res.t_pvalue == 1.0 and res.wilcoxon_pvalue == 1.0
    assert res.p_bootstrap == 1.0 and not res.significant


def test_null_is_not_significant_most_of_the_time() -> None:
    rng = np.random.default_rng(2)
    rejections = 0
    for seed in range(20):
        a = rng.normal(size=400)
        b = a + rng.normal(0, 1, size=400)
        rejections += ab_test(a, b, "m", n_boot=500, seed=seed).p_bootstrap < 0.05
    assert rejections <= 4  # ~5% expected


def test_nan_pairs_dropped_and_dict() -> None:
    a = np.array([0.5, np.nan, 0.7, 0.6, 0.4, 0.8])
    b = np.array([0.6, 0.9, np.nan, 0.65, 0.5, 0.85])
    res = ab_test(a, b, "m", n_boot=200)
    assert res.n == 4
    d = res.to_dict()
    assert "boot_diffs" not in d and d["n"] == 4 and isinstance(d["significant"], bool)
    assert len(res.to_dict(include_boot=True)["boot_diffs"]) == 200


def test_paired_bootstrap_shapes_and_mean() -> None:
    rng = np.random.default_rng(3)
    a = rng.random(1000)
    b = a + 0.1
    diffs, lifts = paired_bootstrap(a, b, n_boot=300, seed=0, chunk=64)
    assert diffs.shape == lifts.shape == (300,)
    np.testing.assert_allclose(diffs, 0.1, atol=1e-12)  # constant paired shift


def test_rejects_unpaired() -> None:
    with pytest.raises(ValueError):
        ab_test(np.zeros(3), np.zeros(4), "m")
