"""Hand-computed and reference checks for the vectorized grouped metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    dcg_at_k,
    group_mean,
    ideal_dcg_at_k,
    ndcg_at_k,
    pad_by_group,
    ranks_within_groups,
    recall_at_k,
    topk_mean,
)


def test_pad_by_group_roundtrip() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sizes = np.array([3, 1, 2])
    padded, mask = pad_by_group(values, sizes, fill=-1.0)
    assert padded.shape == (3, 3)
    np.testing.assert_array_equal(padded[0], [1, 2, 3])
    np.testing.assert_array_equal(padded[1], [4, -1, -1])
    np.testing.assert_array_equal(padded[2], [5, 6, -1])
    np.testing.assert_array_equal(padded[mask], values)


def test_wikipedia_dcg_example_linear_gain() -> None:
    # Classic example: relevances in ranked order [3, 2, 3, 0, 1, 2].
    rel = np.array([3, 2, 3, 0, 1, 2], dtype=float)
    score = np.arange(6, 0, -1, dtype=float)  # keep the given order
    sizes = np.array([6])
    dcg = dcg_at_k(rel, score, sizes, k=6, gain="linear")[0]
    # 3 + 2/log2(3) + 3/2 + 0 + 1/log2(6) + 2/log2(7)
    expected = 3 + 2 / np.log2(3) + 1.5 + 0 + 1 / np.log2(6) + 2 / np.log2(7)
    assert dcg == pytest.approx(expected)
    assert dcg == pytest.approx(6.861, abs=1e-3)
    idcg = ideal_dcg_at_k(rel, sizes, k=6, gain="linear")[0]
    assert idcg == pytest.approx(7.141, abs=1e-3)
    assert ndcg_at_k(rel, score, sizes, k=6, gain="linear")[0] == pytest.approx(0.961, abs=1e-3)


def test_ndcg_at_3_exp2_hand_computed() -> None:
    # model ranking: items with grades [1, 3, 0, 2]; ideal: [3, 2, 1, 0]
    rel = np.array([1, 3, 0, 2], dtype=float)
    score = np.array([0.9, 0.8, 0.7, 0.1])
    sizes = np.array([4])
    dcg = 1.0 + 7.0 / np.log2(3) + 0.0  # (2^1-1)/1 + (2^3-1)/log2(3) + 0
    idcg = 7.0 + 3.0 / np.log2(3) + 1.0 / 2.0
    assert dcg_at_k(rel, score, sizes, 3)[0] == pytest.approx(dcg)
    assert ndcg_at_k(rel, score, sizes, 3)[0] == pytest.approx(dcg / idcg)
    assert dcg == pytest.approx(5.4165, abs=1e-4) and idcg == pytest.approx(9.3928, abs=1e-4)
    assert dcg / idcg == pytest.approx(0.5767, abs=1e-4)


def test_ndcg_matches_python_loop_on_ragged_groups() -> None:
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 12, size=200)
    n = int(sizes.sum())
    rel = rng.integers(0, 5, size=n).astype(float)
    score = rng.normal(size=n)
    k = 5
    fast = ndcg_at_k(rel, score, sizes, k)
    start = 0
    for g, size in enumerate(sizes):
        r = rel[start : start + size]
        s = score[start : start + size]
        order = np.argsort(-s, kind="stable")
        disc = 1 / np.log2(np.arange(2, size + 2))
        dcg = ((2 ** r[order] - 1) * disc)[:k].sum()
        idcg = ((2 ** np.sort(r)[::-1] - 1) * disc)[:k].sum()
        if idcg == 0:
            assert np.isnan(fast[g])
        else:
            assert fast[g] == pytest.approx(dcg / idcg)
        start += size


def test_ndcg_bounds_and_perfect_ranking() -> None:
    rng = np.random.default_rng(1)
    sizes = np.full(50, 10)
    rel = rng.integers(0, 5, size=500).astype(float)
    perfect = ndcg_at_k(rel, rel + 1e-6 * rng.random(500), sizes, 5)
    assert np.allclose(perfect[~np.isnan(perfect)], 1.0)
    random = ndcg_at_k(rel, rng.random(500), sizes, 5)
    valid = random[~np.isnan(random)]
    assert np.all((valid >= 0) & (valid <= 1 + 1e-12))


def test_all_zero_group_is_nan() -> None:
    rel = np.array([0, 0, 0, 1, 0], dtype=float)
    out = ndcg_at_k(rel, np.array([3, 2, 1, 0.5, 0.2]), np.array([3, 2]), 2)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)


def test_recall_at_k_hand_computed() -> None:
    # group 0: relevant (>=3) items are rows 1 and 3; top-2 by score = rows 0, 1 -> 1/2
    # group 1: no relevant item -> NaN
    rel = np.array([0, 4, 1, 3, 1, 2], dtype=float)
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.5, 0.4])
    out = recall_at_k(rel, score, np.array([4, 2]), k=2, threshold=3)
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])


def test_ranks_and_topk_mean() -> None:
    score = np.array([0.1, 0.9, 0.5, 2.0, 1.0])
    sizes = np.array([3, 2])
    np.testing.assert_array_equal(ranks_within_groups(score, sizes), [3, 1, 2, 1, 2])
    values = np.array([10.0, 20.0, 30.0, 1.0, 2.0])
    np.testing.assert_allclose(topk_mean(values, score, sizes, 2), [25.0, 1.5])
    np.testing.assert_allclose(topk_mean(values, score, sizes, 5), [20.0, 1.5])  # k > group size
    np.testing.assert_allclose(group_mean(values, sizes), [20.0, 1.5])


def test_stable_tie_breaking() -> None:
    # equal scores keep the original row order
    np.testing.assert_array_equal(ranks_within_groups(np.zeros(4), np.array([4])), [1, 2, 3, 4])
