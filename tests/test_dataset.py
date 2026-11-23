"""Simulator invariants: shapes, splits, bias structure, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset import FEATURE_COLUMNS, SimConfig, attractiveness, feature_matrix, group_sizes, simulate


@pytest.fixture(scope="module")
def logs():
    return simulate(SimConfig(n_sessions=4000, seed=11))


def test_shapes_and_positions(logs) -> None:
    cfg = logs.config
    df = logs.df
    assert len(df) == cfg.n_sessions * cfg.list_size
    sizes = group_sizes(df)
    assert sizes.size == cfg.n_sessions and np.all(sizes == cfg.list_size)
    pos = df["position"].to_numpy().reshape(cfg.n_sessions, cfg.list_size)
    assert np.all(pos == np.arange(1, cfg.list_size + 1))
    assert set(df["split"].unique()) == {"rand", "train", "val", "test"}
    assert df["relevance"].between(0, cfg.max_grade).all()
    assert set(df["click"].unique()) <= {0, 1}
    # no duplicated item inside a session
    assert not df.duplicated(["session_id", "item_id"]).any()
    X = feature_matrix(df)
    assert X.shape == (len(df), len(FEATURE_COLUMNS)) and np.isfinite(X).all()


def test_logged_policy_is_confounded_with_freshness(logs) -> None:
    logged = logs.df[logs.df["split"] == "train"]
    rand = logs.df[logs.df["split"] == "rand"]
    assert np.corrcoef(logged["position"], logged["freshness"])[0, 1] < -0.3
    assert abs(np.corrcoef(rand["position"], rand["freshness"])[0, 1]) < 0.05
    # freshness itself is only weakly related to true relevance
    assert abs(np.corrcoef(logs.df["freshness"], logs.df["relevance"])[0, 1]) < 0.15


def test_split_is_by_session(logs) -> None:
    per_session = logs.df.groupby("session_id")["split"].nunique()
    assert (per_session == 1).all()


def test_true_propensities_and_attractiveness() -> None:
    cfg = SimConfig(eta=1.5)
    p = cfg.true_propensities()
    assert p[0] == 1.0 and np.all(np.diff(p) < 0)
    assert p[3] == pytest.approx(4**-1.5)
    g = attractiveness(np.arange(5), eps=0.05, max_grade=4)
    assert g[0] == pytest.approx(0.05) and g[-1] == pytest.approx(1.0)
    assert np.all(np.diff(g) > 0)


def test_deterministic_given_seed() -> None:
    a = simulate(SimConfig(n_sessions=1000, seed=7)).df
    b = simulate(SimConfig(n_sessions=1000, seed=7)).df
    pd.testing.assert_frame_equal(a, b)
    c = simulate(SimConfig(n_sessions=1000, seed=8)).df
    assert not a["click"].equals(c["click"])
