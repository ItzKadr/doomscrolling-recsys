"""Tiny end-to-end run of the pipeline + artifact round-trip."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.artifacts import REQUIRED_FILES, load_bundle, save_artifacts
from src.dataset import SimConfig
from src.pipeline import PipelineConfig, run_pipeline
from src.ranker import RankerConfig
from src.tracking import sanitize_key

TINY = PipelineConfig(
    sim=SimConfig(n_sessions=3000),
    ranker=replace(RankerConfig(), n_estimators=40, min_child_samples=50, n_jobs=2),
    run_ablations=False,
    n_alphas=21,
    n_boot=300,
    n_boot_propensity=50,
    n_inspector_sessions=20,
)


@pytest.fixture(scope="module")
def result():
    return run_pipeline(TINY)


def test_pipeline_outputs(result) -> None:
    comp = result.comparison.set_index("policy")
    assert {"oracle", "logger", "random", "naive", "ips", "ips_moo"} <= set(comp.index)
    assert comp.loc["oracle", "ndcg@5"] == pytest.approx(1.0)
    assert comp.loc["random", "ndcg@5"] < comp.loc["naive", "ndcg@5"]
    assert len(result.ab_tests) == 4
    assert result.weight_check["unit_vs_unweighted_max_abs_diff"] < 1e-9
    assert result.weight_check["ips_vs_naive_max_abs_diff"] > 0
    assert 0.0 <= result.knee.alpha <= 1.0
    assert result.knee.ndcg_drop_rel <= TINY.max_ndcg_drop + 1e-12
    assert set(result.best_iterations) == {"naive", "ips"}
    assert result.inspector["session_id"].nunique() == 20
    headline = result.headline()
    assert all(np.isfinite(v) for v in headline.values())


def test_artifacts_roundtrip(result, tmp_path) -> None:
    save_artifacts(result, tmp_path)
    for name in REQUIRED_FILES:
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "models" / "ips_lgbm.txt").exists()
    bundle = load_bundle(tmp_path)
    assert bundle is not None and bundle.source == "artifacts"
    assert len(bundle.ab_tests) == 4
    assert bundle.ab_boot.shape == (TINY.n_boot, 4)
    assert bundle.summary["knee"]["alpha"] == pytest.approx(result.knee.alpha)
    assert load_bundle(tmp_path / "missing") is None


def test_sanitize_mlflow_keys() -> None:
    assert sanitize_key("test.ips.ndcg@5") == "test.ips.ndcg_at_5"
    assert sanitize_key("a|b>c") == "a_b_c"
