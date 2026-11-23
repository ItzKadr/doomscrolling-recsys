"""Smoke tests of the Streamlit dashboard (cold-start demo and artifacts mode)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.artifacts import save_artifacts
from src.dataset import SimConfig
from src.pipeline import PipelineConfig, run_pipeline
from src.ranker import RankerConfig

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_cold_start_demo(tmp_path, monkeypatch) -> None:
    """No artifacts -> the app computes a small demo pipeline in memory."""
    monkeypatch.setenv("RECSYS_ARTIFACTS_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 5


def test_app_with_artifacts(tmp_path, monkeypatch) -> None:
    cfg = PipelineConfig(
        sim=SimConfig(n_sessions=2500),
        ranker=replace(RankerConfig(), n_estimators=30, min_child_samples=50, n_jobs=2),
        run_ablations=True,
        n_alphas=21,
        n_boot=200,
        n_boot_propensity=50,
        n_inspector_sessions=10,
    )
    save_artifacts(run_pipeline(cfg), tmp_path)
    monkeypatch.setenv("RECSYS_ARTIFACTS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    # interact: change the A/B comparison and the inspector session
    at.selectbox[2].set_value(at.selectbox[2].options[1]).run()
    assert not at.exception
    at.slider[0].set_value(0.5).run()
    assert not at.exception
