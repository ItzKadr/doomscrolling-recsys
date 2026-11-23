"""Serialization of pipeline results into a dashboard bundle (artifacts/ folder).

The same :class:`DashboardBundle` is built either from an in-memory
:class:`~src.pipeline.PipelineResult` (Streamlit cold start) or from the files
written by ``train.py`` so the dashboard code has a single data path.

Layout of ``artifacts/``::

    metrics.json              config, log stats, headline metrics, knee, weight check
    model_comparison.csv      test metrics per ranking policy
    propensities.csv          true p_k, estimated p_hat_k (+ CI), CTR by position
    pareto.csv                alpha sweep on the test split (front, chord distance)
    pareto_val.csv            alpha sweep on validation (used to select the knee)
    ab_test.json              offline A/B test results
    ab_bootstrap.csv          bootstrap replicates of the mean difference per test
    curves.csv                validation metrics vs boosting round
    feature_importance.csv    normalized gain importance per model
    inspector_sessions.csv    a sample of test sessions with model scores
    models/<name>_lgbm.txt    LightGBM boosters (selected number of trees)
    figures/*.png             static figures for the README
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline import PipelineResult

logger = logging.getLogger(__name__)

REQUIRED_FILES: tuple[str, ...] = (
    "metrics.json",
    "model_comparison.csv",
    "propensities.csv",
    "pareto.csv",
    "ab_test.json",
    "ab_bootstrap.csv",
    "curves.csv",
    "feature_importance.csv",
    "inspector_sessions.csv",
)


@dataclass
class DashboardBundle:
    """Everything the dashboard renders."""

    source: str  # "artifacts" or "demo"
    summary: dict[str, Any]
    propensities: pd.DataFrame
    comparison: pd.DataFrame
    curves: pd.DataFrame
    pareto: pd.DataFrame
    pareto_val: pd.DataFrame
    ab_tests: list[dict[str, Any]]
    ab_boot: pd.DataFrame
    importance: pd.DataFrame
    inspector: pd.DataFrame


def ab_test_id(res: dict[str, Any]) -> str:
    """Stable identifier of an A/B test, e.g. ``ndcg@5:naive_vs_ips``."""
    return f"{res['metric']}:{res['control']}_vs_{res['treatment']}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def bundle_from_result(result: PipelineResult, source: str = "demo") -> DashboardBundle:
    """Convert an in-memory pipeline result into a dashboard bundle."""
    prop = result.propensity.to_frame(result.true_propensities)
    prop["ctr_logged"] = result.ctr_logged
    prop["ips_weight_clipped"] = 1.0 / np.maximum(prop["p_hat"], result.config.tau)
    ab = [r.to_dict() for r in result.ab_tests]
    ab_boot = pd.DataFrame({ab_test_id(d): r.boot_diffs for d, r in zip(ab, result.ab_tests)})
    summary = {
        "config": result.config.to_flat_dict(),
        "logs": result.logs_summary,
        "headline": result.headline(),
        "knee": result.knee.to_dict(),
        "knee_test": result.knee_test,
        "weight_check": result.weight_check,
        "best_iterations": result.best_iterations,
        "propensity": {
            "method": result.propensity.method,
            "n_sessions": result.propensity.n_sessions,
            "eta_hat": result.propensity.eta_hat,
            "eta_true": result.config.sim.eta,
            "tau": result.config.tau,
        },
        "timings": result.timings,
    }
    return DashboardBundle(
        source=source,
        summary=_jsonable(summary),
        propensities=prop,
        comparison=result.comparison.copy(),
        curves=result.curves.copy(),
        pareto=result.pareto_test.copy(),
        pareto_val=result.pareto_val.copy(),
        ab_tests=_jsonable(ab),
        ab_boot=ab_boot,
        importance=result.importance.copy(),
        inspector=result.inspector.copy(),
    )


def _save_figures(bundle: DashboardBundle, fig_dir: Path) -> list[Path]:
    """Static PNGs (matplotlib, headless) for the README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    prop = bundle.propensities
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(prop["position"], prop["p_true"], "k--", label="true examination $p_k$")
    ax.errorbar(
        prop["position"],
        prop["p_hat"],
        yerr=[prop["p_hat"] - prop["p_hat_ci_low"], prop["p_hat_ci_high"] - prop["p_hat"]],
        fmt="o-",
        capsize=3,
        label="estimated $\\hat p_k$ (randomized slice, 95% CI)",
    )
    ax.plot(prop["position"], prop["ctr_logged"] / prop["ctr_logged"].iloc[0], "s:", label="logged CTR / CTR$_1$")
    tau = bundle.summary["propensity"]["tau"]
    ax.axhline(tau, color="grey", lw=0.8)
    ax.text(prop["position"].max(), tau, f" tau={tau}", va="bottom", ha="right", color="grey")
    ax.set_xlabel("position k")
    ax.set_ylabel("relative examination / CTR")
    ax.set_title("Position bias")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "propensities.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)

    par = bundle.pareto
    knee = bundle.summary["knee"]
    k = int(bundle.summary["config"]["k"])
    n_col, f_col = f"ndcg@{k}", f"freshness@{k}"
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(par[f_col], par[n_col], "-", color="#4c72b0", lw=1.5, label="alpha sweep (test)")
    front = par[par["is_pareto"]]
    ax.scatter(front[f_col], front[n_col], s=10, color="#4c72b0")
    base = float(par.loc[par["alpha"].idxmin(), n_col])
    ax.axhline(base * (1 - knee["max_rel_drop"]), color="#c44e52", ls="--", lw=1, label="-2% NDCG guard-rail")
    row = par.iloc[int(np.argmin(np.abs(par["alpha"] - knee["alpha"])))]
    ax.scatter([row[f_col]], [row[n_col]], s=90, marker="*", color="#dd8452", zorder=5, label=f"knee alpha={knee['alpha']:.2f}")
    ax.set_xlabel(f"Freshness@{k}")
    ax.set_ylabel(f"NDCG@{k} (true relevance)")
    ax.set_title("Relevance vs freshness Pareto front")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "pareto.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)

    if bundle.ab_tests:
        res = bundle.ab_tests[0]
        diffs = bundle.ab_boot[ab_test_id(res)].to_numpy()
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.hist(diffs, bins=50, color="#55a868", alpha=0.8)
        for x in (res["ci_low"], res["ci_high"]):
            ax.axvline(x, color="k", ls="--", lw=1)
        ax.axvline(0.0, color="#c44e52", lw=1)
        ax.set_xlabel(f"bootstrap mean diff of {res['metric']} ({res['treatment']} - {res['control']})")
        ax.set_title("Paired bootstrap (2.5% / 97.5% percentiles dashed)")
        fig.tight_layout()
        path = fig_dir / "ab_bootstrap.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def save_artifacts(result: PipelineResult, out_dir: Path) -> list[Path]:
    """Write the bundle, the boosters and the figures. Returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle_from_result(result, source="artifacts")
    written: list[Path] = []

    def dump_json(name: str, obj: Any) -> None:
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(obj), fh, indent=2)
        written.append(path)

    def dump_csv(name: str, frame: pd.DataFrame) -> None:
        path = out_dir / name
        frame.to_csv(path, index=False, encoding="utf-8")
        written.append(path)

    dump_json("metrics.json", bundle.summary)
    dump_json("ab_test.json", bundle.ab_tests)
    dump_csv("model_comparison.csv", bundle.comparison)
    dump_csv("propensities.csv", bundle.propensities)
    dump_csv("pareto.csv", bundle.pareto)
    dump_csv("pareto_val.csv", bundle.pareto_val)
    dump_csv("ab_bootstrap.csv", bundle.ab_boot)
    dump_csv("curves.csv", bundle.curves)
    dump_csv("feature_importance.csv", bundle.importance)
    dump_csv("inspector_sessions.csv", bundle.inspector)

    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for name, model in result.models.items():
        path = model_dir / f"{name}_lgbm.txt"
        model.booster_.save_model(str(path), num_iteration=result.best_iterations.get(name))
        written.append(path)
    try:
        written.extend(_save_figures(bundle, out_dir / "figures"))
    except Exception as exc:  # figures are a nice-to-have, never fatal
        logger.warning("could not render figures: %s", exc)
    logger.info("wrote %d artifact files to %s", len(written), out_dir)
    return written


def load_bundle(art_dir: Path) -> DashboardBundle | None:
    """Load a bundle written by :func:`save_artifacts` (None if incomplete)."""
    art_dir = Path(art_dir)
    if not all((art_dir / name).exists() for name in REQUIRED_FILES):
        return None
    try:
        with open(art_dir / "metrics.json", encoding="utf-8") as fh:
            summary = json.load(fh)
        with open(art_dir / "ab_test.json", encoding="utf-8") as fh:
            ab_tests = json.load(fh)
        pareto_val_path = art_dir / "pareto_val.csv"
        pareto = pd.read_csv(art_dir / "pareto.csv")
        return DashboardBundle(
            source="artifacts",
            summary=summary,
            propensities=pd.read_csv(art_dir / "propensities.csv"),
            comparison=pd.read_csv(art_dir / "model_comparison.csv"),
            curves=pd.read_csv(art_dir / "curves.csv"),
            pareto=pareto,
            pareto_val=pd.read_csv(pareto_val_path) if pareto_val_path.exists() else pareto,
            ab_tests=ab_tests,
            ab_boot=pd.read_csv(art_dir / "ab_bootstrap.csv"),
            importance=pd.read_csv(art_dir / "feature_importance.csv"),
            inspector=pd.read_csv(art_dir / "inspector_sessions.csv"),
        )
    except Exception as exc:  # corrupted / partial artifacts -> fall back to demo
        logger.warning("could not load artifacts from %s: %s", art_dir, exc)
        return None
