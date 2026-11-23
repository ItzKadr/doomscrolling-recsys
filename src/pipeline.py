"""End-to-end experiment: simulate -> propensities -> naive vs IPS LambdaMART ->
unbiased + counterfactual evaluation -> offline A/B -> relevance/freshness Pareto.

Used by ``train.py`` (full run, MLflow, artifacts) and by ``app.py`` (small
in-memory demo when no artifacts exist yet).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dataset import FEATURE_COLUMNS, ClickLogs, SimConfig, feature_matrix, group_sizes, simulate
from src.ips import (
    DEFAULT_TAU,
    PropensityEstimate,
    click_ndcg_at_k,
    ctr_by_position,
    estimate_propensities,
    ips_dcg_per_session,
    ips_ndcg_at_k,
    snips_ndcg_at_k,
)
from src.metrics import ndcg_at_k, ranks_within_groups, recall_at_k, topk_mean
from src.pareto import KneePoint, moo_scores, point_at_alpha, select_knee, sweep_alpha
from src.ranker import (
    RankerConfig,
    TrainingSet,
    build_training_set,
    feature_importance,
    fit_ranker,
    predict,
)
from src.stats import ABTestResult, ab_test

logger = logging.getLogger(__name__)

#: Model name -> weighting scheme (see :mod:`src.ranker`).
MODEL_SCHEMES: dict[str, str] = {
    "naive": "none",
    "ips": "pair",
    "ips_clicked": "clicked",
    "ips_row": "row",
}


@dataclass(frozen=True)
class PipelineConfig:
    """All knobs of one experiment run."""

    sim: SimConfig = field(default_factory=SimConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    tau: float = DEFAULT_TAU
    ips_scheme: str = "pair"
    run_ablations: bool = True  # also train the "clicked" and "row" weighting schemes
    k: int = 5
    recall_threshold: float = 3.0  # relevant := grade >= 3 for Recall@k
    n_alphas: int = 101
    max_ndcg_drop: float = 0.02
    n_boot: int = 2000
    n_boot_propensity: int = 300
    eval_every: int = 10  # boosting rounds between validation-curve points
    # Counterfactual model selection: every arm picks its boosting round on the
    # validation logs with the offline metric it would realistically trust
    # (naive arm: click-NDCG@k; IPS arms: SNIPS-NDCG@k). Ground truth is never used.
    select_iteration: bool = True
    n_inspector_sessions: int = 300
    seed: int = 42

    def to_flat_dict(self) -> dict[str, object]:
        """Flat ``section.key -> value`` dict for MLflow params."""
        flat: dict[str, object] = {}
        for key, val in asdict(self).items():
            if isinstance(val, dict):
                for sub, sub_val in val.items():
                    flat[f"{key}.{sub}"] = sub_val
            else:
                flat[key] = val
        return flat


@dataclass
class SplitData:
    """Arrays of one split, rows contiguous per session (logged order)."""

    frame: pd.DataFrame
    X: np.ndarray
    sizes: np.ndarray
    clicks: np.ndarray
    positions: np.ndarray
    relevance: np.ndarray
    freshness: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "SplitData":
        return cls(
            frame=frame,
            X=feature_matrix(frame),
            sizes=group_sizes(frame),
            clicks=frame["click"].to_numpy(dtype=np.int64),
            positions=frame["position"].to_numpy(dtype=np.int64),
            relevance=frame["relevance"].to_numpy(dtype=float),
            freshness=frame["freshness"].to_numpy(dtype=float),
        )


@dataclass
class PipelineResult:
    """Everything produced by :func:`run_pipeline`."""

    config: PipelineConfig
    logs_summary: dict[str, float]
    true_propensities: np.ndarray
    propensity: PropensityEstimate
    ctr_logged: np.ndarray
    models: dict[str, lgb.LGBMRanker]
    best_iterations: dict[str, int]
    comparison: pd.DataFrame
    curves: pd.DataFrame
    ab_tests: list[ABTestResult]
    pareto_val: pd.DataFrame
    pareto_test: pd.DataFrame
    knee: KneePoint
    knee_test: dict[str, float]
    weight_check: dict[str, float]
    importance: pd.DataFrame
    inspector: pd.DataFrame
    timings: dict[str, float]

    def headline(self) -> dict[str, float]:
        """Key scalar metrics (for MLflow / README)."""
        comp = self.comparison.set_index("policy")
        k = self.config.k
        out: dict[str, float] = {}
        for policy in comp.index:
            for col in comp.columns:
                val = comp.loc[policy, col]
                if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                    out[f"test.{policy}.{col}"] = float(val)
        out["propensity.eta_hat"] = float(self.propensity.eta_hat)
        out["propensity.max_abs_err"] = float(
            np.max(np.abs(self.propensity.p_hat - self.true_propensities))
        )
        for key, val in self.knee.to_dict().items():
            out[f"knee.{key}"] = val
        for key, val in self.knee_test.items():
            out[f"knee_test.{key}"] = float(val)
        for res in self.ab_tests:
            tag = f"ab.{res.metric}.{res.control}_vs_{res.treatment}"
            out[f"{tag}.mean_diff"] = res.mean_diff
            out[f"{tag}.rel_lift"] = res.rel_lift
            out[f"{tag}.p_bootstrap"] = res.p_bootstrap
            out[f"{tag}.t_pvalue"] = res.t_pvalue
            out[f"{tag}.wilcoxon_pvalue"] = res.wilcoxon_pvalue
        for key, val in self.weight_check.items():
            out[f"weights.{key}"] = float(val)
        for name, it in self.best_iterations.items():
            out[f"best_iteration.{name}"] = float(it)
        out[f"ndcg@{k}.ips_minus_naive"] = float(
            comp.loc["ips", f"ndcg@{k}"] - comp.loc["naive", f"ndcg@{k}"]
        )
        return out


# ---------------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------------
def evaluate_policy(
    name: str,
    scores: np.ndarray,
    data: SplitData,
    p_hat: np.ndarray,
    cfg: PipelineConfig,
) -> dict[str, float | str]:
    """True-relevance and click-based metrics of one ranking policy on one split."""
    k = cfg.k
    return {
        "policy": name,
        f"ndcg@{k}": float(np.nanmean(ndcg_at_k(data.relevance, scores, data.sizes, k))),
        "ndcg@10": float(np.nanmean(ndcg_at_k(data.relevance, scores, data.sizes, 10))),
        f"recall@{k}": float(
            np.nanmean(recall_at_k(data.relevance, scores, data.sizes, k, cfg.recall_threshold))
        ),
        f"ips_ndcg@{k}": ips_ndcg_at_k(scores, data.clicks, data.positions, data.sizes, p_hat, k, cfg.tau),
        f"snips_ndcg@{k}": snips_ndcg_at_k(
            scores, data.clicks, data.positions, data.sizes, p_hat, k, cfg.tau
        ),
        f"click_ndcg@{k}": click_ndcg_at_k(scores, data.clicks, data.sizes, k),
        f"freshness@{k}": float(np.mean(topk_mean(data.freshness, scores, data.sizes, k))),
    }


def staged_scores(model: lgb.LGBMRanker, X: np.ndarray, every: int) -> list[tuple[int, np.ndarray]]:
    """Raw scores after every ``every`` boosting rounds, computed incrementally (O(T))."""
    booster = model.booster_
    total = booster.current_iteration()
    out: list[tuple[int, np.ndarray]] = []
    acc = np.zeros(X.shape[0])
    start = 0
    while start < total:
        step = min(every, total - start)
        acc = acc + booster.predict(X, start_iteration=start, num_iteration=step, raw_score=True)
        start += step
        out.append((start, acc.copy()))
    return out


def selection_metric(model_name: str, k: int) -> str:
    """Offline metric used to pick the boosting round of an arm (no ground truth)."""
    return f"click_ndcg@{k}" if model_name == "naive" else f"snips_ndcg@{k}"


def select_iterations(
    curves: pd.DataFrame, models: dict[str, lgb.LGBMRanker], k: int, enabled: bool = True
) -> dict[str, int]:
    """Best boosting round per model according to :func:`selection_metric`."""
    best: dict[str, int] = {}
    for name, model in models.items():
        total = int(model.booster_.current_iteration())
        if not enabled:
            best[name] = total
            continue
        sub = curves[(curves["model"] == name) & (curves["metric"] == selection_metric(name, k))]
        best[name] = int(sub.loc[sub["value"].idxmax(), "iteration"]) if len(sub) else total
    return best


def validation_curves(
    models: dict[str, lgb.LGBMRanker],
    val: SplitData,
    p_hat: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Validation NDCG / IPS-NDCG / SNIPS-NDCG / click-NDCG vs boosting round."""
    k = cfg.k
    rows = []
    for name, model in models.items():
        for iteration, scores in staged_scores(model, val.X, cfg.eval_every):
            metrics = {
                f"ndcg@{k}": float(np.nanmean(ndcg_at_k(val.relevance, scores, val.sizes, k))),
                f"ips_ndcg@{k}": ips_ndcg_at_k(
                    scores, val.clicks, val.positions, val.sizes, p_hat, k, cfg.tau
                ),
                f"snips_ndcg@{k}": snips_ndcg_at_k(
                    scores, val.clicks, val.positions, val.sizes, p_hat, k, cfg.tau
                ),
                f"click_ndcg@{k}": click_ndcg_at_k(scores, val.clicks, val.sizes, k),
            }
            for metric, value in metrics.items():
                rows.append({"model": name, "iteration": iteration, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def weight_effect_check(
    X_train: np.ndarray,
    ts_naive: TrainingSet,
    naive_model: lgb.LGBMRanker,
    ips_scores: np.ndarray,
    naive_scores: np.ndarray,
    test: SplitData,
    cfg: PipelineConfig,
) -> dict[str, float]:
    """Empirical proof that LightGBM uses the per-row weights in lambdarank.

    * control: the naive training set with all weights = 1 must reproduce the
      unweighted model exactly (max |score diff| ~ 0);
    * treatment: the IPS model must differ (rank correlation < 1, different top-k).
    """
    unit = TrainingSet(
        row_index=ts_naive.row_index,
        labels=ts_naive.labels,
        sizes=ts_naive.sizes,
        weights=np.ones(ts_naive.row_index.size),
        scheme="none+unit_weights",
    )
    unit_model = fit_ranker(X_train, unit, cfg.ranker, list(FEATURE_COLUMNS))
    unit_scores = predict(unit_model, test.X)
    naive_ref = predict(naive_model, test.X)

    k = cfg.k
    r_naive = ranks_within_groups(naive_scores, test.sizes)
    r_ips = ranks_within_groups(ips_scores, test.sizes)
    top_naive = (r_naive <= k).astype(np.int64)
    top_ips = (r_ips <= k).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(test.sizes)[:-1]))
    differs = np.add.reduceat(np.abs(top_naive - top_ips), offsets) > 0
    rank_corr = float(pd.Series(naive_scores).corr(pd.Series(ips_scores), method="spearman"))
    return {
        "unit_vs_unweighted_max_abs_diff": float(np.max(np.abs(unit_scores - naive_ref))),
        "ips_vs_naive_max_abs_diff": float(np.max(np.abs(ips_scores - naive_scores))),
        "ips_vs_naive_spearman": rank_corr,
        f"share_sessions_top{k}_changed": float(differs.mean()),
        "mean_abs_rank_shift": float(np.mean(np.abs(r_naive - r_ips))),
    }


def _summarize_logs(logs: ClickLogs) -> dict[str, float]:
    df = logs.df
    logged = df[df["split"] != "rand"]
    summary: dict[str, float] = {
        "n_rows": float(len(df)),
        "ctr": float(df["click"].mean()),
        "mean_relevance": float(df["relevance"].mean()),
        "share_sessions_with_click": float(df.groupby("session_id")["click"].max().mean()),
        "corr_position_freshness_logged": float(np.corrcoef(logged["position"], logged["freshness"])[0, 1]),
        "corr_position_relevance_logged": float(np.corrcoef(logged["position"], logged["relevance"])[0, 1]),
        "corr_freshness_relevance": float(np.corrcoef(df["freshness"], df["relevance"])[0, 1]),
    }
    for split, n in df.groupby("split")["session_id"].nunique().items():
        summary[f"n_sessions_{split}"] = float(n)
    return summary


def _inspector_frame(
    test: SplitData, scores: dict[str, np.ndarray], n_sessions: int
) -> pd.DataFrame:
    frame = test.frame
    n_rows = int(np.sum(test.sizes[:n_sessions]))
    cols = ["session_id", "user_id", "item_id", "position", "click", "relevance", "freshness"]
    out = frame.loc[: n_rows - 1, cols].copy()
    for name, s in scores.items():
        out[f"score_{name}"] = s[:n_rows]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------------
def run_pipeline(
    cfg: PipelineConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the whole experiment and return all results in memory."""
    cfg = cfg or PipelineConfig()
    say = progress or (lambda msg: logger.info(msg))
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    k = cfg.k
    feats = list(FEATURE_COLUMNS)

    # 1) logs ------------------------------------------------------------------------
    say("simulating click logs")
    logs = simulate(cfg.sim)
    splits = {name: SplitData.from_frame(logs.split(name)) for name in ("rand", "train", "val", "test")}
    train, val, test = splits["train"], splits["val"], splits["test"]
    timings["simulate_s"] = time.perf_counter() - t0

    # 2) propensities ------------------------------------------------------------------
    t = time.perf_counter()
    say("estimating propensities from the randomized slice")
    rand = splits["rand"]
    prop = estimate_propensities(
        rand.positions,
        rand.clicks,
        rand.sizes,
        cfg.sim.list_size,
        n_boot=cfg.n_boot_propensity,
        seed=cfg.seed,
    )
    ctr_logged = ctr_by_position(train.positions, train.clicks, cfg.sim.list_size)
    timings["propensity_s"] = time.perf_counter() - t

    # 3) rankers ----------------------------------------------------------------------
    t = time.perf_counter()
    model_names = ["naive", "ips"] + (["ips_clicked", "ips_row"] if cfg.run_ablations else [])
    schemes = dict(MODEL_SCHEMES, ips=cfg.ips_scheme)
    models: dict[str, lgb.LGBMRanker] = {}
    training_sets: dict[str, TrainingSet] = {}
    for name in model_names:
        say(f"training LambdaMART: {name} (scheme={schemes[name]})")
        ts = build_training_set(
            train.clicks, train.positions, train.sizes, prop.p_hat, cfg.tau, schemes[name], seed=cfg.seed
        )
        training_sets[name] = ts
        models[name] = fit_ranker(train.X, ts, cfg.ranker, feats)
    timings["train_s"] = time.perf_counter() - t

    # 4) validation curves + counterfactual model selection -----------------------------
    t = time.perf_counter()
    say("validation curves and counterfactual model selection")
    curves = validation_curves(models, val, prop.p_hat, cfg)
    best_iter = select_iterations(curves, models, k, cfg.select_iteration)
    say("selected boosting rounds: " + ", ".join(f"{n}={i}" for n, i in best_iter.items()))

    # 5) evaluation -------------------------------------------------------------------
    say("evaluating on true relevance and with counterfactual estimators")
    rng = np.random.default_rng(cfg.seed)
    test_scores: dict[str, np.ndarray] = {
        "logger": -test.positions.astype(float),
        "random": rng.random(test.positions.size),
    }
    val_scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        test_scores[name] = predict(model, test.X, best_iter[name])
        val_scores[name] = predict(model, val.X, best_iter[name])
    # tiny jitter breaks ties among equal grades deterministically
    test_scores["oracle"] = test.relevance + 1e-6 * rng.random(test.relevance.size)
    timings["evaluate_s"] = time.perf_counter() - t

    # 6) multi-objective re-ranking ------------------------------------------------------
    t = time.perf_counter()
    say("Pareto sweep relevance vs freshness (selection on validation)")
    alphas = np.linspace(0.0, 1.0, cfg.n_alphas)
    pareto_val = sweep_alpha(val_scores["ips"], val.freshness, val.relevance, val.sizes, alphas, k)
    knee = select_knee(pareto_val, k=k, max_rel_drop=cfg.max_ndcg_drop)
    pareto_test = sweep_alpha(test_scores["ips"], test.freshness, test.relevance, test.sizes, alphas, k)
    test_point = point_at_alpha(pareto_test, knee.alpha)
    knee_test = {
        "alpha": float(knee.alpha),
        f"ndcg@{k}": float(test_point[f"ndcg@{k}"]),
        f"freshness@{k}": float(test_point[f"freshness@{k}"]),
        "ndcg_drop_rel": float(test_point["ndcg_drop_rel"]),
        "freshness_gain_rel": float(test_point["freshness_gain_rel"]),
    }
    test_scores["ips_moo"] = moo_scores(test_scores["ips"], test.freshness, test.sizes, knee.alpha)
    timings["pareto_s"] = time.perf_counter() - t

    order = ["oracle", "logger", "random", "naive", "ips", "ips_moo"] + [
        n for n in ("ips_clicked", "ips_row") if n in models
    ]
    comparison = pd.DataFrame(
        [evaluate_policy(name, test_scores[name], test, prop.p_hat, cfg) for name in order]
    )

    # 7) offline A/B tests ---------------------------------------------------------------
    t = time.perf_counter()
    say("offline A/B tests (paired bootstrap, t-test, Wilcoxon)")

    def ndcg_per(name: str) -> np.ndarray:
        return ndcg_at_k(test.relevance, test_scores[name], test.sizes, k)

    def ips_dcg_per(name: str) -> np.ndarray:
        return ips_dcg_per_session(
            test_scores[name], test.clicks, test.positions, test.sizes, prop.p_hat, k, cfg.tau
        )

    def fresh_per(name: str) -> np.ndarray:
        return topk_mean(test.freshness, test_scores[name], test.sizes, k)

    ab_specs: list[tuple[str, np.ndarray, np.ndarray, str, str]] = [
        (f"ndcg@{k}", ndcg_per("naive"), ndcg_per("ips"), "naive", "ips"),
        (f"ips_dcg@{k}", ips_dcg_per("naive"), ips_dcg_per("ips"), "naive", "ips"),
        (f"ndcg@{k}", ndcg_per("ips"), ndcg_per("ips_moo"), "ips", "ips_moo"),
        (f"freshness@{k}", fresh_per("ips"), fresh_per("ips_moo"), "ips", "ips_moo"),
    ]
    ab_tests = [
        ab_test(a, b, metric, ca, cb, n_boot=cfg.n_boot, seed=cfg.seed + i)
        for i, (metric, a, b, ca, cb) in enumerate(ab_specs)
    ]
    timings["ab_test_s"] = time.perf_counter() - t

    # 8) diagnostics -------------------------------------------------------------------
    t = time.perf_counter()
    say("verifying that sample weights change the LambdaMART model")
    weight_check = weight_effect_check(
        train.X,
        training_sets["naive"],
        models["naive"],
        test_scores["ips"],
        test_scores["naive"],
        test,
        cfg,
    )
    importance = pd.DataFrame(
        {name: feature_importance(model, feats, best_iter[name]) for name, model in models.items()}
    ).rename_axis("feature").reset_index()
    inspector = _inspector_frame(
        test,
        {"naive": test_scores["naive"], "ips": test_scores["ips"]},
        min(cfg.n_inspector_sessions, test.sizes.size),
    )
    timings["diagnostics_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t0
    say(f"pipeline finished in {timings['total_s']:.1f}s")

    return PipelineResult(
        config=cfg,
        logs_summary=_summarize_logs(logs),
        true_propensities=logs.true_propensities,
        propensity=prop,
        ctr_logged=ctr_logged,
        models=models,
        best_iterations=best_iter,
        comparison=comparison,
        curves=curves,
        ab_tests=ab_tests,
        pareto_val=pareto_val,
        pareto_test=pareto_test,
        knee=knee,
        knee_test=knee_test,
        weight_check=weight_check,
        importance=importance,
        inspector=inspector,
        timings=timings,
    )
