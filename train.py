"""Train and evaluate the debiased LTR + multi-objective re-ranking pipeline.

Examples
--------
    python train.py                 # default run (~30k sessions, < 1 min on a laptop CPU)
    python train.py --quick         # smoke run (~12k sessions)
    python train.py --full          # bigger run (~120k sessions, 500 trees)
    python train.py --eta 1.5 --tau 0.1 --scheme clicked --no-mlflow

Outputs: ``artifacts/`` (json/csv/png/LightGBM models) and an MLflow run in the
local store ``mlflow.db`` (+ ``mlruns/`` for run artifacts).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.artifacts import save_artifacts  # noqa: E402
from src.dataset import SimConfig  # noqa: E402
from src.pipeline import PipelineConfig, PipelineResult, run_pipeline  # noqa: E402
from src.ranker import RankerConfig  # noqa: E402
from src.tracking import SafeMlflowRun  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("train")

PRESETS: dict[str, dict[str, int]] = {
    "quick": {"n_sessions": 12_000, "n_estimators": 200},
    "default": {"n_sessions": 30_000, "n_estimators": 300},
    "full": {"n_sessions": 120_000, "n_estimators": 500},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="small fast run (~12k sessions)")
    mode.add_argument("--full", action="store_true", help="large run (~120k sessions, 500 trees)")
    parser.add_argument("--sessions", type=int, default=None, help="override number of sessions")
    parser.add_argument("--trees", type=int, default=None, help="override number of boosting rounds")
    parser.add_argument("--eta", type=float, default=1.0, help="true position-bias exponent, p_k = k^-eta")
    parser.add_argument("--tau", type=float, default=0.15, help="IPS clipping threshold")
    parser.add_argument(
        "--scheme",
        choices=("pair", "clicked", "row"),
        default="pair",
        help="IPS weighting scheme of the main IPS model (see src/ranker.py)",
    )
    parser.add_argument("--max-drop", type=float, default=0.02, help="max relative NDCG drop for the MOO knee")
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap iterations for A/B tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ablations", action="store_true", help="skip the 'clicked'/'row' ablation models")
    parser.add_argument("--no-selection", action="store_true", help="use all trees (no counterfactual model selection)")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    parser.add_argument("--experiment", default="doomscrolling-recsys", help="MLflow experiment name")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_DIR / "artifacts")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> tuple[str, PipelineConfig]:
    preset_name = "quick" if args.quick else "full" if args.full else "default"
    preset = PRESETS[preset_name]
    sim = SimConfig(
        n_sessions=args.sessions or preset["n_sessions"],
        eta=args.eta,
        seed=args.seed,
    )
    ranker = replace(RankerConfig(), n_estimators=args.trees or preset["n_estimators"], seed=args.seed)
    cfg = PipelineConfig(
        sim=sim,
        ranker=ranker,
        tau=args.tau,
        ips_scheme=args.scheme,
        run_ablations=not args.no_ablations,
        max_ndcg_drop=args.max_drop,
        n_boot=args.n_boot,
        select_iteration=not args.no_selection,
        seed=args.seed,
    )
    return preset_name, cfg


def log_to_mlflow(run: SafeMlflowRun, result: PipelineResult, preset: str, art_dir: Path) -> None:
    run.set_tags({"preset": preset, "ips_scheme": result.config.ips_scheme, "project": "debiased-ltr-moo"})
    run.log_params(result.config.to_flat_dict())
    run.log_metrics(result.headline())
    run.log_metrics({f"logs.{k}": v for k, v in result.logs_summary.items()})
    run.log_metrics({f"time.{k}": v for k, v in result.timings.items()})
    records: list[tuple[str, float, int]] = []
    # curves with steps: validation metrics vs boosting round
    for model, metric, it, val in result.curves[["model", "metric", "iteration", "value"]].itertuples(index=False):
        records.append((f"val_curve.{model}.{metric}", float(val), int(it)))
    # Pareto sweep with step = round(100 * alpha)
    k = result.config.k
    for alpha, ndcg, fresh in result.pareto_test[["alpha", f"ndcg@{k}", f"freshness@{k}"]].itertuples(index=False):
        step = int(round(100 * alpha))
        records.append((f"pareto.ndcg@{k}", float(ndcg), step))
        records.append((f"pareto.freshness@{k}", float(fresh), step))
    # estimated vs true propensities with step = position
    for pos, p_hat, p_true in zip(result.propensity.positions, result.propensity.p_hat, result.true_propensities):
        records.append(("propensity.p_hat", float(p_hat), int(pos)))
        records.append(("propensity.p_true", float(p_true), int(pos)))
    run.log_metric_steps(records)
    run.log_artifacts(art_dir)


def print_report(result: PipelineResult, art_dir: Path) -> None:
    k = result.config.k
    cols = ["policy", f"ndcg@{k}", "ndcg@10", f"recall@{k}", f"ips_ndcg@{k}", f"snips_ndcg@{k}", f"click_ndcg@{k}", f"freshness@{k}"]
    print("\n=== Test-set comparison (true relevance + counterfactual estimators) ===")
    print(result.comparison[cols].round(4).to_string(index=False))
    prop = result.propensity
    print("\n=== Propensities (randomized slice, p_1 = 1) ===")
    print("p_true:", " ".join(f"{p:.3f}" for p in result.true_propensities))
    print("p_hat :", " ".join(f"{p:.3f}" for p in prop.p_hat), f"| eta_hat={prop.eta_hat:.3f}")
    print("\n=== Offline A/B tests (paired, test sessions) ===")
    for res in result.ab_tests:
        print(
            f"{res.metric:13s} {res.control:>6s} -> {res.treatment:<8s} diff={res.mean_diff:+.4f} "
            f"CI95=[{res.ci_low:+.4f}, {res.ci_high:+.4f}] lift={100 * res.rel_lift:+.2f}% "
            f"p_boot={res.p_bootstrap:.4f} p_t={res.t_pvalue:.2e} p_wilcoxon={res.wilcoxon_pvalue:.2e}"
        )
    knee = result.knee
    print("\n=== Multi-objective operating point (selected on validation) ===")
    print(
        f"alpha*={knee.alpha:.2f} | val: NDCG@{k} drop={100 * knee.ndcg_drop_rel:.2f}% "
        f"freshness gain={100 * knee.freshness_gain_rel:+.2f}% | test: NDCG@{k} drop="
        f"{100 * result.knee_test['ndcg_drop_rel']:.2f}% freshness gain="
        f"{100 * result.knee_test['freshness_gain_rel']:+.2f}% | unconstrained knee alpha="
        f"{knee.unconstrained_alpha:.2f}"
    )
    wc = result.weight_check
    print("\n=== Do sample weights change LambdaMART? ===")
    print(
        f"unit weights vs unweighted: max|diff|={wc['unit_vs_unweighted_max_abs_diff']:.2e} | "
        f"IPS vs naive: spearman={wc['ips_vs_naive_spearman']:.3f}, "
        f"top-{k} changed in {100 * wc[f'share_sessions_top{k}_changed']:.1f}% of sessions"
    )
    print("best boosting rounds:", result.best_iterations)
    print(f"\nTotal time {result.timings['total_s']:.1f}s. Artifacts: {art_dir}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    for noisy in ("alembic", "mlflow", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args(argv)
    preset, cfg = build_config(args)
    logger.info("preset=%s sessions=%d trees=%d", preset, cfg.sim.n_sessions, cfg.ranker.n_estimators)

    result = run_pipeline(cfg)
    art_dir = Path(args.artifacts_dir)
    save_artifacts(result, art_dir)

    with SafeMlflowRun(REPO_DIR, args.experiment, run_name=f"{preset}-{cfg.ips_scheme}-seed{cfg.seed}", enabled=not args.no_mlflow) as run:
        log_to_mlflow(run, result, preset, art_dir)
        if run.active:
            logger.info("logged MLflow run %s", run.run_id)

    print_report(result, art_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
