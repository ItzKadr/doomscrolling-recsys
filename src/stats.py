r"""Offline A/B testing of two rankers on the same sessions (paired design).

For every session ``s`` we have the metric of the control ``a_s`` and of the
treatment ``b_s`` (e.g. NDCG@5 of the naive vs. the IPS ranker on the same
logged list), so the natural unit is the paired difference ``d_s = b_s - a_s``.

* **Paired bootstrap** (non-parametric): resample sessions with replacement
  ``B`` times, recompute ``mean(d*)`` and the relative lift
  ``mean(b*) / mean(a*) - 1``; percentile 95 % CIs. The two-sided p-value is
  the percentile-bootstrap p-value
  ``p = min(1, 2 (1 + min(#{d*_b <= 0}, #{d*_b >= 0})) / (B + 1))``, i.e. twice
  the bootstrap mass on the far side of zero; it is consistent with the
  percentile CI (the 95 % CI excludes 0 iff p < 0.05, up to the +1 correction).
* **Paired Student t-test**: ``t = d_bar / (s_d / sqrt(n))`` with ``n - 1`` dof;
  relies on the CLT for the mean of ``d`` (fine for thousands of sessions).
* **Wilcoxon signed-rank test**: rank-based test of a symmetric distribution of
  ``d`` around 0; robust to outliers, ignores zero differences
  (``zero_method="wilcox"``), tests a location shift rather than the mean.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Outcome of a paired offline A/B comparison (treatment - control)."""

    metric: str
    control: str
    treatment: str
    n: int
    mean_control: float
    mean_treatment: float
    mean_diff: float
    ci_low: float
    ci_high: float
    rel_lift: float
    rel_lift_ci_low: float
    rel_lift_ci_high: float
    p_bootstrap: float
    t_stat: float
    t_pvalue: float
    wilcoxon_stat: float
    wilcoxon_pvalue: float
    n_nonzero: int
    n_boot: int
    ci_level: float
    boot_diffs: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def significant(self) -> bool:
        """True when the bootstrap CI of the mean difference excludes 0."""
        return bool(self.ci_low > 0 or self.ci_high < 0)

    def to_dict(self, include_boot: bool = False) -> dict[str, object]:
        out = {k: v for k, v in asdict(self).items() if k != "boot_diffs"}
        out = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}
        out["significant"] = self.significant
        if include_boot:
            out["boot_diffs"] = self.boot_diffs.tolist()
        return out


def paired_bootstrap(
    control: np.ndarray,
    treatment: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    chunk: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap replicates of ``mean(b - a)`` and ``mean(b) / mean(a) - 1``.

    Sessions are resampled jointly for both arms (pairing preserved). Work is
    done in chunks of ``chunk`` replicates to bound memory (chunk x n indices).
    """
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    n = a.size
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    lifts = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        ma = a[idx].mean(axis=1)
        mb = b[idx].mean(axis=1)
        diffs[start : start + m] = mb - ma
        with np.errstate(divide="ignore", invalid="ignore"):
            lifts[start : start + m] = np.where(ma != 0, mb / ma - 1.0, np.nan)
    return diffs, lifts


def ab_test(
    control: np.ndarray,
    treatment: np.ndarray,
    metric: str,
    control_name: str = "A",
    treatment_name: str = "B",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> ABTestResult:
    """Paired bootstrap + paired t-test + Wilcoxon signed-rank on per-session metrics.

    Pairs where either arm is NaN (e.g. sessions with an undefined NDCG) are
    dropped before testing.
    """
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.shape != b.shape:
        raise ValueError("control and treatment must be paired (same shape)")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = a.size
    if n < 2:
        raise ValueError("need at least two paired observations")
    d = b - a
    d_bar = float(d.mean())
    diffs, lifts = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
    q = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(diffs, [q, 1.0 - q])
    lift_low, lift_high = np.nanquantile(lifts, [q, 1.0 - q])
    tail = min(np.sum(diffs <= 0.0), np.sum(diffs >= 0.0))
    p_boot = min(1.0, 2.0 * (1.0 + tail) / (n_boot + 1.0))

    if np.allclose(d, 0.0):
        t_stat, t_p, w_stat, w_p = 0.0, 1.0, 0.0, 1.0
    else:
        t_res = sps.ttest_rel(b, a)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
        w_res = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)

    mean_a = float(a.mean())
    result = ABTestResult(
        metric=metric,
        control=control_name,
        treatment=treatment_name,
        n=int(n),
        mean_control=mean_a,
        mean_treatment=float(b.mean()),
        mean_diff=d_bar,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        rel_lift=float(b.mean() / mean_a - 1.0) if mean_a != 0 else float("nan"),
        rel_lift_ci_low=float(lift_low),
        rel_lift_ci_high=float(lift_high),
        p_bootstrap=float(p_boot),
        t_stat=t_stat,
        t_pvalue=t_p,
        wilcoxon_stat=w_stat,
        wilcoxon_pvalue=w_p,
        n_nonzero=int(np.count_nonzero(d)),
        n_boot=int(n_boot),
        ci_level=float(ci),
        boot_diffs=diffs,
    )
    logger.info(
        "A/B %s: %s -> %s diff=%.5f CI=[%.5f, %.5f] lift=%.2f%% p_boot=%.4g p_t=%.4g p_w=%.4g",
        metric,
        control_name,
        treatment_name,
        d_bar,
        ci_low,
        ci_high,
        100 * result.rel_lift,
        p_boot,
        t_p,
        w_p,
    )
    return result
