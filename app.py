"""Streamlit dashboard: debiased learning-to-rank & relevance/freshness trade-off.

Run with ``streamlit run app.py``. Uses ``artifacts/`` written by ``train.py``
when present; otherwise it runs a small in-memory demo pipeline (cached) so the
dashboard works on a fresh clone. ``RECSYS_ARTIFACTS_DIR`` overrides the folder.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.artifacts import DashboardBundle, ab_test_id, bundle_from_result, load_bundle  # noqa: E402
from src.dataset import SimConfig  # noqa: E402
from src.metrics import ndcg_at_k, topk_mean  # noqa: E402
from src.pareto import moo_scores  # noqa: E402
from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.ranker import RankerConfig  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent
ART_DIR = Path(os.environ.get("RECSYS_ARTIFACTS_DIR", REPO_DIR / "artifacts"))

# Fixed categorical colour per entity (validated reference palette, slot order).
COLORS: dict[str, str] = {
    "ips": "#2a78d6",  # blue
    "naive": "#eb6834",  # orange
    "ips_moo": "#1baf7a",  # aqua
    "logger": "#eda100",  # yellow
    "ips_clicked": "#e87ba4",  # magenta
    "ips_row": "#008300",  # green
    "random": "#8a8984",
    "oracle": "#52514e",
}
INK_MUTED = "#8a8984"
GUARD = "#e34948"

DEMO_CONFIG = PipelineConfig(
    sim=SimConfig(n_sessions=6000),
    ranker=replace(RankerConfig(), n_estimators=120),
    run_ablations=False,
    n_alphas=41,
    n_boot=1000,
    n_boot_propensity=200,
    eval_every=10,
    n_inspector_sessions=150,
)

st.set_page_config(page_title="Debiased LTR & MOO RecSys", layout="wide")


@st.cache_resource(show_spinner="No artifacts found - running a small demo pipeline (~10 s)...")
def demo_bundle() -> DashboardBundle:
    return bundle_from_result(run_pipeline(DEMO_CONFIG), source="demo")


@st.cache_data(show_spinner=False)
def artifacts_bundle(path: str, stamp: float) -> DashboardBundle | None:
    del stamp  # only part of the cache key: reload when metrics.json changes
    return load_bundle(Path(path))


def get_bundle() -> DashboardBundle:
    stamp_file = ART_DIR / "metrics.json"
    if stamp_file.exists():
        bundle = artifacts_bundle(str(ART_DIR), stamp_file.stat().st_mtime)
        if bundle is not None:
            return bundle
    return demo_bundle()


def fmt_pct(x: float, signed: bool = True) -> str:
    return f"{100 * x:+.2f}%" if signed else f"{100 * x:.2f}%"


def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def base_layout(fig: go.Figure, title: str, x: str, y: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


bundle = get_bundle()
summary = bundle.summary
cfg = summary["config"]
K = int(cfg["k"])
N_COL, F_COL = f"ndcg@{K}", f"freshness@{K}"
knee = summary["knee"]

# ---------------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------------
with st.sidebar:
    st.header("Run")
    if bundle.source == "demo":
        st.info("Demo mode: small in-memory run. Execute `python train.py` for the full experiment.")
    else:
        st.success(f"Loaded artifacts from `{ART_DIR.name}/`")
    st.markdown(
        f"""
- sessions: **{int(cfg['sim.n_sessions']):,}** (K = {int(cfg['sim.list_size'])})
- true bias: $p_k = k^{{-{cfg['sim.eta']}}}$
- IPS clip tau = **{cfg['tau']}**, scheme = `{cfg['ips_scheme']}`
- boosting rounds: {", ".join(f"{k}={v}" for k, v in summary['best_iterations'].items())}
- MOO guard-rail: NDCG@{K} drop <= {100 * float(cfg['max_ndcg_drop']):.1f}%
"""
    )
    st.caption("Ground truth (relevance, true p_k) exists only because the data is simulated.")

st.title("Debiased Learning-to-Rank & Multi-Objective RecSys")
st.caption(
    "Short-video feed: LambdaMART on position-biased clicks, IPS debiasing, "
    "counterfactual evaluation, offline A/B testing and a relevance/freshness Pareto front."
)

tabs = st.tabs(
    ["Data & position bias", "Model comparison", "Pareto front", "A/B test", "Session inspector"]
)

# ---------------------------------------------------------------------------------
# 1) data & position bias
# ---------------------------------------------------------------------------------
with tabs[0]:
    logs = summary["logs"]
    prop = bundle.propensities
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall CTR", fmt_pct(logs["ctr"], signed=False))
    c2.metric("Sessions with >= 1 click", fmt_pct(logs["share_sessions_with_click"], signed=False))
    c3.metric("corr(position, freshness) - logged", f"{logs['corr_position_freshness_logged']:+.2f}")
    c4.metric(
        "eta: estimated vs true",
        f"{summary['propensity']['eta_hat']:.3f}",
        delta=f"true {summary['propensity']['eta_true']}",
        delta_color="off",
    )

    norm_logged = prop["ctr_logged"] / prop["ctr_logged"].iloc[0]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prop["position"], y=prop["p_true"], name="true examination p_k",
            mode="lines", line=dict(color=INK_MUTED, dash="dash", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prop["position"], y=prop["p_hat"], name="estimated p_hat_k (randomized slice, 95% CI)",
            mode="lines+markers", line=dict(color=COLORS["ips"], width=2), marker=dict(size=8),
            error_y=dict(
                type="data", symmetric=False,
                array=prop["p_hat_ci_high"] - prop["p_hat"],
                arrayminus=prop["p_hat"] - prop["p_hat_ci_low"],
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prop["position"], y=norm_logged, name="logged CTR_k / CTR_1 (confounded)",
            mode="lines+markers", line=dict(color=COLORS["naive"], width=2), marker=dict(size=8, symbol="square"),
        )
    )
    fig.add_hline(
        y=float(summary["propensity"]["tau"]), line=dict(color=GUARD, width=1, dash="dot"),
        annotation_text=f"clip tau = {summary['propensity']['tau']}", annotation_position="top right",
    )
    base_layout(fig, "CTR by position vs examination curve", "position k", "relative examination / CTR")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        wfig = go.Figure(
            go.Bar(
                x=prop["position"], y=prop["ips_weight_clipped"], marker_color=COLORS["ips"],
                name="clipped IPS weight",
                hovertemplate="position %{x}<br>w = %{y:.2f}<extra></extra>",
            )
        )
        wfig.add_trace(
            go.Scatter(
                x=prop["position"], y=1.0 / prop["p_hat"].clip(lower=1e-3), name="unclipped 1/p_hat",
                mode="markers", marker=dict(color=COLORS["naive"], size=9, symbol="x"),
            )
        )
        base_layout(wfig, "IPS weight per logged position: 1/max(p_hat_k, tau)", "position k", "weight", height=320)
        st.plotly_chart(wfig, width="stretch")
    with right:
        st.markdown(
            f"""
**How to read this tab**

* The **orange** curve is what naive analytics sees: CTR collapses with position because
  of *both* lower examination and the old ranker putting its favourite (fresh) items on top.
* On the **randomized slice** ({summary['propensity']['n_sessions']:,} sessions) items are shuffled, so
  relevance is independent of position and `CTR_k / CTR_1` isolates the examination
  probability `p_k`. The **blue** estimate should track the dashed truth.
* Below `tau` the weight is capped at `1/tau = {1 / float(summary['propensity']['tau']):.2f}`: a little
  bias in exchange for much lower variance.
"""
        )
        st.dataframe(prop.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------
# 2) model comparison
# ---------------------------------------------------------------------------------
with tabs[1]:
    comp = bundle.comparison.copy()
    st.subheader("Test-set metrics")
    st.caption(
        f"`ndcg@{K}`, `ndcg@10`, `recall@{K}` use the hidden TRUE relevance. `ips_ndcg`, `snips_ndcg` "
        "and `click_ndcg` use only logged clicks (what you would have in production)."
    )
    st.dataframe(comp.round(4), hide_index=True, width="stretch")

    metric_options = [c for c in comp.columns if c != "policy"]
    m1, m2 = st.columns(2)
    with m1:
        metric = st.selectbox("Metric", metric_options, index=metric_options.index(N_COL))
        sub = comp[comp["policy"] != "oracle"] if metric.startswith(("ndcg", "recall")) else comp
        bfig = go.Figure(
            go.Bar(
                x=sub["policy"], y=sub[metric],
                marker_color=[COLORS.get(p, INK_MUTED) for p in sub["policy"]],
                text=[f"{v:.3f}" for v in sub[metric]], textposition="outside",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        )
        lo = float(sub[metric].min())
        hi = float(sub[metric].max())
        pad = 0.15 * (hi - lo + 1e-9)
        bfig.update_yaxes(range=[max(0.0, lo - pad), hi + pad])
        base_layout(bfig, f"{metric} by policy", "policy", metric, height=380)
        bfig.update_layout(showlegend=False)
        st.plotly_chart(bfig, width="stretch")
    with m2:
        imp = bundle.importance.set_index("feature")
        ifig = go.Figure()
        for model in [m for m in ("naive", "ips") if m in imp.columns]:
            ifig.add_trace(
                go.Bar(
                    y=imp.index, x=imp[model], name=model, orientation="h",
                    marker_color=COLORS[model],
                    hovertemplate="%{y}: %{x:.3f}<extra>" + model + "</extra>",
                )
            )
        base_layout(ifig, "Gain importance share (naive over-trusts freshness)", "share of total gain", "", height=380)
        ifig.update_layout(barmode="group")
        st.plotly_chart(ifig, width="stretch")

    st.subheader("Validation curves vs boosting round")
    curves = bundle.curves
    cm = st.selectbox(
        "Validation metric", sorted(curves["metric"].unique()),
        index=sorted(curves["metric"].unique()).index(f"snips_ndcg@{K}"),
    )
    cfig = go.Figure()
    for model, sub in curves[curves["metric"] == cm].groupby("model"):
        cfig.add_trace(
            go.Scatter(
                x=sub["iteration"], y=sub["value"], name=model, mode="lines",
                line=dict(color=COLORS.get(model, INK_MUTED), width=2),
            )
        )
        best = summary["best_iterations"].get(model)
        hit = sub[sub["iteration"] == best]
        if len(hit):
            cfig.add_trace(
                go.Scatter(
                    x=hit["iteration"], y=hit["value"], mode="markers", showlegend=False,
                    marker=dict(color=COLORS.get(model, INK_MUTED), size=11, symbol="diamond",
                                line=dict(color="white", width=2)),
                    hovertemplate=f"{model}: selected round %{{x}}<extra></extra>",
                )
            )
    base_layout(cfig, f"{cm} on validation (diamond = selected round)", "boosting round", cm, height=360)
    st.plotly_chart(cfig, width="stretch")
    st.caption(
        "Model selection never touches ground truth: the naive arm early-stops on click-NDCG, "
        "the IPS arms on SNIPS-NDCG computed from logged clicks."
    )

    wc = summary["weight_check"]
    st.subheader("Do sample weights really change LambdaMART?")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("unit weights vs no weights: max |score diff|", f"{wc['unit_vs_unweighted_max_abs_diff']:.1e}")
    w2.metric("IPS vs naive: Spearman of scores", f"{wc['ips_vs_naive_spearman']:.3f}")
    w3.metric(f"sessions whose top-{K} set changed", fmt_pct(wc[f'share_sessions_top{K}_changed'], signed=False))
    w4.metric("mean |rank shift|", f"{wc['mean_abs_rank_shift']:.2f}")

# ---------------------------------------------------------------------------------
# 3) Pareto front
# ---------------------------------------------------------------------------------
with tabs[2]:
    par = bundle.pareto
    alphas = par["alpha"].round(4).tolist()
    knee_alpha = min(alphas, key=lambda a: abs(a - knee["alpha"]))
    alpha = st.select_slider("alpha (weight of freshness)", options=alphas, value=knee_alpha, key="alpha_pareto")
    row = par.iloc[int(np.argmin(np.abs(par["alpha"] - alpha)))]
    base_row = par.iloc[int(np.argmin(np.abs(par["alpha"])))]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(f"NDCG@{K} (test)", f"{row[N_COL]:.4f}", delta=fmt_pct(-row["ndcg_drop_rel"]))
    p2.metric(f"Freshness@{K} (test)", f"{row[F_COL]:.4f}", delta=fmt_pct(row["freshness_gain_rel"]))
    p3.metric("selected alpha* (validation knee)", f"{knee['alpha']:.2f}")
    p4.metric("unconstrained knee alpha", f"{knee['unconstrained_alpha']:.2f}")

    guard = float(base_row[N_COL]) * (1 - float(knee["max_rel_drop"]))
    pfig = go.Figure()
    pfig.add_trace(
        go.Scatter(
            x=par[F_COL], y=par[N_COL], mode="lines+markers", name="IPS + MOO sweep (test)",
            line=dict(color=COLORS["ips"], width=2),
            marker=dict(size=[8 if p else 5 for p in par["is_pareto"]], color=COLORS["ips"]),
            customdata=par[["alpha", "ndcg_drop_rel", "freshness_gain_rel"]].to_numpy(),
            hovertemplate=(
                "alpha=%{customdata[0]:.2f}<br>NDCG=%{y:.4f} (drop %{customdata[1]:.2%})"
                "<br>Freshness=%{x:.4f} (gain %{customdata[2]:.2%})<extra></extra>"
            ),
        )
    )
    pfig.add_hline(
        y=guard, line=dict(color=GUARD, dash="dash", width=1.5),
        annotation_text=f"-{100 * float(knee['max_rel_drop']):.0f}% NDCG guard-rail",
        annotation_position="bottom left",
    )
    k_row = par.iloc[int(np.argmin(np.abs(par["alpha"] - knee["alpha"])))]
    pfig.add_trace(
        go.Scatter(
            x=[k_row[F_COL]], y=[k_row[N_COL]], mode="markers", name=f"knee alpha*={knee['alpha']:.2f}",
            marker=dict(symbol="star", size=20, color=COLORS["ips_moo"], line=dict(color="white", width=2)),
        )
    )
    pfig.add_trace(
        go.Scatter(
            x=[row[F_COL]], y=[row[N_COL]], mode="markers", name=f"slider alpha={alpha:.2f}",
            marker=dict(symbol="circle-open", size=18, color=COLORS["ips"], line=dict(width=3)),
        )
    )
    comp_idx = bundle.comparison.set_index("policy")
    for ref in ("naive", "logger"):
        if ref in comp_idx.index:
            pfig.add_trace(
                go.Scatter(
                    x=[comp_idx.loc[ref, F_COL]], y=[comp_idx.loc[ref, N_COL]], mode="markers+text",
                    name=f"{ref} model", text=[ref], textposition="top center",
                    marker=dict(symbol="diamond", size=12, color=COLORS[ref], line=dict(color="white", width=2)),
                )
            )
    base_layout(pfig, "Relevance vs freshness: Pareto front of the IPS ranker", f"Freshness@{K}", f"NDCG@{K} (true relevance)", height=480)
    st.plotly_chart(pfig, width="stretch")

    dfig = go.Figure(
        go.Scatter(
            x=par["alpha"], y=par["chord_distance"], mode="lines", line=dict(color=COLORS["ips"], width=2),
            name="distance to chord",
        )
    )
    dfig.add_vline(x=knee["alpha"], line=dict(color=COLORS["ips_moo"], width=2), annotation_text="alpha*")
    dfig.add_vline(x=knee["unconstrained_alpha"], line=dict(color=INK_MUTED, dash="dot"), annotation_text="unconstrained knee")
    base_layout(dfig, "Knee criterion: normalized distance to the chord between the extremes", "alpha", "distance", height=300)
    st.plotly_chart(dfig, width="stretch")
    st.caption(
        "alpha* is chosen on the VALIDATION sweep (max chord distance subject to the NDCG guard-rail) "
        "and only evaluated on test, so the guard-rail can be slightly exceeded on test."
    )

# ---------------------------------------------------------------------------------
# 4) A/B test
# ---------------------------------------------------------------------------------
with tabs[3]:
    tests = bundle.ab_tests
    labels = [f"{t['metric']}: {t['control']} -> {t['treatment']}" for t in tests]
    choice = st.selectbox("Comparison", labels, index=0)
    res = tests[labels.index(choice)]
    diffs = bundle.ab_boot[ab_test_id(res)].to_numpy()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("mean diff (B - A)", f"{res['mean_diff']:+.4f}", delta=f"CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]", delta_color="off")
    a2.metric(
        "relative lift", fmt_pct(res["rel_lift"]),
        delta=f"CI [{fmt_pct(res['rel_lift_ci_low'])}, {fmt_pct(res['rel_lift_ci_high'])}]", delta_color="off",
    )
    a3.metric("sessions (paired)", f"{res['n']:,}", delta=f"{res['n_nonzero']:,} non-zero diffs", delta_color="off")
    a4.metric("significant (95% CI excludes 0)", "yes" if res["significant"] else "no")

    hfig = go.Figure(
        go.Histogram(x=diffs, nbinsx=60, marker_color=COLORS["ips"], opacity=0.85, name="bootstrap mean diff")
    )
    for x, name in ((res["ci_low"], "2.5%"), (res["ci_high"], "97.5%")):
        hfig.add_vline(x=x, line=dict(color=INK_MUTED, dash="dash"), annotation_text=name)
    hfig.add_vline(x=0.0, line=dict(color=GUARD, width=2), annotation_text="0", annotation_position="top left")
    hfig.add_vline(x=res["mean_diff"], line=dict(color=COLORS["ips_moo"], width=2), annotation_text="observed")
    base_layout(
        hfig, f"Paired bootstrap ({res['n_boot']} resamples of sessions)",
        f"mean {res['metric']} difference ({res['treatment']} - {res['control']})", "count", height=380,
    )
    hfig.update_layout(showlegend=False, bargap=0.05)
    st.plotly_chart(hfig, width="stretch")

    st.markdown("**p-values (two-sided)**")
    st.dataframe(
        pd.DataFrame(
            {
                "test": ["paired bootstrap (percentile)", "paired Student t-test", "Wilcoxon signed-rank"],
                "statistic": [np.nan, res["t_stat"], res["wilcoxon_stat"]],
                "p-value": [fmt_p(res["p_bootstrap"]), fmt_p(res["t_pvalue"]), fmt_p(res["wilcoxon_pvalue"])],
                "tests": ["mean diff (no distributional assumption)", "mean diff (CLT / normality)", "symmetric location shift of diffs (ignores zeros)"],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "When the per-session differences are sparse and heavy-tailed (e.g. IPS-DCG, driven by a few "
        "high-weight clicks) the mean-based tests and Wilcoxon can disagree: they test different hypotheses."
    )
    st.markdown("**All offline A/B tests**")
    st.dataframe(pd.DataFrame(tests).round(5), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------
# 5) session inspector
# ---------------------------------------------------------------------------------
with tabs[4]:
    ins = bundle.inspector
    session_ids = ins["session_id"].unique().tolist()
    s1, s2 = st.columns([1, 2])
    with s1:
        sid = st.selectbox("Test session", session_ids, index=0)
    with s2:
        alpha_s = st.slider("alpha for MOO re-ranking", 0.0, 1.0, float(round(knee["alpha"], 2)), 0.01, key="alpha_session")
    sess = ins[ins["session_id"] == sid].reset_index(drop=True)
    sizes = np.array([len(sess)])
    s_ips = sess["score_ips"].to_numpy()
    s_moo = moo_scores(s_ips, sess["freshness"].to_numpy(), sizes, alpha_s)
    sess["rank_logged"] = sess["position"]
    sess["rank_naive"] = sess["score_naive"].rank(ascending=False, method="first").astype(int)
    sess["rank_ips"] = pd.Series(s_ips).rank(ascending=False, method="first").astype(int)
    sess["rank_moo"] = pd.Series(s_moo).rank(ascending=False, method="first").astype(int)

    rel = sess["relevance"].to_numpy(dtype=float)
    fresh = sess["freshness"].to_numpy()
    cols = st.columns(4)
    for col, (name, score) in zip(
        cols,
        (
            ("logged (old ranker)", -sess["position"].to_numpy(dtype=float)),
            ("naive LambdaMART", sess["score_naive"].to_numpy()),
            ("IPS LambdaMART", s_ips),
            (f"IPS + MOO (alpha={alpha_s:.2f})", s_moo),
        ),
    ):
        col.metric(
            name,
            f"NDCG@{K} {float(ndcg_at_k(rel, score, sizes, K)[0]):.3f}",
            delta=f"Freshness@{K} {float(topk_mean(fresh, score, sizes, K)[0]):.3f}",
            delta_color="off",
        )

    sfig = go.Figure()
    for _, r in sess.iterrows():
        sfig.add_trace(
            go.Scatter(
                x=["IPS (alpha=0)", f"MOO (alpha={alpha_s:.2f})"], y=[r["rank_ips"], r["rank_moo"]],
                mode="lines+markers+text", showlegend=False,
                text=[f"item {int(r['item_id'])} | rel {int(r['relevance'])}", f"fresh {r['freshness']:.2f}"],
                textposition=["middle left", "middle right"],
                line=dict(color=f"rgba(42,120,214,{0.25 + 0.75 * r['freshness']:.2f})", width=2 + 3 * r["freshness"]),
                marker=dict(size=9, color=COLORS["ips"]),
                hovertemplate=(
                    f"item {int(r['item_id'])}<br>relevance {int(r['relevance'])}<br>"
                    f"freshness {r['freshness']:.2f}<br>rank %{{y}}<extra></extra>"
                ),
            )
        )
    sfig.update_yaxes(autorange="reversed", dtick=1, title="rank")
    base_layout(sfig, "Rank changes after MOO re-ranking (thicker/darker = fresher)", "", "rank", height=460)
    sfig.update_xaxes(range=[-0.8, 1.8])
    st.plotly_chart(sfig, width="stretch")

    show = sess[
        ["item_id", "relevance", "freshness", "click", "rank_logged", "rank_naive", "rank_ips", "rank_moo"]
    ].sort_values("rank_moo")
    show["moo_shift"] = show["rank_ips"] - show["rank_moo"]
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    st.caption("`moo_shift` > 0: the item moved up thanks to freshness. `relevance` is the hidden ground-truth grade (0-4).")
