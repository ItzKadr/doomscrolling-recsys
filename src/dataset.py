"""Synthetic short-video clickstream simulator.

The simulator produces *logged* feed impressions with the two classic biases of
implicit feedback in ranking systems:

1. **Position bias** (examination hypothesis). A user clicks item ``i`` shown at
   position ``k`` iff she *examines* the slot and finds the item *attractive*::

       P(click_i = 1 | k, rel_i) = p_k * gamma(rel_i),     p_k = k^(-eta)

   with ``gamma(r) = eps + (1 - eps) * (2^r - 1) / (2^r_max - 1)``.

2. **Confounding by the logging policy.** The old production ranker that
   decided the positions boosts *fresh* items, so freshness is correlated with
   position and therefore with clicks, even though freshness carries only a
   small amount of true relevance.

Ground truth (graded relevance 0..4 and the true examination curve ``p_k``) is
kept next to the logs so that models trained on biased clicks can be evaluated
without bias.

A small slice of traffic is served with **uniformly randomized** lists
(``split == "rand"``); it is the intervention data used by
:func:`src.ips.estimate_propensities` to identify ``p_k``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Model input features, in a fixed order (LightGBM column order).
FEATURE_COLUMNS: tuple[str, ...] = (
    "emb_affinity",  # noisy two-tower dot product user . item
    "category_match",  # item category == user's favourite category
    "creator_follow",  # user follows the item's creator
    "quality_proxy",  # noisy historical completion-rate logit of the item
    "popularity",  # log(1 + views); grows with quality *and* with age
    "freshness",  # exp(-age_days / tau), 1 = brand new
    "duration_z",  # standardized log video length (irrelevant by construction)
    "user_activity",  # log activity of the user (constant within a session)
)

SPLITS: tuple[str, ...] = ("rand", "train", "val", "test")


@dataclass(frozen=True)
class SimConfig:
    """Configuration of the synthetic world, logging policy and click model."""

    n_users: int = 3000
    n_items: int = 4000
    n_creators: int = 400
    n_categories: int = 12
    latent_dim: int = 8
    n_sessions: int = 30_000
    list_size: int = 10  # K items shown per session
    # click model
    eta: float = 1.0  # position-bias severity, p_k = k^-eta
    click_noise: float = 0.05  # eps: click probability of an irrelevant item
    max_grade: int = 4
    grade_quantiles: tuple[float, ...] = (0.45, 0.70, 0.85, 0.94)
    # true relevance model
    relevance_noise: float = 0.5
    true_fresh_effect: float = 0.2  # small genuine preference for fresh content
    # observation noise of features
    embedding_noise: float = 0.6
    quality_noise: float = 0.5
    # catalogue
    mean_age_days: float = 10.0
    freshness_tau_days: float = 7.0
    # candidate retrieval (Gumbel top-K over two-tower scores)
    retrieval_temperature: float = 1.0
    # old logging policy: score = w_rel * relevance proxy + w_fresh * freshness + noise
    logger_rel_weight: float = 1.0
    logger_fresh_weight: float = 2.5
    logger_noise: float = 0.5
    # session splits
    rand_frac: float = 0.10  # randomized-intervention slice (propensity estimation)
    val_frac: float = 0.15
    test_frac: float = 0.20
    seed: int = 42

    def true_propensities(self) -> np.ndarray:
        """True examination probabilities ``p_k = k^-eta`` for k = 1..K."""
        return np.arange(1, self.list_size + 1, dtype=float) ** (-self.eta)

    def to_dict(self) -> dict[str, object]:
        """Flat dict for logging."""
        return asdict(self)


@dataclass
class World:
    """Latent users and items of the simulated platform."""

    user_emb: np.ndarray  # (U, d) true taste vectors
    user_emb_obs: np.ndarray  # (U, d) noisy two-tower user embeddings
    user_fav_cat: np.ndarray  # (U,)
    user_activity: np.ndarray  # (U,) log activity
    follows: np.ndarray  # (U, C) bool
    item_emb: np.ndarray  # (I, d)
    item_emb_obs: np.ndarray  # (I, d)
    item_cat: np.ndarray  # (I,)
    item_creator: np.ndarray  # (I,)
    item_quality: np.ndarray  # (I,) true quality
    item_quality_obs: np.ndarray  # (I,)
    item_age_days: np.ndarray  # (I,)
    item_freshness: np.ndarray  # (I,) in (0, 1]
    item_popularity: np.ndarray  # (I,)
    item_duration_z: np.ndarray  # (I,)


@dataclass
class ClickLogs:
    """Simulated impression logs plus the ground truth needed for unbiased evaluation.

    ``df`` has one row per (session, shown item), rows are contiguous per
    session and sorted by logged position (1 = top of the feed).
    """

    df: pd.DataFrame
    true_propensities: np.ndarray
    config: SimConfig

    def split(self, name: str) -> pd.DataFrame:
        """Rows of one split (``rand``/``train``/``val``/``test``), index reset."""
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        return self.df[self.df["split"] == name].reset_index(drop=True)


def group_sizes(df: pd.DataFrame, key: str = "session_id") -> np.ndarray:
    """Sizes of contiguous groups in ``df[key]`` (rows must be grouped contiguously)."""
    values = df[key].to_numpy()
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    edges = np.concatenate(([0], boundaries, [values.size]))
    return np.diff(edges).astype(np.int64)


def attractiveness(grades: np.ndarray, eps: float, max_grade: int) -> np.ndarray:
    """``gamma(r) = eps + (1 - eps) (2^r - 1) / (2^r_max - 1)``: P(click | examined, r)."""
    grades = np.asarray(grades, dtype=float)
    return eps + (1.0 - eps) * (np.exp2(grades) - 1.0) / (2.0**max_grade - 1.0)


def make_world(cfg: SimConfig, rng: np.random.Generator) -> World:
    """Sample latent users, items and creators."""
    d = cfg.latent_dim
    user_emb = rng.normal(size=(cfg.n_users, d))
    item_emb = rng.normal(size=(cfg.n_items, d))
    # Observed (two-tower) embeddings: fixed per entity noise, like a learned model.
    user_emb_obs = user_emb + cfg.embedding_noise * rng.normal(size=user_emb.shape)
    item_emb_obs = item_emb + cfg.embedding_noise * rng.normal(size=item_emb.shape)

    item_quality = rng.normal(size=cfg.n_items)
    item_quality_obs = item_quality + cfg.quality_noise * rng.normal(size=cfg.n_items)
    item_age_days = rng.exponential(cfg.mean_age_days, size=cfg.n_items)
    item_freshness = np.exp(-item_age_days / cfg.freshness_tau_days)
    # Popularity accumulates with quality and with time on the platform.
    log_views = (
        3.0
        + 0.8 * item_quality
        + 0.9 * np.log1p(item_age_days)
        + 0.5 * rng.normal(size=cfg.n_items)
    )
    item_popularity = np.log1p(np.exp(log_views))
    log_duration = rng.normal(np.log(25.0), 0.6, size=cfg.n_items)
    item_duration_z = (log_duration - log_duration.mean()) / log_duration.std()

    follows = rng.random((cfg.n_users, cfg.n_creators)) < 0.03
    return World(
        user_emb=user_emb,
        user_emb_obs=user_emb_obs,
        user_fav_cat=rng.integers(0, cfg.n_categories, size=cfg.n_users),
        user_activity=rng.normal(0.0, 0.7, size=cfg.n_users),
        follows=follows,
        item_emb=item_emb,
        item_emb_obs=item_emb_obs,
        item_cat=rng.integers(0, cfg.n_categories, size=cfg.n_items),
        item_creator=rng.integers(0, cfg.n_creators, size=cfg.n_items),
        item_quality=item_quality,
        item_quality_obs=item_quality_obs,
        item_age_days=item_age_days,
        item_freshness=item_freshness,
        item_popularity=item_popularity,
        item_duration_z=item_duration_z,
    )


def retrieve_candidates(
    world: World,
    users: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
    chunk: int = 1024,
) -> np.ndarray:
    """Candidate generation: Gumbel-top-K sampling over two-tower scores.

    ``argtop_K(s_ij / T + G_ij)`` with ``G ~ Gumbel(0, 1)`` draws K items without
    replacement with probabilities proportional to ``softmax(s / T)``.
    Returns an ``(n_sessions, K)`` array of item ids.
    """
    d = cfg.latent_dim
    k = cfg.list_size
    out = np.empty((users.size, k), dtype=np.int64)
    for start in range(0, users.size, chunk):
        u = users[start : start + chunk]
        scores = world.user_emb_obs[u] @ world.item_emb_obs.T / np.sqrt(d)
        gumbel = rng.gumbel(size=scores.shape)
        out[start : start + chunk] = np.argpartition(
            -(scores / cfg.retrieval_temperature + gumbel), k - 1, axis=1
        )[:, :k]
    return out


def _assign_splits(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    n = cfg.n_sessions
    n_rand = int(round(cfg.rand_frac * n))
    n_val = int(round(cfg.val_frac * n))
    n_test = int(round(cfg.test_frac * n))
    n_train = n - n_rand - n_val - n_test
    if min(n_rand, n_val, n_test, n_train) <= 0:
        raise ValueError("split fractions leave an empty split; increase n_sessions")
    labels = np.array(
        ["rand"] * n_rand + ["train"] * n_train + ["val"] * n_val + ["test"] * n_test,
        dtype=object,
    )
    return labels[rng.permutation(n)]


def simulate(cfg: SimConfig | None = None) -> ClickLogs:
    """Generate impression logs, clicks and ground truth.

    Pipeline per session ``s`` with user ``u``:

    1. retrieve K candidates (Gumbel top-K over noisy two-tower scores);
    2. true latent relevance
       ``l = u.v/sqrt(d) + 0.6 q + 0.8 cat_match + 0.6 follow + b_f f + N(0, s^2)``
       is cut into grades 0..4 by global quantiles;
    3. the logging policy orders the list by
       ``w_rel * z(emb_affinity + 0.5 quality_proxy) + w_fresh * freshness + noise``
       (randomized slice: uniform random permutation);
    4. clicks ``~ Bernoulli(p_k * gamma(grade))``.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg, rng)
    S, K, d = cfg.n_sessions, cfg.list_size, cfg.latent_dim

    activity_w = np.exp(world.user_activity)
    users = rng.choice(cfg.n_users, size=S, p=activity_w / activity_w.sum())
    items = retrieve_candidates(world, users, cfg, rng)  # (S, K)
    u_rep = np.repeat(users[:, None], K, axis=1)

    # ---- ground truth -------------------------------------------------------
    dot_true = np.einsum("skd,skd->sk", world.user_emb[u_rep], world.item_emb[items]) / np.sqrt(d)
    cat_match = (world.item_cat[items] == world.user_fav_cat[u_rep]).astype(float)
    follow = world.follows[u_rep, world.item_creator[items]].astype(float)
    freshness = world.item_freshness[items]
    latent = (
        dot_true
        + 0.6 * world.item_quality[items]
        + 0.8 * cat_match
        + 0.6 * follow
        + cfg.true_fresh_effect * freshness
        + cfg.relevance_noise * rng.normal(size=(S, K))
    )
    thresholds = np.quantile(latent, cfg.grade_quantiles)
    grades = np.digitize(latent, thresholds).astype(np.int64)  # 0..len(quantiles)

    # ---- observable features -------------------------------------------------
    emb_aff = (
        np.einsum("skd,skd->sk", world.user_emb_obs[u_rep], world.item_emb_obs[items]) / np.sqrt(d)
    )
    feats = {
        "emb_affinity": emb_aff,
        "category_match": cat_match,
        "creator_follow": follow,
        "quality_proxy": world.item_quality_obs[items],
        "popularity": world.item_popularity[items],
        "freshness": freshness,
        "duration_z": world.item_duration_z[items],
        "user_activity": world.user_activity[u_rep],
    }

    # ---- logging policy (old ranker) -----------------------------------------
    rel_proxy = emb_aff + 0.5 * world.item_quality_obs[items]
    rel_proxy_z = (rel_proxy - rel_proxy.mean()) / rel_proxy.std()
    log_score = (
        cfg.logger_rel_weight * rel_proxy_z
        + cfg.logger_fresh_weight * freshness
        + cfg.logger_noise * rng.normal(size=(S, K))
    )
    split = _assign_splits(cfg, rng)
    is_rand = split == "rand"
    order = np.argsort(-log_score, axis=1, kind="stable")
    order[is_rand] = np.argsort(rng.random((int(is_rand.sum()), K)), axis=1)

    def by_order(a: np.ndarray) -> np.ndarray:
        return np.take_along_axis(a, order, axis=1)

    items = by_order(items)
    grades = by_order(grades)
    feats = {name: by_order(val) for name, val in feats.items()}
    log_score = by_order(log_score)

    # ---- click model -----------------------------------------------------------
    p_true = cfg.true_propensities()
    examine = np.broadcast_to(p_true, (S, K))
    gamma = attractiveness(grades, cfg.click_noise, cfg.max_grade)
    clicks = (rng.random((S, K)) < examine * gamma).astype(np.int64)

    df = pd.DataFrame(
        {
            "session_id": np.repeat(np.arange(S), K),
            "user_id": np.repeat(users, K),
            "item_id": items.ravel(),
            "position": np.tile(np.arange(1, K + 1), S),
            "click": clicks.ravel(),
            "relevance": grades.ravel(),
            "logger_score": log_score.ravel(),
            "split": np.repeat(split, K),
            **{name: val.ravel() for name, val in feats.items()},
        }
    )
    logger.info(
        "simulated %d sessions x %d items: CTR=%.4f, mean grade=%.2f",
        S,
        K,
        df["click"].mean(),
        df["relevance"].mean(),
    )
    return ClickLogs(df=df, true_propensities=p_true, config=cfg)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Model input matrix (float64) in :data:`FEATURE_COLUMNS` order."""
    return df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
