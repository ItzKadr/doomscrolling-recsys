"""Debiased Learning-to-Rank & Multi-Objective RecSys Engine (short-video feed).

Modules
-------
dataset   synthetic clickstream simulator (position bias + freshness confounding)
ips       propensity estimation (result randomization), clipped IPS weights,
          counterfactual IPS / SNIPS-NDCG estimators
metrics   vectorized grouped ranking metrics (DCG, NDCG, Recall, top-k means)
ranker    LightGBM LambdaMART helpers (naive vs IPS-weighted training sets)
pareto    relevance-vs-freshness multi-objective re-ranking, Pareto front, knee point
stats     offline A/B testing (paired bootstrap, paired t-test, Wilcoxon)
pipeline  end-to-end orchestration used by ``train.py`` and ``app.py``
artifacts serialisation of pipeline outputs into a dashboard bundle
tracking  fail-safe MLflow logging
"""

from __future__ import annotations

import os

# LightGBM and other OpenMP users can clash on Windows ("libiomp5md.dll already initialized").
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
