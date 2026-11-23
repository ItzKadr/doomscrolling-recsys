"""Fail-safe MLflow logging.

Every call is wrapped so that an MLflow problem (locked DB, missing package,
bad URI ...) only produces a warning: the pipeline and the ``artifacts/`` folder
never depend on the tracking server.

The store is a local SQLite DB inside the repository (``mlflow.db``) with run
artifacts under ``mlruns/``; browse it with
``mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

_INVALID_KEY_CHARS = re.compile(r"[^/\w.\- :]")


def sanitize_key(key: str) -> str:
    """MLflow keys allow ``[A-Za-z0-9_./ :-]``; map ``@`` to ``_at_`` and others to ``_``."""
    return _INVALID_KEY_CHARS.sub("_", key.replace("@", "_at_"))


def default_tracking_uri(repo_dir: Path) -> str:
    """``sqlite:///<repo>/mlflow.db`` (absolute, forward slashes; spaces are fine)."""
    return "sqlite:///" + (Path(repo_dir).resolve() / "mlflow.db").as_posix()


class SafeMlflowRun:
    """Context manager around one MLflow run that never raises."""

    def __init__(
        self,
        repo_dir: Path,
        experiment: str,
        run_name: str | None = None,
        enabled: bool = True,
        tracking_uri: str | None = None,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.experiment = experiment
        self.run_name = run_name
        self.enabled = enabled
        self.tracking_uri = tracking_uri or default_tracking_uri(self.repo_dir)
        self._mlflow: Any = None
        self.run_id: str | None = None

    # -- context manager -------------------------------------------------------------
    def __enter__(self) -> "SafeMlflowRun":
        if not self.enabled:
            return self
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            for noisy in ("alembic", "mlflow.store.db.utils", "mlflow.tracking.fluent"):
                logging.getLogger(noisy).setLevel(logging.WARNING)
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            exp = mlflow.get_experiment_by_name(self.experiment)
            if exp is None:
                artifact_root = (self.repo_dir / "mlruns").as_uri()
                mlflow.create_experiment(self.experiment, artifact_location=artifact_root)
            mlflow.set_experiment(self.experiment)
            run = mlflow.start_run(run_name=self.run_name)
            self.run_id = run.info.run_id
            self._mlflow = mlflow
            logger.info("MLflow run %s started (%s)", self.run_id, self.tracking_uri)
        except Exception as exc:
            logger.warning("MLflow disabled: %s", exc)
            self._mlflow = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        except Exception as err:
            logger.warning("MLflow end_run failed: %s", err)

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    # -- logging helpers -------------------------------------------------------------
    def _safe(self, what: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        if self._mlflow is None:
            return
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("MLflow %s failed: %s", what, exc)

    def log_params(self, params: dict[str, Any]) -> None:
        clean = {sanitize_key(k): str(v)[:5000] for k, v in params.items()}
        if self._mlflow is not None:
            self._safe("log_params", self._mlflow.log_params, clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {
            sanitize_key(k): float(v)
            for k, v in metrics.items()
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
        }
        if self._mlflow is not None and clean:
            self._safe("log_metrics", self._mlflow.log_metrics, clean, step=step)

    def log_metric_steps(self, records: list[tuple[str, float, int]]) -> None:
        """Log many ``(key, value, step)`` points in batches (one DB round-trip per 1000)."""
        if self._mlflow is None or not records:
            return
        try:
            import time

            from mlflow.entities import Metric
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            now = int(time.time() * 1000)
            metrics = [
                Metric(sanitize_key(key), float(val), now, int(step))
                for key, val, step in records
                if val is not None and math.isfinite(float(val))
            ]
            for start in range(0, len(metrics), 1000):
                client.log_batch(self.run_id, metrics=metrics[start : start + 1000])
        except Exception as exc:
            logger.warning("MLflow log_metric_steps failed: %s", exc)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._mlflow is not None:
            self._safe("set_tags", self._mlflow.set_tags, {sanitize_key(k): str(v) for k, v in tags.items()})

    def log_artifacts(self, local_dir: Path, artifact_path: str | None = None) -> None:
        if self._mlflow is not None:
            self._safe("log_artifacts", self._mlflow.log_artifacts, str(local_dir), artifact_path)
