"""Optional, failure-safe MLflow tracking for the aptamer pipeline."""
from __future__ import annotations

import atexit
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "mlflow.db"


def mlflow_enabled_from_env() -> bool:
    return os.getenv("MLFLOW_ENABLED", "1").strip().lower() not in FALSE_VALUES


def default_tracking_uri(db_path: Path | None = None) -> str:
    """Absolute SQLite URI rooted in the project directory (cwd-independent)."""
    path = (db_path or DEFAULT_DB_PATH).resolve()
    # Keep spaces as-is. Percent-encoding creates a literal "%20" folder on Windows.
    return f"sqlite:///{path.as_posix()}"


def flatten_params(
    values: Mapping[str, Any],
    prefix: str = "",
    max_length: int = 500,
) -> dict[str, str | int | float | bool]:
    """Flatten nested config and convert unsupported MLflow param values."""
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_params(value, full_key, max_length=max_length))
            continue
        if value is None:
            normalized: str | int | float | bool = "null"
        elif isinstance(value, (str, int, float, bool)):
            normalized = value
        elif isinstance(value, Path):
            normalized = str(value)
        else:
            try:
                normalized = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                normalized = repr(value)
        if isinstance(normalized, str) and len(normalized) > max_length:
            normalized = normalized[: max_length - 3] + "..."
        flattened[full_key] = normalized
    return flattened


def numeric_metrics(values: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        metrics[f"{prefix}.{key}" if prefix else str(key)] = number
    return metrics


class AptamerRunTracker:
    """Thin MLflow adapter that degrades to a no-op on any tracking failure."""

    def __init__(
        self,
        experiment_name: str = "aptamer-nonbinder",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        enabled: bool | None = None,
        pipeline_id: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.pipeline_id = pipeline_id or uuid.uuid4().hex[:12]
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.enabled = mlflow_enabled_from_env() if enabled is None else enabled
        self.active = False
        self.run_id: str | None = None
        self._mlflow = None
        self._owns_run = False
        self._closed = False
        self.disabled_reason: str | None = None

    def _disable(self, reason: str) -> None:
        if self.enabled and self.disabled_reason is None:
            print(f"MLflow disabled for this run: {reason}")
            print(
                "  Tip: install with `pip install mlflow` in the SAME python env "
                "used for `python \"Contrast Learning.py\"`."
            )
            print(f"  Then open UI with:")
            print(f"    mlflow ui --backend-store-uri {default_tracking_uri()}")
        self.enabled = False
        self.active = False
        self.disabled_reason = reason

    def start(self, tags: Mapping[str, Any] | None = None) -> "AptamerRunTracker":
        if not self.enabled or self.active:
            if not self.enabled and self.disabled_reason is None:
                print("MLflow tracking is OFF (MLFLOW_ENABLED=0)")
            return self
        try:
            import mlflow

            self._mlflow = mlflow
            uri = self.tracking_uri or default_tracking_uri()
            self.tracking_uri = uri
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.experiment_name)

            active_run = mlflow.active_run()
            if active_run is None:
                active_run = mlflow.start_run(run_name=self.run_name)
                self._owns_run = True
            self.run_id = active_run.info.run_id
            self.active = True
            mlflow.set_tag("pipeline_id", self.pipeline_id)
            mlflow.set_tag("tracking_schema", "aptamer-pipeline-v1")
            if tags:
                self.set_tags(tags)
            atexit.register(self._close_unfinished)
            print(
                f"MLflow run: {self.run_id} | experiment={self.experiment_name} "
                f"| uri={uri}"
            )
            print(
                f"  View later: mlflow ui --backend-store-uri {uri}"
            )
        except ImportError:
            self._disable("package 'mlflow' is not installed")
        except Exception as exc:
            self._disable(f"initialization failed: {exc}")
        return self

    def _safe(self, action: str, callback) -> None:
        if not self.active or self._mlflow is None:
            return
        try:
            callback()
        except Exception as exc:
            print(f"MLflow warning ({action}): {exc}")

    def log_params(self, values: Mapping[str, Any], prefix: str = "") -> None:
        params = flatten_params(values, prefix=prefix)
        self._safe("log_params", lambda: self._mlflow.log_params(params))

    def log_metrics(
        self,
        values: Mapping[str, Any],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        metrics = numeric_metrics(values, prefix=prefix)
        if metrics:
            self._safe(
                "log_metrics",
                lambda: self._mlflow.log_metrics(metrics, step=step),
            )

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        normalized = {key: str(value) for key, value in tags.items()}
        self._safe("set_tags", lambda: self._mlflow.set_tags(normalized))

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        artifact = Path(path)
        if not artifact.exists():
            return
        self._safe(
            "log_artifact",
            lambda: self._mlflow.log_artifact(str(artifact), artifact_path),
        )

    def log_dict(
        self,
        values: Mapping[str, Any],
        artifact_file: str,
    ) -> None:
        self._safe(
            "log_dict",
            lambda: self._mlflow.log_dict(dict(values), artifact_file),
        )

    def close(self, status: str = "FINISHED") -> None:
        if self._closed:
            return
        if self.active and self._mlflow is not None and self._owns_run:
            try:
                self._mlflow.end_run(status=status)
            except Exception as exc:
                print(f"MLflow warning (end_run): {exc}")
        self.active = False
        self._closed = True

    def _close_unfinished(self) -> None:
        if not self._closed:
            self.close(status="KILLED")


def create_pipeline_tracker(
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> AptamerRunTracker:
    return AptamerRunTracker(run_name=run_name).start(tags=tags)
