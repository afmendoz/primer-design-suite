"""Training stage: feature table -> trained flagship model artifact.

Trains a model from the ladder in ``predictor.models`` on the featurized
dataset. Cross-validation is **grouped by gene/template** (config
``cv.grouping_key``) so a primer and its sequence neighbors never span train
and test folds — the project forbids ungrouped CV metrics. We report Spearman
correlation and RMSE (not bare R²).

The saved artifact matches ``primer_core.tools.score.score_candidate``'s
contract exactly (``{"model", "feature_names", "metadata"}``), closing the
loop between the flagship predictor and the copilot's scoring tool.

CLI-able: invoked directly or via ``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NON_FEATURE_COLS = ("primer_id", "template_id", "label")


def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    with open(config_path) as handle:
        return yaml.safe_load(handle)


def feature_columns(df: Any) -> list[str]:
    """Return the ordered feature column names (all but id/group/label)."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def grouped_oof_predictions(
    estimator_factory: Any,
    x: Any,
    y: Any,
    groups: Any,
    n_splits: int,
) -> Any:
    """Compute out-of-fold predictions with GroupKFold (no leakage).

    Args:
        estimator_factory: Zero-arg callable returning a fresh unfitted
            estimator for each fold.
        x: Feature matrix (n_samples, n_features).
        y: Target vector.
        groups: Group labels for each row (grouping key values).
        n_splits: Number of grouped folds.

    Returns:
        A numpy array of out-of-fold predictions aligned to ``x``.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    oof = np.full(shape=y.shape, fill_value=np.nan, dtype=float)

    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(x, y, groups):
        model = estimator_factory()
        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict(x[test_idx])
    return oof


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute Spearman correlation and RMSE (the reported regression pair)."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    spearman = float(spearmanr(y_true, y_pred).correlation)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"spearman": spearman, "rmse": rmse}


def _effective_n_splits(config_n_splits: int, n_groups: int) -> int:
    """Clamp n_splits to the number of groups (GroupKFold requirement)."""
    return max(2, min(config_n_splits, n_groups))


def train_model(
    feature_table_path: str | Path,
    model_name: str,
    output_model_path: str | Path,
    config_path: str | Path = "predictor/workflows/configs/config.yaml",
) -> dict[str, Any]:
    """Train a model with grouped CV and persist the flagship artifact.

    Args:
        feature_table_path: Path to the CSV feature table from ``featurize``.
        model_name: Registry name of the model to train (see
            ``predictor.models.MODEL_BUILDERS``).
        output_model_path: Destination path for the joblib artifact.
        config_path: Path to the pipeline config (model params, CV settings).

    Returns:
        A metrics dict with grouped-CV ``spearman``, ``rmse``, and per-fold
        breakdown. A suspiciously high score should be investigated, not
        celebrated (project convention).

    Raises:
        FileNotFoundError: If ``feature_table_path`` or ``config_path`` missing.
        ValueError: If ``model_name`` is unknown.
    """
    import joblib
    import pandas as pd

    from predictor.models import get_model

    feature_table_path = Path(feature_table_path)
    if not feature_table_path.exists():
        raise FileNotFoundError(f"feature table not found: {feature_table_path}")

    config = _load_config(config_path)
    grouping_key = config["cv"]["grouping_key"]
    model_params = config.get("model", {}).get("params", {})
    ladder_params = config.get("ladder", {}).get(model_name, model_params)

    df = pd.read_csv(feature_table_path)
    feat_cols = feature_columns(df)
    x = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=float)
    groups = df[grouping_key].to_numpy()

    n_groups = len(set(groups.tolist()))
    n_splits = _effective_n_splits(config["cv"]["n_splits"], n_groups)

    def factory() -> Any:
        return get_model(model_name, ladder_params)

    oof = grouped_oof_predictions(factory, x, y, groups, n_splits)
    metrics = regression_metrics(y, oof)
    metrics["per_fold"] = _per_fold_metrics(factory, x, y, groups, n_splits)
    metrics["n_splits"] = n_splits
    metrics["model_name"] = model_name

    final_model = factory()
    final_model.fit(x, y)

    provenance = _read_provenance(config)
    artifact = {
        "model": final_model,
        "feature_names": feat_cols,
        "metadata": {
            "model_name": model_name,
            "cv_metrics": {k: metrics[k] for k in ("spearman", "rmse", "per_fold", "n_splits")},
            "is_proxy_label": provenance.get("is_proxy_label", None),
            "grouping_key": grouping_key,
        },
    }
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_model_path)
    logger.info(
        "trained %s | grouped-CV spearman=%.4f rmse=%.4f (%d folds) -> %s",
        model_name,
        metrics["spearman"],
        metrics["rmse"],
        n_splits,
        output_model_path,
    )

    _log_mlflow(config, model_name, ladder_params, metrics)
    return metrics


def _per_fold_metrics(
    factory: Any, x: Any, y: Any, groups: Any, n_splits: int
) -> list[dict[str, float]]:
    import numpy as np
    from sklearn.model_selection import GroupKFold

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    folds: list[dict[str, float]] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = factory()
        model.fit(x[train_idx], y[train_idx])
        preds = model.predict(x[test_idx])
        folds.append(regression_metrics(y[test_idx], preds))
    return folds


def _read_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Best-effort read of the dataset provenance sidecar."""
    try:
        from primer_core.io.datasets import load_provenance

        dataset_path = config["paths"]["dataset"]
        return load_provenance(dataset_path).model_dump()
    except Exception as exc:  # noqa: BLE001 - provenance is advisory here
        logger.warning("could not read provenance: %s", exc)
        return {}


def _log_mlflow(
    config: dict[str, Any],
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Log params/metrics to MLflow if available; never break training."""
    mlflow_cfg = config.get("mlflow")
    if not mlflow_cfg:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "primer-design-suite"))
        with mlflow.start_run(run_name=f"train-{model_name}"):
            mlflow.log_param("model_name", model_name)
            for key, value in params.items():
                mlflow.log_param(key, value)
            mlflow.log_metric("cv_spearman", metrics["spearman"])
            mlflow.log_metric("cv_rmse", metrics["rmse"])
    except Exception as exc:  # noqa: BLE001 - MLflow is optional plumbing
        logger.warning("mlflow logging skipped: %s", exc)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a primer efficiency model.")
    parser.add_argument("--features", required=True, help="Path to feature table")
    parser.add_argument("--model", default=None, help="Model name (default: config model.name)")
    parser.add_argument("--output", required=True, help="Path to write trained model artifact")
    parser.add_argument(
        "--config",
        default="predictor/workflows/configs/config.yaml",
        help="Path to pipeline config YAML",
    )
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m predictor.pipeline.train``."""
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    model_name = args.model or _load_config(args.config)["model"]["name"]
    train_model(args.features, model_name, args.output, args.config)


if __name__ == "__main__":
    main()
