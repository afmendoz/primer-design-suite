"""Evaluation stage: trained model -> grouped CV metrics + ladder + SHAP.

Reports, honestly and grouped by template (never ungrouped — project convention):

1. Grouped-CV Spearman + RMSE for the flagship model.
2. The full modeling ladder (ElasticNet / RandomForest / XGBoost / LightGBM)
   under the same grouped CV, so baseline-vs-boosting is visible.
3. SHAP feature importances for the flagship if it is a tree model, with a
   biological sanity flag: the top features should include 3'-ΔG, dimer ΔG,
   hairpin ΔG, or the GC-clamp; if not, that is a leakage / featurization
   warning, not a result.

When the dataset's provenance marks the label as a proxy, a prominent note is
placed at the top of the report — proxy metrics are never validation on real
experimental data.

CLI-able: invoked directly or via ``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from predictor.models import LADDER_MODELS, get_model
from predictor.pipeline.train import (
    _effective_n_splits,
    _load_config,
    feature_columns,
    grouped_oof_predictions,
    regression_metrics,
)

logger = logging.getLogger(__name__)

_TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}
_SENSIBLE_FEATURES = {"three_prime_end_dg", "homodimer_dg", "hairpin_dg", "gc_clamp"}


def evaluate_model(
    model_path: str | Path,
    feature_table_path: str | Path,
    output_report_path: str | Path,
    config_path: str | Path = "predictor/workflows/configs/config.yaml",
) -> dict[str, Any]:
    """Compute grouped CV metrics, the ladder, and SHAP for the flagship model.

    Args:
        model_path: Path to the trained flagship joblib artifact.
        feature_table_path: Path to the CSV feature table.
        output_report_path: Destination path for the JSON report.
        config_path: Path to the pipeline config.

    Returns:
        The report dict (also written to ``output_report_path``).

    Raises:
        FileNotFoundError: If a required input path is missing.
    """
    import pandas as pd

    feature_table_path = Path(feature_table_path)
    if not feature_table_path.exists():
        raise FileNotFoundError(f"feature table not found: {feature_table_path}")

    config = _load_config(config_path)
    grouping_key = config["cv"]["grouping_key"]
    flagship_name = config["model"]["name"]

    df = pd.read_csv(feature_table_path)
    feat_cols = feature_columns(df)
    x = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=float)
    groups = df[grouping_key].to_numpy()
    n_groups = len(set(groups.tolist()))
    n_splits = _effective_n_splits(config["cv"]["n_splits"], n_groups)

    # 1. Flagship grouped-CV metrics.
    flagship_params = config.get("ladder", {}).get(flagship_name, {})
    flagship_oof = grouped_oof_predictions(
        lambda: get_model(flagship_name, flagship_params), x, y, groups, n_splits
    )
    flagship_metrics = regression_metrics(y, flagship_oof)

    # 2. Full ladder under the same grouped CV.
    ladder: dict[str, Any] = {}
    for name in LADDER_MODELS:
        params = config.get("ladder", {}).get(name, {})
        try:
            oof = grouped_oof_predictions(
                lambda n=name, p=params: get_model(n, p), x, y, groups, n_splits
            )
            ladder[name] = regression_metrics(y, oof)
        except Exception as exc:  # noqa: BLE001 - a missing model shouldn't sink the report
            ladder[name] = {"error": str(exc)}

    boosting_beat_baselines = _boosting_vs_baselines(ladder)

    # 3. SHAP on the fitted flagship model.
    shap_report = _shap_report(model_path, feat_cols, x, flagship_name)

    provenance = _read_provenance(config)
    is_proxy = bool(provenance.get("is_proxy_label", False))

    report: dict[str, Any] = {
        "proxy_label_warning": (
            (
                "PROXY LABEL — metrics are NOT on experimental data. "
                + str(provenance.get("notes", ""))
            )
            if is_proxy
            else None
        ),
        "is_proxy_label": is_proxy,
        "grouping_key": grouping_key,
        "n_splits": n_splits,
        "n_samples": int(len(df)),
        "n_groups": n_groups,
        "flagship_model": flagship_name,
        "flagship_cv": flagship_metrics,
        "ladder": ladder,
        "boosting_beat_baselines": boosting_beat_baselines,
        "shap": shap_report,
        "feature_names": feat_cols,
    }

    output_report_path = Path(output_report_path)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    logger.info("wrote evaluation report -> %s", output_report_path)
    return report


def _boosting_vs_baselines(ladder: dict[str, Any]) -> dict[str, Any]:
    """Compare best boosting vs best baseline Spearman, stated plainly."""

    def best_spearman(names: tuple[str, ...]) -> tuple[str | None, float]:
        best_name, best_val = None, float("-inf")
        for n in names:
            m = ladder.get(n, {})
            if "spearman" in m and m["spearman"] is not None and m["spearman"] > best_val:
                best_name, best_val = n, m["spearman"]
        return best_name, best_val

    base_name, base_val = best_spearman(("elasticnet", "random_forest"))
    boost_name, boost_val = best_spearman(("xgboost", "lightgbm"))
    if base_name is None or boost_name is None:
        return {"verdict": "incomplete ladder"}
    beat = boost_val > base_val
    return {
        "best_baseline": base_name,
        "best_baseline_spearman": base_val,
        "best_boosting": boost_name,
        "best_boosting_spearman": boost_val,
        "boosting_beat_baselines": beat,
        "verdict": (
            f"{boost_name} (boosting) {'beats' if beat else 'does NOT beat'} the best "
            f"baseline {base_name} on grouped-CV Spearman "
            f"({boost_val:.4f} vs {base_val:.4f})"
        ),
    }


def _shap_report(
    model_path: str | Path, feat_cols: list[str], x: Any, flagship_name: str
) -> dict[str, Any]:
    """Compute mean|SHAP| feature ranking for a tree flagship, else skip."""
    if flagship_name not in _TREE_MODELS:
        return {"status": "skipped", "reason": f"{flagship_name} is not a tree model"}

    model_path = Path(model_path)
    if not model_path.exists():
        return {"status": "skipped", "reason": f"model artifact not found: {model_path}"}

    try:
        import joblib
        import numpy as np
        import shap

        artifact = joblib.load(model_path)
        model = artifact["model"] if isinstance(artifact, dict) else artifact

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranking = sorted(zip(feat_cols, mean_abs.tolist()), key=lambda kv: kv[1], reverse=True)
        top_features = [name for name, _ in ranking]
        top3 = set(top_features[:3])
        sensible = bool(top3 & _SENSIBLE_FEATURES)
        result: dict[str, Any] = {
            "status": "ok",
            "mean_abs_shap": {name: val for name, val in ranking},
            "top_features": top_features,
            "top3": top_features[:3],
            "sensible_top_features": sensible,
        }
        if not sensible:
            result["warning"] = (
                "Top-3 SHAP features do not include any of "
                f"{sorted(_SENSIBLE_FEATURES)}; this may indicate leakage or a "
                "featurization bug (a biological sanity check)."
            )
        return result
    except Exception as exc:  # noqa: BLE001 - SHAP is optional / model-dependent
        return {"status": "skipped", "reason": str(exc)}


def _read_provenance(config: dict[str, Any]) -> dict[str, Any]:
    try:
        from primer_core.io.datasets import load_provenance

        return load_provenance(config["paths"]["dataset"]).model_dump()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read provenance: %s", exc)
        return {}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained primer efficiency model.")
    parser.add_argument("--model", required=True, help="Path to trained model artifact")
    parser.add_argument("--features", required=True, help="Path to feature table")
    parser.add_argument("--output", required=True, help="Path to write evaluation report")
    parser.add_argument(
        "--config",
        default="predictor/workflows/configs/config.yaml",
        help="Path to pipeline config YAML",
    )
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m predictor.pipeline.evaluate``."""
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    evaluate_model(args.model, args.features, args.output, args.config)


if __name__ == "__main__":
    main()
