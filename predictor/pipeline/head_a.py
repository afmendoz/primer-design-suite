"""Head A (openPrimeR): featurize + train the classification and regression heads.

Source A anchors two in-domain heads on designed IGHV primers:
  * classification  -> amplified (efficiency >= threshold), reported with
    ROC-AUC / PR-AUC / MCC (the label is ~14% positive, so accuracy is useless);
  * regression      -> continuous primer efficiency, reported with Spearman/RMSE.

Both use grouped CV by ``template_id`` (no primer/template leakage across folds)
and the shared ``primer_core`` featurizer, including the primer-template
annealing features. Fitted-on-all models are saved in the ``score_candidate``
artifact contract so the copilot can score in-domain.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from predictor.models import (
    CLASSIFIER_LADDER,
    LADDER_MODELS,
    get_classifier,
    get_model,
)
from primer_core.featurize import ANNEALING_FEATURES, INTRINSIC_FEATURES, featurize_primer
from primer_core.io.openprimer import load_openprimer

FEATURES = INTRINSIC_FEATURES + ANNEALING_FEATURES
MATRIX = "data/public/openprimer/feature_matrix.csv"
TEMPLATES = "data/public/openprimer/templates/Homo_sapiens_IGH_functional_exon.fasta"


def build_feature_table(matrix_path: str = MATRIX, templates_path: str = TEMPLATES) -> pd.DataFrame:
    """Load openPrimeR and compute the shared feature set for every row."""
    df = load_openprimer(matrix_path, templates_path)
    feats = [featurize_primer(r.primer_seq, r.template_seq) for r in df.itertuples()]
    fx = pd.DataFrame(feats, columns=FEATURES)
    return pd.concat([df.reset_index(drop=True), fx], axis=1)


def _oof(estimator_fn, X, y, groups, n_splits, proba: bool):
    """Grouped-CV out-of-fold predictions (probabilities if ``proba``)."""
    oof = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        est = estimator_fn()
        est.fit(X[tr], y[tr])
        oof[te] = est.predict_proba(X[te])[:, 1] if proba else est.predict(X[te])
    return oof


def _ece(y, proba, n_bins=10) -> float:
    """Expected calibration error: gap between predicted P and observed rate."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(proba, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(proba[m].mean() - y[m].mean())
    return float(ece)


def run_classification(X, y, groups, n_splits=5) -> dict[str, dict[str, float]]:
    out = {}
    for name in CLASSIFIER_LADDER:
        proba = _oof(lambda n=name: get_classifier(n), X, y, groups, n_splits, proba=True)
        pred = (proba >= 0.5).astype(int)
        out[name] = {
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "mcc": float(matthews_corrcoef(y, pred)),
            # Calibration: is a predicted P(amplify) trustworthy as a probability?
            "brier": float(brier_score_loss(y, proba)),
            "ece": _ece(y, proba),
        }
    return out


def run_regression(X, y, groups, n_splits=5) -> dict[str, dict[str, float]]:
    out = {}
    params = {
        "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5},
        "random_forest": {"n_estimators": 300, "random_state": 0},
        "xgboost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
        "lightgbm": {"n_estimators": 300, "verbose": -1},
    }
    for name in LADDER_MODELS:
        oof = _oof(lambda n=name: get_model(n, params[n]), X, y, groups, n_splits, proba=False)
        # Split-conformal half-width: the 90th percentile of |residual| gives a
        # marginal 90% prediction interval (pred +/- q90) for a new primer.
        out[name] = {
            "spearman": float(spearmanr(y, oof).correlation),
            "rmse": float(np.sqrt(mean_squared_error(y, oof))),
            "conformal_q90": float(np.quantile(np.abs(y - oof), 0.90)),
        }
    return out


def _shap_top(model, X, feature_names, k=5):
    try:
        import shap

        vals = shap.TreeExplainer(model).shap_values(X)
        if isinstance(vals, list):  # classifier -> per-class; take positive
            vals = vals[-1]
        imp = np.abs(vals).mean(axis=0)
        order = np.argsort(imp)[::-1][:k]
        return [feature_names[i] for i in order]
    except Exception as exc:  # noqa: BLE001
        return [f"shap unavailable: {exc}"]


def main() -> None:
    tbl = build_feature_table()
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    tbl.to_csv("data/reports/features_a.csv", index=False)  # generated -> local-only
    X = tbl[FEATURES].to_numpy(dtype=float)
    groups = tbl["template_id"].to_numpy()
    y_cls = tbl["amplified"].to_numpy(dtype=int)
    y_reg = tbl["efficiency"].to_numpy(dtype=float)
    print(
        f"rows={len(tbl)}  templates={tbl['template_id'].nunique()}  "
        f"positives={y_cls.mean():.1%}  features={len(FEATURES)}\n"
    )

    print("== Classification head (amplify y/n), grouped CV by template ==")
    cls = run_classification(X, y_cls, groups)
    print(f"{'model':<15}{'ROC-AUC':>9}{'PR-AUC':>9}{'MCC':>8}{'Brier':>8}{'ECE':>7}")
    for m, d in cls.items():
        print(
            f"{m:<15}{d['roc_auc']:>9.3f}{d['pr_auc']:>9.3f}{d['mcc']:>8.3f}"
            f"{d['brier']:>8.3f}{d['ece']:>7.3f}"
        )

    print("\n== Regression head (efficiency), grouped CV by template ==")
    reg = run_regression(X, y_reg, groups)
    print(f"{'model':<15}{'Spearman':>10}{'RMSE':>8}{'q90(+/-)':>10}")
    for m, d in reg.items():
        print(f"{m:<15}{d['spearman']:>10.3f}{d['rmse']:>8.3f}{d['conformal_q90']:>10.3f}")

    # fit best-on-all + save in the score_candidate contract
    best_cls = max(cls, key=lambda m: cls[m]["pr_auc"])
    best_reg = max(reg, key=lambda m: reg[m]["spearman"])
    # Isotonic-calibrate the final classifier so predict_proba is a trustworthy
    # probability (the report keeps the pre-calibration Brier/ECE it started from).
    from sklearn.calibration import CalibratedClassifierCV

    clf = CalibratedClassifierCV(get_classifier(best_cls), method="isotonic", cv=5)
    clf.fit(X, y_cls)
    rgr = get_model(
        best_reg,
        (
            {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}
            if best_reg in ("xgboost", "lightgbm")
            else {"n_estimators": 300} if best_reg == "random_forest" else {"alpha": 0.01}
        ),
    )
    rgr.fit(X, y_reg)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_names": FEATURES,
            "metadata": {
                "head": "A_classification",
                "source": "openprimer",
                "label": "amplified",
                "cv": cls[best_cls],
            },
        },
        "data/models/head_a_classifier.joblib",
    )
    joblib.dump(
        {
            "model": rgr,
            "feature_names": FEATURES,
            "metadata": {
                "head": "A_regression",
                "source": "openprimer",
                "label": "primer_efficiency",
                "cv": reg[best_reg],
                "conformal_q90": reg[best_reg]["conformal_q90"],
                "is_proxy_label": False,
            },
        },
        "data/models/head_a_regressor.joblib",
    )
    shap_top = _shap_top(rgr, X, FEATURES) if best_reg != "elasticnet" else ["(linear model)"]
    report = {
        "source": "openprimer",
        "n_rows": int(len(tbl)),
        "n_templates": int(tbl["template_id"].nunique()),
        "positive_rate": float(y_cls.mean()),
        "cv": "grouped by template_id (see model card for the stricter primer-grouped view)",
        "classification": cls,
        "regression": reg,
        "best_classifier": best_cls,
        "best_regressor": best_reg,
        "shap_top_regressor": shap_top,
    }
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    with open("data/reports/head_a.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nbest classifier: {best_cls}  |  best regressor: {best_reg}")
    print(
        f"calibration ({best_cls}): Brier={cls[best_cls]['brier']:.3f} "
        f"ECE={cls[best_cls]['ece']:.3f}  (saved classifier is isotonic-calibrated)"
    )
    print(
        f"regressor 90% prediction interval: +/- {reg[best_reg]['conformal_q90']:.3f} (conformal)"
    )
    print("SHAP top-5 (regressor):", shap_top)
    print("saved: data/models/head_a_{classifier,regressor}.joblib, data/reports/head_a.json")


if __name__ == "__main__":
    main()
