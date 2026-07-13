"""Render the head-A result figures used in the README and model card.

Two honest figures, both from real artifacts (no hand-drawn numbers):

  1. ``calibration_head_a.png`` — a reliability diagram from the *grouped-CV
     out-of-fold* probabilities of the best classifier (the same OOF the report
     computes), annotated with Brier / ECE. This is the visual of the
     "predicted P(amplify) is trustworthy as a probability" claim.
  2. ``feature_importance_head_a.png`` — impurity importance of the deployed
     RandomForest efficiency head, i.e. which features actually drive the score.

Run inside the ``primer`` env after the models are built::

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from predictor.models import CLASSIFIER_LADDER, get_classifier  # noqa: E402
from predictor.pipeline.head_a import FEATURES, _ece, _oof, build_feature_table  # noqa: E402

OUT = Path("docs/figures")
INK = "#1f2a44"
ACCENT = "#c0392b"
GRID = "#d9dee8"


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.7)


def calibration_figure(X, y, groups):
    """Reliability diagram from grouped-CV OOF probabilities of the best model."""
    # pick the lowest-Brier (best-calibrated) model on grouped-CV OOF
    scored = {
        n: _oof(lambda m=n: get_classifier(m), X, y, groups, 5, True) for n in CLASSIFIER_LADDER
    }
    best = min(scored, key=lambda n: brier_score_loss(y, scored[n]))
    proba = scored[best]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
    brier, ece = brier_score_loss(y, proba), _ece(y, proba)

    fig, (ax, axh) = plt.subplots(
        2,
        1,
        figsize=(5.2, 5.4),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    _style(ax)
    ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1.4, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=ACCENT, lw=2, ms=7, label=f"{best} (grouped-CV OOF)")
    ax.set_ylabel("observed amplification rate", color=INK)
    ax.set_title("Head A — classifier is well-calibrated", color=INK, fontsize=12, weight="bold")
    ax.text(
        0.03,
        0.92,
        f"Brier = {brier:.3f}\nECE = {ece:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color=INK,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f3f5f9", ec=GRID),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_ylim(-0.02, 1.02)

    # the predictions are bimodal (zero-inflated) -> few calibration points is
    # expected; the histogram makes the mass-at-the-extremes explicit.
    _style(axh)
    axh.grid(False, axis="x")
    axh.hist(proba, bins=np.linspace(0, 1, 21), color=INK, alpha=0.85)
    axh.set_yscale("log")
    axh.set_ylabel("count\n(log)", color=INK, fontsize=8)
    axh.set_xlabel("predicted P(amplify)", color=INK)
    axh.set_xlim(-0.02, 1.02)
    fig.savefig(OUT / "calibration_head_a.png", bbox_inches="tight")
    plt.close(fig)
    return best, brier, ece


def importance_figure():
    """Impurity importance of the deployed RandomForest efficiency head."""
    art = joblib.load("data/models/head_a_regressor.joblib")
    model, names = art["model"], art["feature_names"]
    imp = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(imp)
    names = [names[i] for i in order]
    imp = imp[order]
    colors = [
        ACCENT if names[i] in ("mismatch_count", "annealing_dg") else INK for i in range(len(names))
    ]

    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.grid(False, axis="y")
    ax.barh(names, imp, color=colors, height=0.68)
    ax.set_xlabel("impurity importance", color=INK)
    ax.set_title(
        "Head A — efficiency head leans on the biology",
        color=INK,
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance_head_a.png", bbox_inches="tight")
    plt.close(fig)
    return names[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tbl = build_feature_table()
    X = tbl[FEATURES].to_numpy(dtype=float)
    groups = tbl["template_id"].to_numpy()
    y_cls = tbl["amplified"].to_numpy(dtype=int)
    best, brier, ece = calibration_figure(X, y_cls, groups)
    top = importance_figure()
    print(f"calibration: {best}  Brier={brier:.3f} ECE={ece:.3f}")
    print(f"top efficiency feature: {top}")
    print(f"wrote {OUT}/calibration_head_a.png, {OUT}/feature_importance_head_a.png")


if __name__ == "__main__":
    main()
