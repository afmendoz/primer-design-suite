"""Head B (ETH PCR-bias): efficiency regression + cross-source generalization.

Source B is the generalization benchmark, not the copilot's scoring domain: its
sequences are 90-160 nt synthetic-pool amplicons (Primer3 duplex thermodynamics
are undefined there), so features are length-agnostic composition — GC content +
3-mer frequencies. Two evaluations, mirroring the ETH scripts:

* internal  -> pooled 5-fold regression (Spearman / RMSE), the modeling ladder;
* external  -> leave-one-source-out: train on six sources, test on the held-out
  one. Cross-source transfer is the interesting, honest result — domain shift
  between the seven sources is the thing to analyze, not hide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from predictor.models import LADDER_MODELS, get_model
from primer_core.features import composition as comp
from primer_core.features.kmer import all_kmers, kmer_frequencies
from primer_core.io.ethpcr import SOURCES, load_ethpcr

DATA_DIR = "data/eth_pcr_bias/Data"
FEATURES = ["gc_content"] + all_kmers(3)
_REG_PARAMS = {
    "elasticnet": {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000},
    "random_forest": {"n_estimators": 200, "n_jobs": -1, "random_state": 0},
    "xgboost": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    "lightgbm": {"n_estimators": 300, "verbose": -1},
}


def build_feature_table(data_dir: str = DATA_DIR, per_source_cap: int = 3000) -> pd.DataFrame:
    df = load_ethpcr(data_dir, per_source_cap=per_source_cap)
    df = df[df["sequence"].str.fullmatch(r"[ACGT]+")].reset_index(drop=True)
    feats = [{"gc_content": comp.gc_content(s), **kmer_frequencies(s, 3)} for s in df["sequence"]]
    fx = pd.DataFrame(feats, columns=FEATURES)
    return pd.concat([df, fx], axis=1)


def _metrics(y, pred):
    return {
        "spearman": float(spearmanr(y, pred).correlation),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def run_internal(X, y, n_splits=5) -> dict[str, dict[str, float]]:
    out = {}
    for name in LADDER_MODELS:
        oof = np.zeros(len(y))
        for tr, te in KFold(n_splits, shuffle=True, random_state=0).split(X):
            est = get_model(name, _REG_PARAMS[name])
            est.fit(X[tr], y[tr])
            oof[te] = est.predict(X[te])
        out[name] = _metrics(y, oof)
    return out


def run_generalization(tbl, model_name="xgboost") -> dict[str, dict[str, float]]:
    """Leave-one-source-out cross-source transfer."""
    X = tbl[FEATURES].to_numpy(float)
    y = tbl["eff"].to_numpy(float)
    src = tbl["source"].to_numpy()
    out = {}
    for held in SOURCES:
        te = src == held
        tr = ~te
        est = get_model(model_name, _REG_PARAMS[model_name])
        est.fit(X[tr], y[tr])
        out[held] = _metrics(y[te], est.predict(X[te]))
    return out


def run_transfer_matrix(
    tbl, model_name="xgboost", test_frac=0.3, seed=0
) -> dict[str, dict[str, float]]:
    """Full train-on-source-X / test-on-source-Y Spearman matrix.

    A single fixed per-source hold-out makes every cell comparable and leakage-
    free: the diagonal (train X_train -> test X_test) is honest *within*-source
    generalization, the off-diagonal is cross-source transfer. The contrast
    between the diagonal and its row is the domain-shift signal, made pairwise.
    """
    rng = np.random.default_rng(seed)
    X = tbl[FEATURES].to_numpy(float)
    y = tbl["eff"].to_numpy(float)
    src = tbl["source"].to_numpy()
    train_idx, test_idx = {}, {}
    for s in SOURCES:
        idx = rng.permutation(np.flatnonzero(src == s))
        cut = int(len(idx) * (1.0 - test_frac))
        train_idx[s], test_idx[s] = idx[:cut], idx[cut:]

    matrix = {}
    for s_tr in SOURCES:
        est = get_model(model_name, _REG_PARAMS[model_name])
        est.fit(X[train_idx[s_tr]], y[train_idx[s_tr]])
        matrix[s_tr] = {
            s_te: float(spearmanr(y[test_idx[s_te]], est.predict(X[test_idx[s_te]])).correlation)
            for s_te in SOURCES
        }
    return matrix


def main() -> None:
    tbl = build_feature_table()
    X = tbl[FEATURES].to_numpy(float)
    y = tbl["eff"].to_numpy(float)
    print(
        f"rows={len(tbl)}  sources={tbl['source'].nunique()}  features={len(FEATURES)}  "
        f"eff[{y.min():.3f},{y.max():.3f}]\n"
    )

    print("== Internal (pooled 5-fold) regression ladder ==")
    internal = run_internal(X, y)
    print(f"{'model':<15}{'Spearman':>10}{'RMSE':>9}")
    for m, d in internal.items():
        print(f"{m:<15}{d['spearman']:>10.3f}{d['rmse']:>9.4f}")

    print("\n== Cross-source generalization (leave-one-source-out, xgboost) ==")
    gen = run_generalization(tbl)
    print(f"{'held-out source':<16}{'Spearman':>10}{'RMSE':>9}")
    for s, d in gen.items():
        print(f"{s:<16}{d['spearman']:>10.3f}{d['rmse']:>9.4f}")
    mean_sp = float(np.mean([d["spearman"] for d in gen.values()]))
    best_int = max(internal, key=lambda m: internal[m]["spearman"])

    print("\n== Transfer matrix (train source -> test source, Spearman, xgboost) ==")
    matrix = run_transfer_matrix(tbl)
    hdr = "".join(f"{s[:5]:>7}" for s in SOURCES)
    corner = "train\\test"
    print(f"{corner:<12}{hdr}")
    for s_tr in SOURCES:
        row = "".join(f"{matrix[s_tr][s_te]:>7.2f}" for s_te in SOURCES)
        print(f"{s_tr:<12}{row}")
    diag = float(np.mean([matrix[s][s] for s in SOURCES]))
    offdiag = float(np.mean([matrix[a][b] for a in SOURCES for b in SOURCES if a != b]))
    print(
        f"mean diagonal (within-source) = {diag:.3f}  |  mean off-diagonal (cross) = {offdiag:.3f}"
    )

    report = {
        "source": "eth_pcr_bias",
        "n_rows": int(len(tbl)),
        "n_sources": int(tbl["source"].nunique()),
        "features": "gc_content + 3-mer frequencies (primer thermo undefined >60bp)",
        "internal_5fold": internal,
        "cross_source_loso": gen,
        "mean_cross_source_spearman": mean_sp,
        "best_internal_model": best_int,
        "transfer_matrix": matrix,
        "transfer_diag_mean": diag,
        "transfer_offdiag_mean": offdiag,
        "note": "internal vs cross-source gap is the domain-shift signal",
    }
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    with open("data/reports/head_b.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(
        f"\nmean cross-source Spearman = {mean_sp:.3f}  (vs best internal "
        f"{best_int} = {internal[best_int]['spearman']:.3f})"
    )
    print("-> the gap between internal and cross-source is the domain-shift signal.")
    print("saved: data/reports/head_b.json")


if __name__ == "__main__":
    main()
