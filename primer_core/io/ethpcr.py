"""Loader for the ETH PCR-bias dataset (Source B).

Seven synthetic-pool sources, each shipping ``bad_seqs_<threshold>.pkl`` as a
dict ``{"rest": df, "bottom": df}`` with ``sequence`` and ``eff`` (continuous
efficiency) columns. The threshold only splits bottom/rest, so the continuous
labels are threshold-independent; we load one threshold and keep every sequence
with its efficiency, tagged by source. Optionally balanced-subsampled per source
(the sources range from ~4.8k to ~210k rows) so cross-source analysis is not
dominated by the largest set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ["GCall", "GCfix", "Choi_et_al", "Erlich_et_al", "Gao_et_al", "Koch_et_al", "Song_et_al"]


def load_ethpcr(
    data_dir: str | Path,
    threshold: str = "2perc",
    per_source_cap: int | None = 3000,
    seed: int = 0,
) -> pd.DataFrame:
    """Load the seven ETH sources into one tidy table.

    Args:
        data_dir: Path to the repo's ``Data/`` directory.
        threshold: Which ``bad_seqs_<threshold>.pkl`` to read (labels are the
            same across thresholds).
        per_source_cap: If set, randomly subsample each source to this many rows.
        seed: RNG seed for subsampling.

    Returns:
        DataFrame with columns: sequence, eff (continuous efficiency), source.
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)
    frames = []
    for src in SOURCES:
        d = pd.read_pickle(data_dir / src / f"bad_seqs_{threshold}.pkl")
        seq = np.hstack([d["bottom"]["sequence"].values, d["rest"]["sequence"].values])
        eff = np.hstack([d["bottom"]["eff"].values, d["rest"]["eff"].values])
        f = pd.DataFrame({"sequence": seq, "eff": eff.astype(float), "source": src})
        if per_source_cap is not None and len(f) > per_source_cap:
            f = f.iloc[rng.choice(len(f), per_source_cap, replace=False)]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
