"""Dataset loaders for the predictor pipeline.

Reads labeled primer/assay datasets from ``data/`` (never committed — see
CLAUDE.md's data rules) alongside their ``*.provenance.yaml`` sidecars, and
returns validated, tabular structures ready for featurization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from primer_core.io.schema import ProvenanceRecord

_DATASET_SUFFIXES = (".csv", ".tsv", ".parquet")


def load_dataset(path: str | Path) -> Any:
    """Load a labeled primer/assay dataset from disk.

    Args:
        path: Path to a dataset file (``.csv``/``.tsv``/``.parquet``) under
            ``data/``.

    Returns:
        A ``pandas.DataFrame`` of records.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the dataset's provenance sidecar is missing or invalid,
            or the file suffix is unsupported.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    try:
        load_provenance(path)
    except Exception as exc:  # noqa: BLE001 - re-raised uniformly as ValueError
        raise ValueError(f"missing/invalid provenance sidecar for dataset {path}: {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported dataset suffix: {suffix!r}")


def load_provenance(dataset_path: str | Path) -> ProvenanceRecord:
    """Load and validate the ``*.provenance.yaml`` sidecar for a dataset.

    Args:
        dataset_path: Path to the dataset file whose sidecar should be loaded
            (sidecar is expected alongside it, same stem).

    Returns:
        A validated ``primer_core.io.schema.ProvenanceRecord``.

    Raises:
        FileNotFoundError: If the sidecar file is missing.
    """
    import yaml

    sidecar = Path(dataset_path).with_suffix(".provenance.yaml")
    if not sidecar.exists():
        raise FileNotFoundError(f"provenance sidecar not found: {sidecar}")
    with sidecar.open() as handle:
        data = yaml.safe_load(handle)
    return ProvenanceRecord(**data)


def list_datasets(data_dir: str | Path = "data") -> list[Path]:
    """List dataset files available under ``data_dir``.

    Args:
        data_dir: Root directory to search (defaults to the repo's ``data/``).

    Returns:
        Sorted paths to discovered dataset files (provenance sidecars
        excluded).
    """
    root = Path(data_dir)
    datasets: list[Path] = []
    for candidate in root.glob("*"):
        if candidate.name.endswith(".provenance.yaml"):
            continue
        if candidate.suffix.lower() in _DATASET_SUFFIXES:
            datasets.append(candidate)
    return sorted(datasets)
