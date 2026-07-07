"""Featurization stage: dataset -> feature table.

Loads a labeled dataset via ``primer_core.io.datasets``, computes composition,
thermodynamic, structure, and specificity features via ``primer_core.features``
for every record, and writes a feature table for ``train.py`` to consume.

CLI-able: intended to be invoked directly or via
``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def featurize_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    config_path: str | Path = "predictor/workflows/configs/config.yaml",
) -> Path:
    """Compute the full feature table for a dataset and write it to disk.

    Args:
        dataset_path: Path to a raw labeled dataset (see
            ``primer_core.io.datasets``).
        output_path: Destination path for the computed feature table.
        config_path: Path to the pipeline config controlling feature toggles.

    Returns:
        Path to the written feature table.

    Raises:
        FileNotFoundError: If ``dataset_path`` or ``config_path`` is missing.
    """
    raise NotImplementedError


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module."""
    parser = argparse.ArgumentParser(description="Featurize a primer/assay dataset.")
    parser.add_argument("--dataset", required=True, help="Path to raw dataset")
    parser.add_argument("--output", required=True, help="Path to write feature table")
    parser.add_argument(
        "--config",
        default="predictor/workflows/configs/config.yaml",
        help="Path to pipeline config YAML",
    )
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m predictor.pipeline.featurize``."""
    args = _build_arg_parser().parse_args()
    featurize_dataset(args.dataset, args.output, args.config)


if __name__ == "__main__":
    main()
