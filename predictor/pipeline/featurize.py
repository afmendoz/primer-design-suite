"""Featurization stage: dataset -> feature table.

Loads a labeled dataset via ``primer_core.io.datasets.load_dataset`` (which
validates the provenance sidecar), computes per-primer features from
``primer_core.features`` gated by the config ``features:`` toggles, and writes
a flat CSV feature table for ``train.py`` / ``evaluate.py`` to consume.

Feature computation always delegates to the shared ``primer_core`` functions —
this module is just the I/O + orchestration layer around that pure core.

CLI-able: invoked directly or via ``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from primer_core.features import composition, structure, thermo

logger = logging.getLogger(__name__)

_COMPOSITION_COLS = ["gc_content", "gc_clamp", "length"]
_THERMO_COLS = ["tm", "hairpin_dg", "homodimer_dg", "three_prime_end_dg"]
_STRUCTURE_COLS = ["binding_site_accessibility", "amplicon_secondary_structure_dg"]
_SPECIFICITY_COLS = [
    "off_target_hit_count",
    "best_off_target_bitscore",
    "three_prime_offtarget_complementarity",
]


def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    with open(config_path) as handle:
        return yaml.safe_load(handle)


def featurize_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    config_path: str | Path = "predictor/workflows/configs/config.yaml",
) -> Path:
    """Compute the feature table for a dataset and write it to disk as CSV.

    Args:
        dataset_path: Path to a raw labeled dataset (validated via
            ``primer_core.io.datasets.load_dataset``).
        output_path: Destination CSV path for the computed feature table.
        config_path: Path to the pipeline config controlling feature toggles.

    Returns:
        Path to the written feature table.

    Raises:
        FileNotFoundError: If ``dataset_path`` or ``config_path`` is missing.
    """
    import pandas as pd

    from primer_core.io.datasets import load_dataset

    config = _load_config(config_path)
    toggles = config.get("features", {})
    thermo_settings = config.get("thermo", {}).get("primer3_settings", {})

    df = load_dataset(dataset_path)

    do_structure = bool(toggles.get("structure", False))
    have_structure_cols = "template_seq" in df.columns and "site_start" in df.columns
    if do_structure and not have_structure_cols:
        logger.warning("structure enabled but template_seq/site_start columns absent; skipping")
        do_structure = False

    do_specificity = bool(toggles.get("specificity", False))
    blast_db = config.get("paths", {}).get("blast_db", "")
    if do_specificity and not (blast_db and Path(blast_db).exists()):
        logger.warning("skipping specificity: no BLAST DB at %s", blast_db)
        do_specificity = False

    feature_cols: list[str] = []
    if toggles.get("composition", True):
        feature_cols += _COMPOSITION_COLS
    if toggles.get("thermo", True):
        feature_cols += _THERMO_COLS
    if do_structure:
        feature_cols += _STRUCTURE_COLS
    if do_specificity:
        feature_cols += _SPECIFICITY_COLS

    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        seq = row.sequence
        rec: dict[str, Any] = {
            "primer_id": row.primer_id,
            "template_id": row.template_id,
            "label": row.label,
        }
        if toggles.get("composition", True):
            rec["gc_content"] = composition.gc_content(seq)
            rec["gc_clamp"] = composition.gc_clamp(seq, 5)
            rec["length"] = composition.length(seq)
        if toggles.get("thermo", True):
            rec["tm"] = thermo.calc_tm(seq, **thermo_settings)
            rec["hairpin_dg"] = thermo.calc_hairpin(seq, **thermo_settings)
            rec["homodimer_dg"] = thermo.calc_homodimer(seq, **thermo_settings)
            rec["three_prime_end_dg"] = thermo.three_prime_end_dg(seq, **thermo_settings)
        if do_structure:
            template_seq = row.template_seq
            site_start = int(row.site_start)
            site_end = site_start + len(seq)
            rec["binding_site_accessibility"] = structure.binding_site_accessibility(
                template_seq, site_start, site_end
            )
            rec["amplicon_secondary_structure_dg"] = structure.amplicon_secondary_structure_dg(
                template_seq
            )
        if do_specificity:
            report = _specificity_features(seq, blast_db)
            rec.update(report)
        records.append(rec)

    out = pd.DataFrame(records, columns=["primer_id", "template_id", "label", *feature_cols])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("wrote %d rows x %d features to %s", len(out), len(feature_cols), output_path)
    return output_path


def _specificity_features(seq: str, blast_db: str) -> dict[str, Any]:
    """Compute BLAST specificity features for one primer (DB confirmed present)."""
    from primer_core.tools.specificity import check_specificity

    report = check_specificity(seq, db_path=blast_db)
    return {col: report[col] for col in _SPECIFICITY_COLS}


def _build_arg_parser() -> argparse.ArgumentParser:
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
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    featurize_dataset(args.dataset, args.output, args.config)


if __name__ == "__main__":
    main()
