"""Loader for the openPrimeR IGHV dataset (Source A).

Reads the precomputed feature matrix + the IMGT template FASTA, joins each
primer to its template sequence, and derives the two head-A labels: a
continuous in-domain efficiency (``efficiency``) and a binary amplification
label (``amplified`` = efficiency >= threshold). Degenerate (IUPAC) primers and
primers whose template is absent from the FASTA are dropped by decision (see
the dataset provenance sidecar).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from Bio import SeqIO

_OUT_COLS = [
    "primer_id",
    "primer_seq",
    "template_id",
    "group",
    "template_seq",
    "efficiency",
    "amplified",
]


def load_templates(fasta_path: str | Path) -> dict[str, str]:
    """Map IMGT allele name (2nd '|' field of the header) -> uppercase sequence."""
    return {
        rec.description.split("|")[1]: str(rec.seq).upper()
        for rec in SeqIO.parse(str(fasta_path), "fasta")
    }


def load_openprimer(
    matrix_path: str | Path,
    templates_path: str | Path,
    efficiency_threshold: float = 0.5,
) -> pd.DataFrame:
    """Load and clean the openPrimeR set into a tidy per-(primer, template) table.

    Args:
        matrix_path: Path to ``feature_matrix.csv``.
        templates_path: Path to the IGH template FASTA.
        efficiency_threshold: ``amplified`` is 1 when ``efficiency`` >= this.

    Returns:
        DataFrame with columns: primer_id, primer_seq, template_id, group,
        template_seq, efficiency (continuous), amplified (0/1). Degenerate
        primers and uncovered templates are excluded.
    """
    df = pd.read_csv(matrix_path)
    templates = load_templates(templates_path)
    df["primer_seq"] = df["Primer_Sequence"].astype(str).str.upper()
    df = df[df["primer_seq"].str.fullmatch(r"[ACGT]+")]  # drop degenerate
    df = df[df["Template"].isin(templates)].copy()  # drop uncovered templates
    df["template_seq"] = df["Template"].map(templates)
    df["efficiency"] = pd.to_numeric(df["primer_efficiency"], errors="coerce")
    df = df[df["efficiency"].notna()]
    df["amplified"] = (df["efficiency"] >= efficiency_threshold).astype(int)
    df = df.rename(columns={"Primer": "primer_id", "Template": "template_id", "Group": "group"})
    return df[_OUT_COLS].reset_index(drop=True)
