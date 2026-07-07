"""Sequence loaders (FASTA and related formats).

Thin I/O wrappers backed by Biopython's ``SeqIO`` for parsing. Kept separate
from ``primer_core/features`` so feature functions never touch the filesystem
directly.
"""

from __future__ import annotations

from pathlib import Path


def load_fasta(path: str | Path) -> dict[str, str]:
    """Load sequences from a FASTA file.

    Args:
        path: Path to a FASTA file.

    Returns:
        Mapping of record ID to uppercase sequence string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    from Bio import SeqIO

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    return {record.id: str(record.seq).upper() for record in SeqIO.parse(str(path), "fasta")}


def write_fasta(records: dict[str, str], path: str | Path) -> None:
    """Write sequences to a FASTA file.

    Args:
        records: Mapping of record ID to sequence string.
        path: Destination path.
    """
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    seq_records = [
        SeqRecord(Seq(seq), id=rec_id, description="") for rec_id, seq in records.items()
    ]
    SeqIO.write(seq_records, str(path), "fasta")


def load_single_sequence(path: str | Path) -> str:
    """Load a single sequence from a FASTA file containing exactly one record.

    Args:
        path: Path to a single-record FASTA file.

    Returns:
        The sequence string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains zero or more than one record.
    """
    records = load_fasta(path)
    if len(records) != 1:
        raise ValueError(f"expected exactly one record in {path}, found {len(records)}")
    return next(iter(records.values()))
