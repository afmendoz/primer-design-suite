"""Specificity checking tool — BLAST off-target report.

Runs ``blastn -task blastn-short`` for a primer against a configured
reference/off-target database and summarizes the result using
``primer_core.features.specificity``. This is one of the four tools shared by
the copilot agent and plain scripts (see CLAUDE.md); the agent must call this
rather than reasoning about off-targets itself.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from primer_core.features import specificity as spec_features

# Standard ``blastn -outfmt 6`` column order.
_OUTFMT6_COLUMNS = (
    "subject_id",  # sseqid
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
)


def _parse_outfmt6(stdout: str) -> list[dict[str, Any]]:
    """Parse ``blastn -outfmt 6`` tabular output into hit dicts."""
    hits: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("\t")
        # Standard outfmt 6 emits 12 columns (qseqid first, which we drop).
        if len(fields) < 12:
            continue
        _qseqid, rest = fields[0], fields[1:]
        record: dict[str, Any] = {}
        for name, raw in zip(_OUTFMT6_COLUMNS, rest):
            if name in ("subject_id",):
                record[name] = raw
            elif name in ("length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send"):
                record[name] = int(raw)
            else:
                record[name] = float(raw)
        hits.append(record)
    return hits


def check_specificity(
    primer_seq: str,
    db_path: str | Path,
    exclude_target_id: str | None = None,
    blastn_bin: str = "blastn",
) -> dict[str, Any]:
    """Run BLAST off-target search for a primer and summarize the results.

    Invokes ``blastn -task blastn-short`` against ``db_path`` and reduces the
    hits to the specificity feature set defined in
    ``primer_core.features.specificity`` (off-target hit count, best
    off-target bit score, 3'-end off-target complementarity).

    Args:
        primer_seq: Primer sequence (5'->3') to query.
        db_path: Path to a formatted local BLAST database (see
            ``makeblastdb``). If the call fails, check this path before
            touching code (per CLAUDE.md).
        exclude_target_id: Subject/accession ID of the intended on-target
            sequence, excluded from off-target scoring.
        blastn_bin: Name/path of the ``blastn`` executable.

    Returns:
        A report dict with keys ``off_target_hit_count``,
        ``best_off_target_bitscore``, ``three_prime_offtarget_complementarity``,
        and ``raw_hits`` (parsed BLAST hit records).

    Raises:
        RuntimeError: If the ``blastn`` subprocess fails or the database is
            not found/formatted.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as query_file:
        query_file.write(f">query\n{primer_seq}\n")
        query_path = query_file.name

    try:
        cmd = [
            blastn_bin,
            "-task",
            "blastn-short",
            "-query",
            query_path,
            "-db",
            str(db_path),
            "-outfmt",
            "6",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"blastn executable not found: {blastn_bin!r}; check the BLAST "
                f"installation and the DB path {db_path!r}"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"blastn failed (exit {proc.returncode}) against DB {db_path!r}; "
                f"check the DB path/formatting before touching code. "
                f"stderr: {proc.stderr.strip()}"
            )

        raw_hits = _parse_outfmt6(proc.stdout)
    finally:
        Path(query_path).unlink(missing_ok=True)

    return {
        "off_target_hit_count": spec_features.off_target_hit_count(raw_hits, exclude_target_id),
        "best_off_target_bitscore": spec_features.best_off_target_bitscore(
            raw_hits, exclude_target_id
        ),
        "three_prime_offtarget_complementarity": (
            spec_features.three_prime_offtarget_complementarity(
                primer_seq, raw_hits, exclude_target_id
            )
        ),
        "raw_hits": raw_hits,
    }
