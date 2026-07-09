"""Specificity checking tool — BLAST off-target report (local or remote).

Runs ``blastn -task blastn-short`` for a primer and summarizes the hits with
``primer_core.features.specificity``. Two backends:

* **local** (default) — a formatted local BLAST database (``makeblastdb``).
  Fast, deterministic, offline; the right choice for the interactive agent.
* **remote** — NCBI's BLAST URL API via Biopython (``NCBIWWW.qblast``). No
  local DB or ``blastn`` install needed and it searches NCBI databases (``nt``,
  ``refseq_rna``, ...), but it is **slow (queued, ~30 s-minutes/query) and
  rate-limited** — use it for occasional checks, not for bulk agent scoring.

Either way the parsed hits share one schema (``subject_id``, ``pident``,
``qstart``, ``qend``, ``bitscore``, ...) so the feature functions are identical.
This is one of the four tools shared by the copilot and plain scripts (see
CLAUDE.md); the agent calls it rather than reasoning about off-targets itself.
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
        if len(fields) < 12:  # standard outfmt 6 has 12 columns (qseqid first)
            continue
        rest = fields[1:]
        record: dict[str, Any] = {}
        for name, raw in zip(_OUTFMT6_COLUMNS, rest):
            if name == "subject_id":
                record[name] = raw
            elif name in ("length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send"):
                record[name] = int(raw)
            else:
                record[name] = float(raw)
        hits.append(record)
    return hits


def _parse_blast_record(record: Any) -> list[dict[str, Any]]:
    """Convert a Biopython ``NCBIXML`` record into the shared hit-dict schema."""
    hits: list[dict[str, Any]] = []
    for aln in record.alignments:
        subject_id = getattr(aln, "accession", None) or aln.hit_id
        for hsp in aln.hsps:
            align_len = hsp.align_length or 1
            hits.append(
                {
                    "subject_id": subject_id,
                    "pident": 100.0 * hsp.identities / align_len,
                    "length": int(align_len),
                    "mismatch": int(align_len - hsp.identities),
                    "qstart": int(min(hsp.query_start, hsp.query_end)),
                    "qend": int(max(hsp.query_start, hsp.query_end)),
                    "evalue": float(hsp.expect),
                    "bitscore": float(hsp.bits),
                }
            )
    return hits


def _run_local_blast(primer_seq: str, db_path: str | Path, blastn_bin: str) -> list[dict[str, Any]]:
    """Run local ``blastn -task blastn-short`` and return parsed hits."""
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
                f"check the DB path/formatting before touching code. stderr: {proc.stderr.strip()}"
            )
        return _parse_outfmt6(proc.stdout)
    finally:
        Path(query_path).unlink(missing_ok=True)


def _run_remote_blast(primer_seq: str, database: str) -> list[dict[str, Any]]:
    """Run NCBI remote BLAST via Biopython and return parsed hits (slow!)."""
    try:
        from Bio.Blast import NCBIWWW, NCBIXML
    except ImportError as exc:  # pragma: no cover - biopython is a base dep
        raise RuntimeError("biopython is required for remote BLAST") from exc
    try:
        handle = NCBIWWW.qblast(
            "blastn",
            database,
            primer_seq,
            word_size=7,
            expect=1000.0,
            hitlist_size=50,
            megablast=False,
        )
        record = NCBIXML.read(handle)
    except Exception as exc:  # noqa: BLE001 - network/service errors -> RuntimeError
        raise RuntimeError(f"remote NCBI BLAST failed against {database!r}: {exc}") from exc
    return _parse_blast_record(record)


def check_specificity(
    primer_seq: str,
    db_path: str | Path | None = None,
    exclude_target_id: str | None = None,
    blastn_bin: str = "blastn",
    remote: bool = False,
    database: str = "nt",
) -> dict[str, Any]:
    """Run a BLAST off-target search for a primer and summarize the results.

    Args:
        primer_seq: Primer sequence (5'->3') to query.
        db_path: Local formatted BLAST database (required unless ``remote``).
        exclude_target_id: Subject/accession ID of the intended on-target,
            excluded from off-target scoring.
        blastn_bin: Name/path of the ``blastn`` executable (local mode).
        remote: If True, use NCBI's remote BLAST API instead of a local DB
            (slow and rate-limited; no ``db_path``/``blastn`` needed).
        database: NCBI database name for remote mode (e.g. ``"nt"``,
            ``"refseq_rna"``).

    Returns:
        A report dict with ``off_target_hit_count``, ``best_off_target_bitscore``,
        ``three_prime_offtarget_complementarity``, ``source``, and ``raw_hits``.

    Raises:
        ValueError: If local mode is requested without a ``db_path``.
        RuntimeError: If the BLAST run (local subprocess or remote API) fails.
    """
    if remote:
        raw_hits = _run_remote_blast(primer_seq, database)
        source = f"remote:{database}"
    else:
        if db_path is None:
            raise ValueError("db_path is required for local BLAST (or pass remote=True)")
        raw_hits = _run_local_blast(primer_seq, db_path, blastn_bin)
        source = f"local:{db_path}"

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
        "source": source,
        "raw_hits": raw_hits,
    }
