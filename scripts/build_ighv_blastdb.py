"""Build a small local BLAST DB from the IGHV templates (specificity demo).

Requires ``makeblastdb`` (BLAST+). Subject IDs are the allele names with ``*``
and ``|`` sanitized to ``_`` (BLAST-safe), so ``exclude_target_id`` can be e.g.
``"IGHV1-2_02"``. Off-target here means *other IGHV alleles* — which are highly
similar paralogs, so a primer legitimately hits many. That is exactly the point
of the demo: what counts as "off-target" depends entirely on the database.

    python scripts/build_ighv_blastdb.py   # -> data/blast/ighv_db (gitignored)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from Bio import SeqIO

TEMPLATES = "data/public/openprimer/templates/Homo_sapiens_IGH_functional_exon.fasta"
OUT_DIR = Path("data/blast")
DB = OUT_DIR / "ighv_db"


def sanitize(allele: str) -> str:
    return allele.replace("*", "_").replace("|", "_")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fasta = OUT_DIR / "ighv_alleles.fasta"
    n = 0
    with open(fasta, "w") as fh:
        for rec in SeqIO.parse(TEMPLATES, "fasta"):
            allele = rec.description.split("|")[1]
            fh.write(f">{sanitize(allele)}\n{str(rec.seq).upper()}\n")
            n += 1
    subprocess.run(
        ["makeblastdb", "-in", str(fasta), "-dbtype", "nucl", "-out", str(DB)], check=True
    )
    print(f"built BLAST db {DB} from {n} IGHV alleles")


if __name__ == "__main__":
    main()
