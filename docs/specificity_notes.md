# Specificity (BLAST) — how to read off-target counts

Off-target features (`off_target_hit_count`, `best_off_target_bitscore`,
`three_prime_offtarget_complementarity`) come from a real BLAST search
(`blastn -task blastn-short`), local or via NCBI's remote API. The number is
only as trustworthy as the **database is representative** — read these caveats
before trusting a value.

## The database defines what "off-target" means
BLAST can only report alignments to sequences **that are in the database**. So
the DB *is* the entire universe of possible off-targets:

- **Against a narrow DB, a low/zero count is not "specific" — it's uninformative.**
  The bundled demo DB (`data/blast/ighv_db`) contains only the 155 IGHV alleles.
  A primer for anything else (a non-IGHV gene, a random primer) finds no
  alignments there and returns `off_target_hit_count = 0` — but that 0 just
  means "doesn't hit IGHV," which is trivially true and says nothing about its
  real specificity. **Be most suspicious of a 0 from a small DB.**
- **The same primer scored against different DBs gives different counts.** For a
  genuine, genome-scale answer you need a DB that represents the real off-target
  space (human RefSeq mRNA / genome locally, or NCBI `nt` / `refseq_rna` via the
  remote API).

## The IGHV demo (why the count is *high*, on purpose)
A leader/FR1 primer designed for `IGHV1-2*02`, checked against the IGHV DB with
`exclude_target_id="IGHV1-2_02"`, returns **~209 off-target hits** at 100%
identity — including *other alleles of the same gene* (`IGHV1-2_01/03/04`). That
is correct: IGHV genes are highly similar paralogs, and this primer sits in a
conserved region. So the same primer is **fine for a repertoire/family assay and
useless for a gene-specific one**. Off-target is defined by the DB **and the
assay intent**.

## The raw count is unfiltered
`check_specificity` reports every HSP BLAST returns, with no dedup or
e-value/identity/coverage filtering beyond `-task blastn-short`. Use it alongside
`best_off_target_bitscore` and `three_prime_offtarget_complementarity` (which
weight by alignment quality and 3'-end involvement), not as a standalone number.

## Backends
- **Local** (default): `python scripts/build_ighv_blastdb.py` builds the demo DB;
  point `blast_db` at any `makeblastdb` database. Fast, deterministic, offline.
- **Remote** (`remote=True`): NCBI `NCBIWWW.qblast` — no install, genome-scale
  DBs, but slow (queued, ~30 s-minutes/query) and rate-limited. For occasional
  checks, not bulk agent scoring.
