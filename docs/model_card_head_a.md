# Model card — Head A (openPrimeR IGHV, in-domain)

## Overview
Two heads sharing the `primer_core` featurizer, trained on designed IGHV
antibody primers on real V-gene templates:
- **Classification** — will this primer amplify this template? (`amplified` :=
  `primer_efficiency >= 0.5`).
- **Regression** — continuous `primer_efficiency` in [0, 1].

This is the **in-domain** head the copilot uses to score designed primers.

## Data
- **Source:** openPrimeR feature matrix (Döring & Pfeifer, *Sci Rep* 2019,
  `10.1038/s41598-019-47173-w`; figshare `10.6084/m9.figshare.6736232`).
- **License:** CC BY 4.0 (see `data/public/ATTRIBUTIONS.md`).
- **Rows:** 829 → **783 after excluding 46 IUPAC-degenerate primers** (by
  decision: no degenerate handling).
- **Templates:** 44 of 47 covered by the fetched IMGT FASTA (`IGHV3-49*05`,
  `IGHV5-10-1*03`, `IGHV5-51*03` absent).
- **Label balance:** ~12% positive; `primer_efficiency` is strongly bimodal
  (~75% ≈ 0, a cluster near 1).

## Features (shared `primer_core` featurizer)
Composition (`gc_content`, `gc_clamp`, `length`), intrinsic thermo
(`tm`, `hairpin_dg`, `homodimer_dg`, `three_prime_end_dg`), and **primer-template
annealing** (`annealing_dg`, `mismatch_count`, `three_prime_mismatch`). The
featurizer was validated against openPrimeR's precomputed columns (Spearman:
GC 1.00, length 1.00, Tm 0.86, self-dimer 0.79; annealing ΔG **0.81** vs their
`annealing_DeltaG`, where the intrinsic 3′-end feature was ~0).

## Metrics (grouped CV)
| view | classification (LightGBM) | regression (RandomForest) |
|---|---|---|
| grouped by **template** (unseen templates) | PR-AUC 0.99 · ROC-AUC 1.00 · MCC 0.95 | Spearman 0.82 · RMSE 0.07 |
| grouped by **primer** (unseen primers) | PR-AUC 0.83 · ROC-AUC 0.98 · MCC 0.77 | Spearman 0.71 · RMSE 0.15 |

**SHAP top features:** `mismatch_count`, `annealing_dg`, `homodimer_dg`,
`three_prime_end_dg`, `hairpin_dg` — biologically sensible (passes the CLAUDE.md
sanity check).

## Known limitations (read before trusting a number)
- **Pre-filtering ceiling.** openPrimeR primers were pre-filtered for a GC clamp
  and no self-dimers, so the model never sees those failure modes and will
  **overestimate** amplification for primers with unfavorable properties or
  structured templates.
- **The template-grouped 0.99 is optimistic.** Primers recur across templates,
  so template-grouping leaks primer identity. The **primer-grouped** numbers
  (PR-AUC 0.83, Spearman 0.71) are the honest estimate for the copilot's use
  case (scoring *new* primers), and are what should be quoted.
- **Possible label circularity.** openPrimeR's `primer_efficiency` may be partly
  thermodynamically derived; predicting it from our annealing features could be
  partly circular. Flagged for follow-up (compare against the raw gel-band
  majority vote in `Single_Eval_*.csv`).
- **Ungapped alignment.** `annealing` uses ungapped min-mismatch alignment; its
  mismatch count matches openPrimeR exactly ~57% of the time (they allow gaps).

## Intended use
Score designed IGHV-like primers in-domain. **Out of domain** for non-antibody
primers, degenerate primers, or unusual templates — the copilot surfaces this
caveat on every score.
