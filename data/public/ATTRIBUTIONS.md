# Data attributions

This project trains on public, reusably-licensed datasets. Attribution is
required by their licenses; please preserve this file in redistributions.

## Source A — openPrimeR IGHV feature matrix (committed)
- **Files:** `data/public/openprimer/feature_matrix.csv`
- **License:** **CC BY 4.0** — reuse permitted with attribution.
- **Cite:** Döring A. & Pfeifer N., "Assessment and optimization of the
  interpretability of machine learning models applied to..." *Scientific
  Reports* 9, 2019. `10.1038/s41598-019-47173-w`.
- **Source data:** figshare `10.6084/m9.figshare.6736232` (feature matrix),
  `10.6084/m9.figshare.6736175` (raw PCR results, not committed).

## IGHV template sequences (committed)
- **Files:** `data/public/openprimer/templates/Homo_sapiens_IGH_*.fasta`
- **Provenance:** bundled with the openPrimeR R package
  (`github.com/mattheiter/openPrimeR`), derived from **IMGT®** germline
  reference sequences.
- **Terms:** IMGT® data is free for academic research use with attribution to
  IMGT®, the international ImMunoGeneTics information system
  (Lefranc M.-P. et al.). Cite IMGT® and the openPrimeR package if
  redistributing these sequences. Verify IMGT's current terms before any
  non-academic or commercial use.

## Source B — ETH PCR-bias (fetched, gitignored)
- **Location when fetched:** `data/eth_pcr_bias/` (148 MB; not committed).
- **License:** **BSD 3-Clause** — retain the copyright notice and disclaimer.
  Copyright (c) 2025, Machine Learning and Systems Biology Research Department.
  The upstream `LICENSE.txt` ships inside the fetched repo; keep it if you
  vendor any of their code or data.
- **Cite:** Gimpel A. L. et al., *Nature Communications* 2025,
  `10.1038/s41467-025-64221-4`. Repo `github.com/BorgwardtLab/PCR-bias`,
  archive DOI `10.5281/zenodo.15799030`.

No proprietary or wet-lab data is included in this repository.
