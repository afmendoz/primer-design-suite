# primer-design-suite

[![CI](https://github.com/afmendoz/primer-design-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/afmendoz/primer-design-suite/actions/workflows/ci.yml)

ML-driven PCR primer / assay design, told in two stages over one shared core:

1. **`predictor/` (flagship)** — trains models on **two real, licensed
   datasets** with a two-head design:
   - **Head A (openPrimeR IGHV, in-domain):** a classification head (will this
     primer amplify this template?) and a regression head (continuous
     efficiency) on designed antibody primers. This is what the copilot scores
     with.
   - **Head B (ETH PCR-bias, generalization):** continuous-efficiency
     regression across seven synthetic-pool sources, used as a **cross-source
     generalization benchmark** — the honest "does it transfer?" analysis.
   The two sources are never pooled; their labels and domains differ (see the
   model cards in `docs/`).
2. **`copilot/` (differentiator)** — a **provider-agnostic** LLM agent (OpenAI /
   Anthropic / any OpenAI-compatible endpoint incl. local Ollama) that designs
   primers by orchestrating real tools (Primer3, ViennaRNA, BLAST) and calls
   the trained head-A models to return, per candidate, **P(amplify) + predicted
   efficiency with in-domain caveats** — as ranked JSON plus a prose memo. It
   never fabricates a sequence or a number.

Both apps consume **`primer_core/`**, the single shared library — the agent and
any standalone script call the exact same feature and tool functions
(`primer_core.featurize.featurize_primer` is the one featurizer). See
`CLAUDE.md` for the full scientific and coding conventions.

## Layout

```
primer_core/    shared library: features/ (composition, thermo, annealing, kmer,
                structure, specificity), io/ (openprimer, ethpcr, ...), tools/,
                featurize.py (the shared featurizer)
predictor/      pipeline/ (head_a, head_b, + legacy simulated fixture), models/, workflows/
copilot/        agent/ (provider adapters, tool loop, schemas, system prompt), app/ (Streamlit)
data/public/    committable licensed data: openprimer/ (CC BY 4.0) + ATTRIBUTIONS.md
data/           local-only / gitignored: eth_pcr_bias/ (fetched, BSD-3), models/, reports/
docs/           model cards
tests/          pytest
```

## Commands

```bash
conda activate primer                 # environment is conda on WSL
pip install -e .                       # editable install
pytest -q                              # run before proposing any change is complete

# Head A trains on committed data. Head B needs the ETH set fetched first:
git clone --depth 1 https://github.com/BorgwardtLab/PCR-bias data/eth_pcr_bias

cd predictor/workflows && snakemake -c4         # builds head_a + head_b reports
python -m predictor.pipeline.head_a             # or run a head directly
python -m predictor.pipeline.head_b

streamlit run copilot/app/main.py               # the agent UI (pick a provider/model)
```

Reports land in `data/reports/head_a.json` and `head_b.json`; trained head-A
artifacts in `data/models/head_a_{classifier,regressor}.joblib`.

## Data & licensing

Datasets and their terms are documented in `data/public/ATTRIBUTIONS.md`
(openPrimeR CC BY 4.0; IMGT template terms; ETH PCR-bias BSD 3-Clause). No proprietary or
wet-lab data is included. BLAST specificity features require a local `blastn`
binary and a formatted DB (optional; the pipeline skips them cleanly if absent).
