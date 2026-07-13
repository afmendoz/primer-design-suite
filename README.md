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
(`primer_core.featurize.featurize_primer` is the one featurizer). The project
follows a consistent set of scientific and coding conventions throughout.

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

> Tested on Linux (WSL) and CI (Ubuntu/Python 3.11). macOS/Windows are
> unverified — the scientific wheels (ViennaRNA, primer3-py, LightGBM) may need
> conda rather than pip there.

```bash
# 1. Code + environment (Python 3.11)
git clone https://github.com/afmendoz/primer-design-suite
cd primer-design-suite
conda create -n primer python=3.11 -y && conda activate primer   # or python -m venv .venv

# 2. Install (base + the app/agent deps)
pip install -e ".[copilot]"            # base + streamlit + openai + anthropic
#   LightGBM needs the OpenMP runtime; if `import lightgbm` fails:
#     conda install -c conda-forge libgomp          # (usually already present on Ubuntu)
#   BLAST+ (optional, only for the Specificity tab):
#     conda install -c bioconda blast               # (or apt install ncbi-blast+)

# 3. Run the app
streamlit run copilot/app/main.py      # -> http://localhost:8501
```

**First launch builds the head-A models automatically** (~20 s) from the
committed openPrimeR data — the trained artifacts aren't committed (they're
reproducible). To build them ahead of time instead:
`python -m predictor.pipeline.head_a`.

**What needs what:**
- **Score manual primers** — works immediately, no key. (Needs a template for the
  annealing features.)
- **Specificity** — additionally needs BLAST+ and a local DB
  (`python scripts/build_ighv_blastdb.py`).
- **Design (agent)** — needs a provider key: paste it into the sidebar **API key**
  field, or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` before launching.

**Head B** (the cross-source generalization benchmark) is optional and separate:

```bash
git clone --depth 1 https://github.com/BorgwardtLab/PCR-bias data/eth_pcr_bias
python -m predictor.pipeline.head_b
# or run the whole DAG:  cd predictor/workflows && snakemake -c4
pip install -e ".[dev]" && pytest -q   # dev tooling + the test suite
```

Reports land in `data/reports/head_a.json` / `head_b.json`; trained head-A
artifacts in `data/models/head_a_{classifier,regressor}.joblib`.

## Data & licensing

Datasets and their terms are documented in `data/public/ATTRIBUTIONS.md`
(openPrimeR CC BY 4.0; IMGT template terms; ETH PCR-bias BSD 3-Clause). No proprietary or
wet-lab data is included. BLAST specificity features require a local `blastn`
binary and a formatted DB (optional; the pipeline skips them cleanly if absent).
