# primer-design-suite

ML-driven PCR primer / assay design, told in two stages over one shared core:

1. **`predictor/` (flagship)** — trains models (ElasticNet/RandomForest ->
   XGBoost/LightGBM -> a small PyTorch 1D-CNN/k-mer MLP) to predict primer
   efficiency, specificity, and robustness from sequence, thermodynamic, and
   specificity features.
2. **`copilot/` (differentiator)** — an LLM agent that designs primers by
   orchestrating real tools (Primer3, BLAST, thermodynamics) and calls the
   trained flagship model to score candidates, returning a ranked JSON
   output plus a prose design memo.

Both apps consume **`primer_core/`**, the single shared library — the agent
and any standalone script call the exact same feature and tool functions.
Logic is never forked between the two; see `CLAUDE.md` for the full set of
scientific and coding conventions this repo follows (thermodynamics backend
choices, grouped CV, label provenance, the agent's "never fabricate a
sequence or number" rule, etc.) — read it before making changes.

## Layout

```
primer_core/    shared library: features/ (pure, no I/O), io/, tools/
predictor/      flagship ML pipeline: pipeline/, models/, workflows/, notebooks/
copilot/        LLM agent: agent/ (tool loop, schemas, system prompt), app/ (Streamlit)
data/           never committed — see .gitignore and *.provenance.yaml sidecars
tests/          pytest
```

## Commands

```bash
conda activate primer          # environment is conda on WSL

pip install -e .                # editable install, makes primer_core importable
pytest -q                       # run before proposing any change is complete

cd predictor/workflows && snakemake -c4   # featurize -> train -> evaluate
mlflow ui                                 # track experiments

streamlit run copilot/app/main.py         # run the agent UI
```

BLAST specificity features require a local `blastn` binary and a formatted
reference database — check the DB path in
`predictor/workflows/configs/config.yaml` before touching code if a call
fails.
