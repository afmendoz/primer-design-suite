"""predictor — the flagship ML pipeline.

Trains models to predict PCR primer efficiency, specificity, and robustness
from sequence-derived, thermodynamic, and specificity features produced by
``primer_core``. See ``predictor/pipeline`` for the featurize -> train ->
evaluate -> predict stages and ``predictor/workflows/Snakefile`` for the DAG.
"""
