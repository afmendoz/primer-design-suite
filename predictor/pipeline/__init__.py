"""Pipeline stages: featurize, train, evaluate, predict.

Each module is CLI-able and orchestrated by ``predictor/workflows/Snakefile``.
Feature computation itself always delegates to ``primer_core.features`` —
these modules are the I/O and orchestration layer around that pure core.
"""
