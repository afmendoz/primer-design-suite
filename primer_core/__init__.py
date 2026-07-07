"""primer_core — shared library and single source of truth.

Both the predictor pipeline and the copilot agent import from here so they call
the same underlying feature and tool functions. Do not fork logic downstream.
"""
