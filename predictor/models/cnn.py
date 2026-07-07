"""Small deep model: 1D-CNN / k-mer MLP over raw sequence (PyTorch).

Top of the modeling ladder in CLAUDE.md. ``torch`` is an optional dependency
(see ``pyproject.toml``'s ``predictor`` extra), so the import is deferred
into the functions below rather than performed at module scope — this module
must remain importable even when torch is not installed.
"""

from __future__ import annotations

from typing import Any


def build_sequence_cnn(seq_length: int, **params: object) -> Any:
    """Construct a small 1D-CNN over one-hot-encoded primer sequence.

    Requires ``torch``, imported lazily inside this function so that this
    module remains importable without the optional ``predictor`` extra
    installed.

    Args:
        seq_length: Fixed input sequence length (one-hot encoded, 4 channels).
        **params: Hyperparameters (e.g. number of conv layers/filters,
            dropout), typically sourced from
            ``predictor/workflows/configs/config.yaml``.

    Returns:
        An untrained ``torch.nn.Module``.

    Raises:
        ImportError: If ``torch`` is not installed.
    """
    raise NotImplementedError


def build_kmer_mlp(kmer_vocab_size: int, k: int = 4, **params: object) -> Any:
    """Construct a small MLP over k-mer frequency features of the sequence.

    Requires ``torch``, imported lazily inside this function so that this
    module remains importable without the optional ``predictor`` extra
    installed.

    Args:
        kmer_vocab_size: Number of distinct k-mers (input feature dimension).
        k: k-mer length used to build the input vocabulary.
        **params: Hyperparameters (e.g. hidden layer sizes, dropout),
            typically sourced from
            ``predictor/workflows/configs/config.yaml``.

    Returns:
        An untrained ``torch.nn.Module``.

    Raises:
        ImportError: If ``torch`` is not installed.
    """
    raise NotImplementedError
