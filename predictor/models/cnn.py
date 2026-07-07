"""Small deep model: 1D-CNN / k-mer MLP over raw sequence (PyTorch).

Top of the modeling ladder in CLAUDE.md. ``torch`` is an optional dependency
(see ``pyproject.toml``'s ``predictor`` extra), so the import is deferred
into the functions below rather than performed at module scope — this module
must remain importable even when torch is not installed. These builders are
never called unless the config sets ``model.name`` to ``"cnn"`` / ``"kmer_mlp"``.
"""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    """Import torch, raising a clear ImportError if it is missing."""
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as exc:  # pragma: no cover - torch intentionally optional
        raise ImportError(
            "torch is required for the CNN / k-mer MLP models; install the "
            "'predictor' extra (pip install -e '.[predictor]') to enable them."
        ) from exc


def build_sequence_cnn(
    seq_length: int,
    n_channels: int = 4,
    n_filters: int = 32,
    kernel_size: int = 5,
    dropout: float = 0.2,
    **params: Any,
) -> Any:
    """Construct a small 1D-CNN over one-hot-encoded primer sequence.

    Requires ``torch``, imported lazily so this module stays importable
    without the optional ``predictor`` extra installed.

    Args:
        seq_length: Fixed input sequence length (one-hot encoded).
        n_channels: Number of input channels (4 for one-hot DNA).
        n_filters: Number of convolution filters.
        kernel_size: Convolution kernel width.
        dropout: Dropout probability before the output head.
        **params: Ignored extra config keys.

    Returns:
        An untrained ``torch.nn.Module`` mapping ``(batch, n_channels,
        seq_length)`` to a scalar efficiency prediction.

    Raises:
        ImportError: If ``torch`` is not installed.
    """
    torch = _require_torch()
    from torch import nn

    class SequenceCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(n_channels, n_filters, kernel_size, padding=kernel_size // 2)
            self.act = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(n_filters, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.act(self.conv(x))
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            return self.head(x).squeeze(-1)

    return SequenceCNN()


def build_kmer_mlp(
    kmer_vocab_size: int,
    k: int = 4,
    hidden_sizes: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    **params: Any,
) -> Any:
    """Construct a small MLP over k-mer frequency features of the sequence.

    Requires ``torch``, imported lazily so this module stays importable
    without the optional ``predictor`` extra installed.

    Args:
        kmer_vocab_size: Number of distinct k-mers (input feature dimension).
        k: k-mer length used to build the input vocabulary (recorded for
            reference; the vocab size already encodes it).
        hidden_sizes: Sizes of the hidden layers.
        dropout: Dropout probability between layers.
        **params: Ignored extra config keys.

    Returns:
        An untrained ``torch.nn.Module`` mapping a k-mer frequency vector to a
        scalar efficiency prediction.

    Raises:
        ImportError: If ``torch`` is not installed.
    """
    _require_torch()
    from torch import nn

    layers: list[Any] = []
    in_dim = kmer_vocab_size
    for hidden in hidden_sizes:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))

    class KmerMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
            return self.net(x).squeeze(-1)

    return KmerMLP()
