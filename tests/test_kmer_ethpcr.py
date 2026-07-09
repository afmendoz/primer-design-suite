"""Tests for the k-mer featurizer and the ETH PCR-bias loader."""

import pandas as pd
import pytest

from primer_core.features.kmer import all_kmers, kmer_frequencies
from primer_core.io.ethpcr import SOURCES, load_ethpcr


def test_all_kmers_count_and_order():
    assert len(all_kmers(3)) == 64
    assert all_kmers(1) == ["A", "C", "G", "T"]


def test_kmer_frequencies_hand_checked():
    f = kmer_frequencies("ACGT", k=2)  # windows: AC, CG, GT
    assert len(f) == 16
    assert f["AC"] == pytest.approx(1 / 3)
    assert f["CG"] == pytest.approx(1 / 3)
    assert f["GT"] == pytest.approx(1 / 3)
    assert f["AA"] == 0.0
    assert sum(f.values()) == pytest.approx(1.0)


def test_kmer_frequencies_errors():
    with pytest.raises(ValueError):
        kmer_frequencies("AC", k=3)  # shorter than k
    with pytest.raises(ValueError):
        kmer_frequencies("ACGTN", k=2)  # non-ACGT


def test_load_ethpcr(tmp_path):
    for src in SOURCES:
        (tmp_path / src).mkdir()
        payload = {
            "rest": pd.DataFrame({"sequence": ["ACGTACGT", "GGCCGGCC"], "eff": [1.0, 0.99]}),
            "bottom": pd.DataFrame({"sequence": ["ATATATAT"], "eff": [0.8]}),
        }
        pd.to_pickle(payload, tmp_path / src / "bad_seqs_2perc.pkl")

    df = load_ethpcr(tmp_path, per_source_cap=None)
    assert set(df["source"]) == set(SOURCES)
    assert len(df) == 3 * len(SOURCES)  # 2 rest + 1 bottom per source
    assert set(df.columns) == {"sequence", "eff", "source"}
    assert df["eff"].dtype == float
