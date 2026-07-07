"""Tests for primer_core.io: sequences, schema, datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from primer_core.io.datasets import list_datasets, load_dataset, load_provenance
from primer_core.io.schema import PrimerRecord, ProvenanceRecord, validate_dataset_schema
from primer_core.io.sequences import load_fasta, load_single_sequence, write_fasta

_PROVENANCE_YAML = """\
source: "synthetic test data"
license: "CC0"
label_description: "proxy efficiency from thermodynamics"
is_proxy_label: true
grouping_key: "template_id"
notes: "unit test fixture"
"""


def test_fasta_round_trip(tmp_path: Path) -> None:
    records = {"seqA": "ACGTACGT", "seqB": "TTTTGGGG"}
    fasta = tmp_path / "seqs.fasta"
    write_fasta(records, fasta)
    loaded = load_fasta(fasta)
    assert loaded == records


def test_load_single_sequence(tmp_path: Path) -> None:
    fasta = tmp_path / "one.fasta"
    write_fasta({"only": "ACGTACGT"}, fasta)
    assert load_single_sequence(fasta) == "ACGTACGT"


def test_load_single_sequence_rejects_multi(tmp_path: Path) -> None:
    fasta = tmp_path / "two.fasta"
    write_fasta({"a": "ACGT", "b": "TGCA"}, fasta)
    with pytest.raises(ValueError):
        load_single_sequence(fasta)


def test_load_fasta_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fasta(tmp_path / "nope.fasta")


def test_validate_dataset_schema() -> None:
    records = [
        {"primer_id": "p1", "sequence": "ACGT", "template_id": "t1", "label": 0.9},
        {"primer_id": "p2", "sequence": "TGCA", "template_id": "t1", "label": 0.4},
    ]
    validated = validate_dataset_schema(records)
    assert all(isinstance(r, PrimerRecord) for r in validated)
    assert validated[0].label == 0.9


def _write_dataset_with_provenance(tmp_path: Path) -> Path:
    csv = tmp_path / "ds.csv"
    csv.write_text("primer_id,sequence,template_id,label\np1,ACGT,t1,0.9\n")
    (tmp_path / "ds.provenance.yaml").write_text(_PROVENANCE_YAML)
    return csv


def test_load_provenance(tmp_path: Path) -> None:
    csv = _write_dataset_with_provenance(tmp_path)
    prov = load_provenance(csv)
    assert isinstance(prov, ProvenanceRecord)
    assert prov.is_proxy_label is True
    assert prov.grouping_key == "template_id"


def test_load_provenance_missing(tmp_path: Path) -> None:
    csv = tmp_path / "orphan.csv"
    csv.write_text("primer_id\np1\n")
    with pytest.raises(FileNotFoundError):
        load_provenance(csv)


def test_list_datasets_excludes_sidecar(tmp_path: Path) -> None:
    _write_dataset_with_provenance(tmp_path)
    (tmp_path / "other.tsv").write_text("a\tb\n1\t2\n")
    found = list_datasets(tmp_path)
    names = {p.name for p in found}
    assert names == {"ds.csv", "other.tsv"}
    assert all(not p.name.endswith(".provenance.yaml") for p in found)


def test_load_dataset_happy_path(tmp_path: Path) -> None:
    csv = _write_dataset_with_provenance(tmp_path)
    df = load_dataset(csv)
    assert list(df["primer_id"]) == ["p1"]


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_load_dataset_missing_provenance_raises_value_error(tmp_path: Path) -> None:
    csv = tmp_path / "noprov.csv"
    csv.write_text("primer_id\np1\n")
    with pytest.raises(ValueError):
        load_dataset(csv)
