"""Pydantic schemas for dataset validation and label provenance.

By project convention: every dataset in ``data/`` must record its source and what its
label actually measures in a sidecar ``*.provenance.yaml`` file. Proxy labels
(e.g. thermodynamically-derived rather than experimentally measured) are
allowed as a plumbing stepping-stone but must be flagged as proxy everywhere
they surface (README, notebooks, model cards) — the ``is_proxy_label`` /
``label_description`` fields below exist to make that flag unavoidable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """Provenance sidecar schema (``*.provenance.yaml``) for a dataset.

    Attributes:
        source: Where the dataset came from (e.g. citation, URL, accession).
        license: Data license/terms under which it may be used.
        label_description: Plain-language description of what the label
            column actually measures.
        is_proxy_label: True if the label is a proxy (e.g. thermodynamically
            derived) rather than an experimentally validated measurement.
            Must be surfaced wherever the label is used downstream.
        grouping_key: Column name used to group records for grouped CV
            (e.g. gene or template ID) to prevent train/test leakage.
        notes: Free-text caveats.
    """

    source: str
    license: str
    label_description: str
    is_proxy_label: bool
    grouping_key: str
    notes: str | None = None


class PrimerRecord(BaseModel):
    """Schema for a single primer/assay record in a training dataset.

    Attributes:
        primer_id: Unique identifier for this primer/assay record.
        sequence: Primer sequence (5'->3', DNA alphabet ACGT).
        template_id: Identifier of the template/amplicon this primer targets;
            doubles as the default grouped-CV grouping key candidate.
        label: Target value (e.g. measured or proxy efficiency).
        metadata: Arbitrary additional fields carried through featurization.
    """

    primer_id: str
    sequence: str
    template_id: str
    label: float
    metadata: dict[str, object] = Field(default_factory=dict)


def validate_dataset_schema(records: list[dict[str, object]]) -> list[PrimerRecord]:
    """Validate a list of raw records against ``PrimerRecord``.

    Args:
        records: Raw dataset rows (e.g. from ``DataFrame.to_dict('records')``).

    Returns:
        Validated ``PrimerRecord`` instances.

    Raises:
        pydantic.ValidationError: If any record fails schema validation.
    """
    return [PrimerRecord(**record) for record in records]
