"""Labeling handoff: export pipeline pre-annotations and ingest reviewed corrections.

CVAT is the chosen tool (decision G1); this package speaks its XML dialect and records
provenance so a pre-annotation cannot be mistaken for a human label.
"""

from __future__ import annotations

from crinis.labeling.cvat import (
    CvatAnnotation,
    CvatCorrections,
    CvatDocument,
    CvatExport,
    export_cvat,
    ingest_cvat,
    parse_cvat,
)

__all__ = [
    "CvatAnnotation",
    "CvatCorrections",
    "CvatDocument",
    "CvatExport",
    "export_cvat",
    "ingest_cvat",
    "parse_cvat",
]
