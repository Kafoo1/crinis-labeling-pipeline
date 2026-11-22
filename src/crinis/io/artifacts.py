"""Shared artifact loading helpers.

Every consumer of an upstream artifact must check the keys it depends on *before* indexing
them, and must name the missing key and the artifact it came from when one is absent. Two
stages needing the same twenty lines is how a contract drifts, so it lives here once.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from crinis.errors import ArtifactError
from crinis.io.storage import load_npz


def require_keys(
    artifact: str,
    arrays: Mapping[str, np.ndarray],
    required: Mapping[str, str],
    *,
    stage: str | None = None,
) -> None:
    """Raise ``CRN-2002`` naming the first missing key of *arrays* and why it is needed.

    ``required`` maps a key to the reason the caller needs it, so the message tells the reader
    what breaks rather than only what is absent.
    """
    missing = [key for key in required if key not in arrays]
    if missing:
        key = missing[0]
        raise ArtifactError(
            "CRN-2002",
            f"artifact {artifact} is missing {key!r} ({required[key]}); "
            f"present keys: {sorted(arrays)}",
            stage=stage,
        )


def load_artifact(
    path: Path,
    *,
    artifact: str,
    required: Mapping[str, str],
    stage: str | None = None,
) -> dict[str, np.ndarray]:
    """Load an ``.npz`` artifact and check its required keys in one call."""
    arrays = load_npz(path)
    require_keys(artifact, arrays, required, stage=stage)
    return arrays
