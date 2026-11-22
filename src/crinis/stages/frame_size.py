"""The canonical frame size: what ingest produced, never what the config declared.

Every stage that asserts a frame shape needs the size ingest actually wrote, and ``video.size`` is *not*
it: that value is the bounding **box**, and `fit_size` fits the source into it. A 16:9 clip in a 640x480
box produces 640x360, and `fit_size` also forces even dimensions. Four separate stages had re-derived the
size from the config, so the same failure was waiting in each of them:

    CRN-4001 Array 'frames' has shape (16, 3, 360, 640) but expected (16, 3, 480, 640)

Measured on the first real run of `hands2d`, and found again by an audit of the stages that had not yet
run on hardware - the depth, camera and objects stages each carried their own copy. This module exists so
there is one reader, one key, and one fallback order.

The key and the path below are the contract with `s00_ingest`, which writes both. Keeping them here, and
importing them there, is what stops the two from drifting: `s09_assemble` had been reading
``crinis.image_size`` - a key ingest has never written - and silently falling back to a literal for the
life of the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from crinis.errors import ConfigError
from crinis.stages.base import StageContext

#: Schema-metadata key the ingest stage records the produced size under.
INDEX_SIZE_KEY = b"crinis.output_size"

#: The frame index's path within the ingest stage's artifacts. Must match `s00_ingest`'s constant.
INDEX_RELPATH = "frames/frames_index.parquet"


def frame_size_from_index(index: Path) -> tuple[int, int] | None:
    """``(width, height)`` as the index recorded it, or ``None`` when it cannot say.

    ``None`` rather than an exception: a missing or unreadable index is diagnosed by the frame-count
    check that follows in the stage, and one problem should not be reported twice.
    """
    if not index.exists():
        return None
    try:
        metadata = pq.read_schema(index).metadata or {}
        raw = metadata.get(INDEX_SIZE_KEY)
        if raw is None:
            return None
        width, height = json.loads(raw.decode("utf-8"))
        return int(width), int(height)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def canonical_frame_size(ctx: StageContext, *, stage: str) -> tuple[int, int]:
    """The size ingest produced, falling back to ``video.size`` only when the index cannot say.

    The fallback is a fallback and says so: before this module, four copies of the fallback were the
    *only* path anyone took, and each of them rejected correct frames on a real clip. ``video.size``
    remains the declaration of intent, which is why it is still consulted - last.
    """
    recorded = frame_size_from_index(ctx.layout.artifact("ingest", INDEX_RELPATH))
    if recorded is not None:
        return recorded
    raw = ctx.config_value("video.size")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    # CRN-0001 is the "a config key is missing or the wrong type" code, and it belongs to `ConfigError`
    # (prefix CRN-0). The four copies of this function raised CRN-0002 as a `ModelError` - wrong code,
    # wrong meaning (CRN-0002 is specifically about *even* dimensions) and wrong class, so the error
    # layer would have raised a ValueError about error codes instead of ever showing this message.
    raise ConfigError(
        "CRN-0001",
        f"video.size must be [width, height], got {raw!r}, and the ingest frame index records no "
        f"crinis.output_size for the {stage} stage to read instead",
    )
