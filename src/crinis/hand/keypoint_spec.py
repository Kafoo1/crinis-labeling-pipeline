"""Hand keypoint specification - frozen constants for the 21-joint MANO/MediaPipe order.

Convention (§1.1 of the plan):
- The 21 joints are ordered: wrist, then for each finger (thumb → pinky):
  MCP, PIP, DIP, TIP.  This order is **identical index-for-index** to
  MediaPipe Hands (``WRIST`` = 0, ``THUMB_CMC`` = 1, …, ``PINKY_TIP`` = 20).
- The MANO reorder ``MANO_REORDER`` maps ``th_jtr`` (16 regressor joints) +
  5 fingertip vertices appended, producing the 21-joint sequence above.
  Specifically: ``th_jtr[:, MANO_REORDER[:16]]`` gives joints 0-15, and the
  5 appended fingertip vertex positions give joints 16-20 in the thumb, index,
  middle, ring, pinky order.
- ``HAND_SLOTS = ("left", "right")``; axis-0 of hand arrays: ``LEFT=0``,
  ``RIGHT=1`` (HAWoR's ``hand2idx`` convention).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from crinis.errors import GeometryError

# ---------------------------------------------------------------------------
# Joint names - identical index-for-index to MediaPipe Hands
# ---------------------------------------------------------------------------

JOINT_NAMES: tuple[str, ...] = (
    "wrist",          # 0
    "thumb_mcp",      # 1
    "thumb_pip",      # 2
    "thumb_dip",      # 3
    "thumb_tip",      # 4
    "index_mcp",      # 5
    "index_pip",      # 6
    "index_dip",      # 7
    "index_tip",      # 8
    "middle_mcp",     # 9
    "middle_pip",     # 10
    "middle_dip",     # 11
    "middle_tip",     # 12
    "ring_mcp",       # 13
    "ring_pip",       # 14
    "ring_dip",       # 15
    "ring_tip",       # 16
    "pinky_mcp",      # 17
    "pinky_pip",      # 18
    "pinky_dip",      # 19
    "pinky_tip",      # 20
)

# ---------------------------------------------------------------------------
# MediaPipe landmark names (same index ordering as JOINT_NAMES)
# ---------------------------------------------------------------------------

MEDIAPIPE_JOINT_NAMES: tuple[str, ...] = (
    "WRIST",          # 0
    "THUMB_CMC",      # 1
    "THUMB_MCP",      # 2
    "THUMB_IP",       # 3
    "THUMB_TIP",      # 4
    "INDEX_FINGER_MCP",   # 5
    "INDEX_FINGER_PIP",   # 6
    "INDEX_FINGER_DIP",   # 7
    "INDEX_FINGER_TIP",   # 8
    "MIDDLE_FINGER_MCP",  # 9
    "MIDDLE_FINGER_PIP",  # 10
    "MIDDLE_FINGER_DIP",  # 11
    "MIDDLE_FINGER_TIP",  # 12
    "RING_FINGER_MCP",    # 13
    "RING_FINGER_PIP",    # 14
    "RING_FINGER_DIP",    # 15
    "RING_FINGER_TIP",    # 16
    "PINKY_MCP",          # 17
    "PINKY_PIP",          # 18
    "PINKY_DIP",          # 19
    "PINKY_TIP",          # 20
)

# ---------------------------------------------------------------------------
# MANO reorder
# ---------------------------------------------------------------------------

# ``th_jtr[:, MANO_REORDER]`` selects the 16 regressor joints followed by 5
# fingertip vertices appended, producing the 21-joint canonical order above.
# Source: ``hassony2/manopth`` ``manolayer.py``.
MANO_REORDER: tuple[int, ...] = (
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
)

# ---------------------------------------------------------------------------
# Fingertip constants
# ---------------------------------------------------------------------------

# MANO mesh vertex indices for the five fingertips (thumb, index, middle, ring, pinky).
FINGERTIP_VERTEX_IDS: tuple[int, ...] = (745, 317, 444, 556, 673)

# Joint indices of the five fingertips in the 21-joint canonical order.
FINGERTIP_JOINT_IDS: tuple[int, ...] = (4, 8, 12, 16, 20)

# ---------------------------------------------------------------------------
# Per-finger ranges (inclusive indices into the 21-joint order)
# ---------------------------------------------------------------------------

# Maps finger name → (mcp_index, tip_index), both inclusive.
FINGER_RANGES: dict[str, tuple[int, int]] = {
    "thumb":  (1, 4),
    "index":  (5, 8),
    "middle": (9, 12),
    "ring":   (13, 16),
    "pinky":  (17, 20),
}

# ---------------------------------------------------------------------------
# Scalar constants
# ---------------------------------------------------------------------------

WRIST_JOINT_INDEX: int = 0
NUM_JOINTS: int = 21
NUM_VERTICES: int = 778

# Hand slot ordering: axis-0 of every ``[2, ...]`` hand array.
HAND_SLOTS: tuple[str, str] = ("left", "right")
LEFT: int = 0   # HAWoR hand2idx["left"]  == 0
RIGHT: int = 1  # HAWoR hand2idx["right"] == 1

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def index_of(name: str) -> int:
    """Return the joint index for a given joint name.

    Args:
        name: One of the 21 joint names in :data:`JOINT_NAMES`.

    Returns:
        Integer index 0-20.

    Raises:
        KeyError: if *name* is not a recognised joint name.
    """
    try:
        return JOINT_NAMES.index(name)
    except ValueError:
        raise KeyError(
            f"Unknown joint name {name!r}; valid names are {JOINT_NAMES}"
        ) from None


def finger_ranges() -> dict[str, tuple[int, int]]:
    """Return a copy of :data:`FINGER_RANGES`.

    Use this instead of hardcoding integer ranges in downstream modules.

    Returns:
        Dict mapping finger name → ``(mcp_index, tip_index)`` (inclusive).
    """
    return dict(FINGER_RANGES)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def assert_valid_keypoints(
    kpts: np.ndarray[Any, np.dtype[Any]],
    name: str,
    *,
    stage: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Validate a keypoint array.

    Checks:
    1. Shape: ``(..., 21, 2)`` or ``(..., 21, 3)``.
    2. All values are finite.
    3. For 2-D keypoints (last dim = 2), if *width* and *height* are provided,
       pixel values are inside ``[0, width) x [0, height)``.

    Args:
        kpts:   Keypoint array; last two dims must be ``(21, 2)`` or ``(21, 3)``.
        name:   Array name used in error messages.
        stage:  Optional pipeline stage label for the error.
        width:  Image width (used for 2-D bounds check when provided).
        height: Image height (used for 2-D bounds check when provided).

    Raises:
        GeometryError("CRN-4001"): on any shape, finiteness, or bounds violation.
    """
    kpts = np.asarray(kpts)

    # Shape check
    if kpts.ndim < 2 or kpts.shape[-2] != NUM_JOINTS or kpts.shape[-1] not in (2, 3):
        raise GeometryError(
            "CRN-4001",
            f"Array {name!r} has shape {kpts.shape}; expected (..., {NUM_JOINTS}, 2|3)",
            stage=stage,
        )

    # Finiteness check
    if not np.all(np.isfinite(kpts)):
        bad = int(np.argmax(~np.isfinite(kpts).ravel()))
        idx = np.unravel_index(bad, kpts.shape)
        raise GeometryError(
            "CRN-4001",
            f"Array {name!r} contains non-finite values; first bad index={idx}",
            stage=stage,
        )

    # Bounds check for 2-D keypoints
    if kpts.shape[-1] == 2 and width is not None and height is not None:
        u = kpts[..., 0]
        v = kpts[..., 1]
        if np.any((u < 0) | (u >= width) | (v < 0) | (v >= height)):
            bad_u = np.argmax(((u < 0) | (u >= width)).ravel())
            bad_v = np.argmax(((v < 0) | (v >= height)).ravel())
            bad = int(bad_u) if ((u < 0) | (u >= width)).ravel()[bad_u] else int(bad_v)
            idx = np.unravel_index(int(bad), kpts.shape[:-1])
            raise GeometryError(
                "CRN-4001",
                f"Array {name!r} has pixel coordinates outside [0,{width})x[0,{height}); "
                f"first offending joint index={idx}",
                stage=stage,
            )
