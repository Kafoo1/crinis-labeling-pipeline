"""A label-free per-hand, per-frame gate: which hands may ship, and which go to a human.

Labels exist only *after* the labelling loop, so a gate that needs them cannot gate anything. This
module decides from artifacts the pipeline already wrote - the projected 3D joints, their temporal
neighbours, the hand's apparent size - and from the optional 2D lane when a run has one. It makes
**no accuracy claim**: ``no_signal`` means "every applicable signal was measured and none fired", not
"this hand is correct". ``flagged`` is the state a production run routes to a human.

The states are exactly ``no_signal``, ``flagged``, ``absent``. Every signal is ``ok``, ``flag`` or
``not_evaluated``; a signal with no data is reported as ``not_evaluated`` with the reason and can
never turn into a pass. ``clip_0001`` has no ``hands2d`` and no depth artifact, so two of the five
signals are inert on the clip this was built against - the report says so rather than counting them
as agreement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from crinis.errors import InputError
from crinis.labeling.cvat import (
    HAND_JOINTS,
    HAND_SIDES,
    RUN_ARTIFACTS,
    _project_joints,
)
from crinis.stages.frame_size import frame_size_from_index

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# ── Vocabulary ───────────────────────────────────────────────────────────────

#: Per-signal states. ``not_evaluated`` is a first-class outcome, never a quiet ``ok``.
OK = "ok"
FLAG = "flag"
NOT_EVALUATED = "not_evaluated"

#: Per-hand states. The gate never claims accuracy, and ``absent`` is not a pass.
NO_SIGNAL = "no_signal"
FLAGGED = "flagged"
ABSENT = "absent"

#: The plan's own jump rule skips frame pairs further apart than this: a hand that is not observed
#: for ten frames legitimately moves in those ten frames, so those pairs are not a "step"
#: (``dist/cvat_double_label.py`` uses the same guard when it picks hard frames).
JUMP_MAX_GAP_FRAMES = 3

#: Wrist (0) and middle-finger MCP (9) in the 21-joint MANO/MediaPipe order: their distance is the
#: hand's apparent size in the image.
WRIST_JOINT = 0
MIDDLE_MCP_JOINT = 9

#: The clip the gate was built against; the run's frame index overrides it when it can say.
DEFAULT_SIZE = (1280, 720)

#: The depth artifact the seam below is waiting for.
DEPTH_ARTIFACT = "artifacts/depth/depth_fused.npz"


@dataclass(frozen=True)
class Signal:
    """One measured reason to trust or distrust a hand on one frame.

    ``value`` is the measured number when there is one (a fraction, a pixel count, a ratio source).
    ``reason`` is set exactly when ``state`` is ``not_evaluated`` and says what was missing; it is
    also set on a flag that could not be measured (an unprojectable hand), because "flagged because
    the projection failed" is a different statement from "flagged, here is the number".
    """

    name: str
    state: str
    value: float | None = None
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HandVerdict:
    """The gate's answer for one hand on one frame, with every signal that produced it."""

    frame: int
    side: str
    state: str
    signals: list[Signal] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# ── Reading the run ──────────────────────────────────────────────────────────


def _load_arrays(run_dir: Path) -> tuple[Array, BoolArray, Array]:
    """The three arrays the gate needs, validated by shape rather than assumed."""
    hands3d_path = run_dir / RUN_ARTIFACTS["hands3d"]
    camera_path = run_dir / RUN_ARTIFACTS["camera"]
    for path, stage in ((hands3d_path, "hands3d"), (camera_path, "camera")):
        if not path.exists():
            raise InputError(
                "CRN-1001",
                f"{path} does not exist; the {stage} stage must run before the hand gate can score it",
            )
    with np.load(hands3d_path) as arrays:
        joints = np.asarray(arrays["joints_world_m"], dtype=np.float64)
        observed = np.asarray(arrays["observed"]).reshape(-1, len(HAND_SIDES))
    expected = (len(HAND_SIDES), HAND_JOINTS, 3)
    if joints.ndim != 4 or joints.shape[1:] != expected:
        raise InputError(
            "CRN-1001",
            f"{hands3d_path}: joints_world_m has shape {joints.shape}, expected "
            f"(frames, {expected[0]}, {expected[1]}, {expected[2]})",
        )
    if observed.shape[0] != joints.shape[0]:
        raise InputError(
            "CRN-1001",
            f"{hands3d_path}: observed has {observed.shape[0]} frames but joints_world_m has "
            f"{joints.shape[0]}",
        )
    with np.load(camera_path) as arrays:
        intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float64).reshape(-1)[:4]
    if intrinsics.shape != (4,):
        raise InputError(
            "CRN-1001",
            f"{camera_path}: intrinsics has {intrinsics.size} values, expected fx, fy, cx, cy",
        )
    return joints, observed.astype(bool), intrinsics


def _project_all(
    joints: Array, observed: BoolArray, intrinsics: Array
) -> tuple[Array, BoolArray]:
    """Project every observed hand, reusing the exporter's own projection.

    ``NaN`` and ``False`` mark a hand that is observed but cannot be projected (a joint behind the
    camera, or a non-finite one): the same condition under which the exporter writes no shape.
    """
    frames = joints.shape[0]
    points = np.full(
        (frames, len(HAND_SIDES), HAND_JOINTS, 2), np.nan, dtype=np.float64
    )
    projectable = np.zeros((frames, len(HAND_SIDES)), dtype=bool)
    for frame in range(frames):
        for side in range(len(HAND_SIDES)):
            if not observed[frame, side]:
                continue
            projected = _project_joints(joints[frame, side], intrinsics)
            if projected is None:
                continue
            points[frame, side] = projected
            projectable[frame, side] = True
    return points, projectable


def _load_two_d(run_dir: Path) -> tuple[Array, Array] | None:
    """The optional 2D lane, or ``None`` when this run has none."""
    path = run_dir / RUN_ARTIFACTS["hands2d"]
    if not path.exists():
        return None
    with np.load(path) as arrays:
        keypoints = np.asarray(arrays["kpts2d_px"], dtype=np.float64)
        confidence = np.asarray(arrays["conf"], dtype=np.float64).reshape(-1, len(HAND_SIDES))
    return keypoints, confidence


# ── Signals ──────────────────────────────────────────────────────────────────


def _steps(points: Array, projectable: BoolArray) -> tuple[list[tuple[int, int, float]], float | None]:
    """Median joint displacement for each consecutive observed pair, plus the hand's clip median.

    Only pairs no more than ``JUMP_MAX_GAP_FRAMES`` apart are steps: two observations ten frames
    apart are not evidence of a jump, they are evidence of a gap. The median over joints is robust to
    one bad fingertip, and the median over the whole clip is the hand's own motion baseline, so a
    frame is flagged only when it moves several times faster than this hand usually does. That
    baseline is global by design: on ``clip_0001`` it also flags the fast approach toward the camera,
    which is a real limitation of the signal and not an accuracy claim.
    """
    frames = np.flatnonzero(projectable)
    steps: list[tuple[int, int, float]] = []
    for before, after in pairwise(frames):
        start, stop = int(before), int(after)
        if stop - start > JUMP_MAX_GAP_FRAMES:
            continue
        displacement = float(
            np.median(np.linalg.norm(points[stop] - points[start], axis=1))
        )
        steps.append((start, stop, displacement))
    baseline = float(np.median([step[2] for step in steps])) if steps else None
    return steps, baseline


def _off_frame(points: Array | None, width: int, height: int, limit: float) -> Signal:
    """Fraction of joints outside ``[0, width) x [0, height)``."""
    if points is None:
        return Signal(
            "off_frame",
            FLAG,
            None,
            "the 3D joints could not be projected",
            {"width": width, "height": height},
        )
    inside = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    outside = int(points.shape[0] - np.count_nonzero(inside))
    fraction = outside / float(points.shape[0])
    return Signal(
        "off_frame",
        FLAG if fraction > limit else OK,
        fraction,
        None,
        {
            "width": width,
            "height": height,
            "outside": outside,
            "joints": int(points.shape[0]),
            "limit": limit,
        },
    )


def _jump(
    steps: Sequence[tuple[int, int, float]],
    baseline: float | None,
    frame: int,
    factor: float,
    projectable: bool,
) -> Signal:
    """The largest median joint displacement to a neighbouring observation, against the clip median."""
    if not projectable:
        return Signal(
            "jump", NOT_EVALUATED, None, "the 3D joints could not be projected", {}
        )
    displacements = [step[2] for step in steps if frame in (step[0], step[1])]
    if not displacements:
        return Signal(
            "jump",
            NOT_EVALUATED,
            None,
            f"no observed neighbouring frame within {JUMP_MAX_GAP_FRAMES} frames",
            {"median_step_px": baseline},
        )
    if baseline is None:
        return Signal(
            "jump",
            NOT_EVALUATED,
            None,
            "the clip has no consecutive observed frames to define a median step",
            {},
        )
    jump = max(displacements)
    ratio = jump / baseline if baseline > 0.0 else None
    return Signal(
        "jump",
        FLAG if jump > factor * baseline else OK,
        jump,
        None,
        {"median_step_px": baseline, "ratio": ratio, "factor": factor},
    )


def _palm(
    series: Array, projectable: BoolArray, frame: int, tolerance: float
) -> Signal:
    """``|wrist - middle_mcp|`` against the hand's clip median — a **diagnostic, never a flag**.

    Measured on clip_0001 this rule was actively harmful: it flagged 44 % of observed hand-frames,
    including hands the labeler accepted, because the projected palm length is computed from the very
    3D joints under suspicion and it legitimately changes with the hand's distance from the camera. A
    near hand is not a wrong hand. Kept as a value with its deviation so a report can show it, but it
    never decides a verdict - a gate that fires on good data is worse than no gate.
    """
    if not projectable[frame]:
        return Signal(
            "palm", NOT_EVALUATED, None, "the 3D joints could not be projected", {}
        )
    lengths = np.linalg.norm(
        series[projectable][:, WRIST_JOINT] - series[projectable][:, MIDDLE_MCP_JOINT],
        axis=-1,
    )
    median = float(np.median(lengths))
    value = float(
        np.linalg.norm(series[frame, WRIST_JOINT] - series[frame, MIDDLE_MCP_JOINT])
    )
    deviation = abs(value - median) / median if median > 0.0 else float("nan")
    return Signal(
        "palm",
        OK,
        value,
        "diagnostic only: the projected palm tracks distance, so a deviation is not evidence of error",
        {"median_px": median, "deviation": deviation, "tolerance": tolerance},
    )


def _two_d(
    frame: int,
    side: int,
    points: Array | None,
    artifact: tuple[Array, Array] | None,
    threshold_px: float,
    conf_threshold: float,
) -> Signal:
    """Median distance to the 2D lane, when that lane exists and is confident on this frame."""
    reference = RUN_ARTIFACTS["hands2d"]
    if artifact is None:
        return Signal(
            "two_d",
            NOT_EVALUATED,
            None,
            f"no {reference} in this run",
            {"artifact": reference},
        )
    keypoints, confidence = artifact
    if frame >= keypoints.shape[0]:
        return Signal(
            "two_d",
            NOT_EVALUATED,
            None,
            f"{reference} covers {keypoints.shape[0]} frames, fewer than the {frame + 1} needed",
            {"artifact": reference, "frames": int(keypoints.shape[0])},
        )
    if points is None:
        return Signal(
            "two_d", NOT_EVALUATED, None, "the 3D joints could not be projected", {}
        )
    confidence_value = float(confidence[frame, side])
    if confidence_value < conf_threshold:
        return Signal(
            "two_d",
            NOT_EVALUATED,
            None,
            f"2D confidence {confidence_value:.2f} is below the {conf_threshold:.2f} gate",
            {"confidence": confidence_value, "conf_threshold": conf_threshold},
        )
    distance = float(
        np.median(np.linalg.norm(points - keypoints[frame, side], axis=1))
    )
    return Signal(
        "two_d",
        FLAG if distance > threshold_px else OK,
        distance,
        None,
        {
            "confidence": confidence_value,
            "threshold_px": threshold_px,
            "conf_threshold": conf_threshold,
        },
    )


def _depth_agreement(frame: int, side: int) -> Signal:
    """Inert until a run carries depth. Deliberately not a pass.

    TODO(depth-gate): when ``artifacts/depth/depth_fused.npz`` is present, sample ``depth_m_fused``
    at the projected wrist pixel and compare it with the model's wrist depth (``wrist_trans_m`` or
    ``joints_world_m[frame, side, 0, 2]``). A disagreement of more than a few centimetres is the
    signature that drove 6 of the 9 right-hand corrections on clip_0001 (see
    ``PLANS/labelled-corrections-loop.md`` §14); it is the one signal that can see a depth error,
    and it needs an artifact this machine does not have.
    """
    return Signal(
        "depth_agreement",
        NOT_EVALUATED,
        None,
        "no depth artifact in this run",
        {"artifact": DEPTH_ARTIFACT, "frame": frame, "hand": HAND_SIDES[side]},
    )


# ── Public API ───────────────────────────────────────────────────────────────


def gate_run(
    run_dir: Path | str,
    *,
    max_off_frame: float = 0.2,
    jump_factor: float = 5.0,
    palm_tolerance: float = 0.20,
    two_d_px: float = 20.0,
    two_d_conf: float = 0.3,
) -> list[HandVerdict]:
    """Score every hand on every frame of one run, in ``(frame, left, right)`` order.

    ``off_frame`` is the fraction of the 21 projected joints outside the image; ``jump`` is the
    largest median joint displacement to a neighbouring observation against the hand's own clip
    median step; ``palm`` is the projected wrist-to-middle-MCP length against the hand's clip median;
    ``two_d`` needs the optional 2D lane; ``depth_agreement`` is inert until a depth artifact exists.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise InputError("CRN-1001", f"run directory {run_dir} does not exist")
    joints, observed, intrinsics = _load_arrays(run_dir)
    points, projectable = _project_all(joints, observed, intrinsics)
    width, height = frame_size_from_index(run_dir / RUN_ARTIFACTS["index"]) or DEFAULT_SIZE
    two_d = _load_two_d(run_dir)

    steps = {
        side: _steps(points[:, side], projectable[:, side])
        for side in range(len(HAND_SIDES))
    }

    verdicts: list[HandVerdict] = []
    for frame in range(joints.shape[0]):
        for side, label in enumerate(HAND_SIDES):
            if not observed[frame, side]:
                verdicts.append(HandVerdict(frame=frame, side=label, state=ABSENT))
                continue
            side_steps, baseline = steps[side]
            usable = bool(projectable[frame, side])
            signals = [
                _off_frame(points[frame, side] if usable else None, width, height, max_off_frame),
                _jump(side_steps, baseline, frame, jump_factor, usable),
                _palm(points[:, side], projectable[:, side], frame, palm_tolerance),
                _two_d(
                    frame,
                    side,
                    points[frame, side] if usable else None,
                    two_d,
                    two_d_px,
                    two_d_conf,
                ),
                _depth_agreement(frame, side),
            ]
            flags = [signal.name for signal in signals if signal.state == FLAG]
            verdicts.append(
                HandVerdict(
                    frame=frame,
                    side=label,
                    state=FLAGGED if flags else NO_SIGNAL,
                    signals=signals,
                    flags=flags,
                )
            )
    return verdicts


def summary(verdicts: Sequence[HandVerdict]) -> dict[str, Any]:
    """Counts per hand side plus the frames the gate flagged, in verdict order."""
    counts: dict[str, dict[str, int]] = {
        label: {NO_SIGNAL: 0, FLAGGED: 0, ABSENT: 0} for label in HAND_SIDES
    }
    flagged: list[dict[str, Any]] = []
    for verdict in verdicts:
        counts[verdict.side][verdict.state] += 1
        if verdict.state == FLAGGED:
            flagged.append(
                {
                    "frame": verdict.frame,
                    "hand": verdict.side,
                    "signals": list(verdict.flags),
                }
            )
    return {
        "frames": len({verdict.frame for verdict in verdicts}),
        "counts": counts,
        "flagged_frames": flagged,
    }
