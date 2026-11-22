"""Canonical hand-evidence document, reader and scoring for every 2-D comparison.

**A number without its lineage is not a measurement.** That rule is the whole reason this module
exists: a median joint error is only meaningful next to the grid it was computed on, the reference
it was computed against, the fraction of the reference it actually covers, and the count of joints,
hands and frames that were left out. ``crinis.hand-evidence/1`` is the one schema those facts are
written into, and this module is its single owner - readers, scorer, bootstrap and validator.

What is measured, stated once:

* **Reference visibility.** A joint marked ``visibility`` 0 (occluded) or 1 (outside the frame) is
  not an observation and is excluded from the primary statistic. An **absent** visibility attribute
  is ``None``: *unknown*, and unknown joints are scored rather than silently promoted to visible or
  silently dropped. The all-accepted statistic scores every placed, predicted joint regardless of
  visibility so the sensitivity to that rule is visible instead of implied.
* **Missing hands.** A label hand the run did not predict is counted as missing; its joints are never
  scored as zero error. Extra predictions are counted only on frames the reference covers, because
  every other frame would measure the reference's coverage, not the model's error.
* **Side.** ``left``/``right``/``unknown`` are kept apart. ``unknown`` is a prediction whose anatomy
  is not established; it is never merged into one of the two named sides.
* **Grid.** Every distance is in pixels on the declared prediction grid. Label coordinates are
  rescaled by :func:`grid_scales`, which requires a uniform scale: independent x and y factors are a
  crop, not a resize, and are refused rather than reported as a distance.
* **Uncertainty.** Intervals are percentile bootstraps over **source frames**, resampling whole
  frames (both hands, all joints) so that within-frame correlation is preserved. The paired delta is
  ``candidate - baseline`` on the intersection of exact ``(frame, side, joint)`` keys: negative is
  better. No interval computed here is a universal measurement floor.

The document is JSON: floats are computed unrounded in float64, rounded to six decimals on write and
printed with two. A value that cannot be computed is ``null`` with a non-empty ``absent_reason`` -
never NaN, never 0, never an omitted key.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, cast
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from crinis.hand.keypoint_spec import HAND_SLOTS, JOINT_NAMES
from crinis.labeling.cvat import VISIBILITY_ATTRIBUTE, VISIBILITY_VALUES

#: Version string every evidence document carries as its first key.
SCHEMA = "crinis.hand-evidence/1"

#: The five claims this format can carry. Each one licenses a different sentence.
KINDS: tuple[str, ...] = (
    "human_reference_agreement",
    "independent_accuracy",
    "correction_repeatability",
    "model_agreement",
    "structural_plausibility",
)

#: Keep the suite's existing metric name: renaming it would fork the two readers.
METRIC_HAND = "hand_joint_error_px"
#: The sensitivity statistic: every placed joint with a prediction, whatever its visibility says.
METRIC_HAND_ALL_ACCEPTED = "hand_joint_error_px_all_accepted"

#: Coverage thresholds as a fraction of the labelled hand's pixel extent. Keys use ``f"{t:g}"``,
#: which is the convention ``crinis.release.hands`` already writes ("0.05", "0.1").
COVERAGE_THRESHOLDS: tuple[float, ...] = (0.05, 0.10)

#: Anatomical sides, in report order. ``unknown`` is never folded into the two named sides.
SIDES: tuple[str, ...] = (*HAND_SLOTS, "unknown")

#: How a reference document's labels came to exist. ``anchored_correction`` means the labeler edited
#: a pre-annotation, so the labels share its ancestry and are not independent reads.
LABEL_KINDS: tuple[str, ...] = (
    "anchored_correction",
    "independent_de_novo",
    "model_preannotation",
)

#: Label name <-> side. ``hand_unknown`` is written from S4 on, and read here already so a document
#: carrying it round-trips instead of being silently converted to a named hand.
SIDE_LABELS: dict[str, str] = {
    "left": "hand_left",
    "right": "hand_right",
    "unknown": "hand_unknown",
}
_LABEL_SIDES: dict[str, str] = {label: side for side, label in SIDE_LABELS.items()}

#: Paired-delta scopes, and the statistic each one is required to carry.
PAIRED_SCOPES: dict[str, str] = {
    "pooled_median_px": "median_px",
    "pooled_p90_px": "p90_px",
    "left_median_px": "median_px",
    "right_median_px": "median_px",
}
PAIRED_STATISTICS: tuple[str, ...] = ("median_px", "p90_px")

_PERCENTILES = (2.5, 97.5)
_FRAME_DIGITS_RE = re.compile(r"\d+")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "kind",
    "source_take_sha256",
    "run_sha256",
    "prediction_artifacts",
    "references",
    "grid",
    "n_frames",
    "n_hands",
    "n_joints",
    "missing_hands",
    "extra_hands",
    "unknown_side_predictions",
    "unknown_visibility_joints",
    "unscored_frames",
    "metrics",
    "limitations",
    "created_utc",
)
_OPTIONAL_KEYS: tuple[str, ...] = ("paired_delta",)
_METRIC_KEYS: tuple[str, ...] = (
    "n_joints",
    "n_hands",
    "n_frames",
    "median_px",
    "mean_px",
    "p90_px",
    "max_px",
    "coverage",
    "by_hand",
    "absent_reason",
)
_HAND_KEYS: tuple[str, ...] = (
    "n_joints",
    "n_hands",
    "n_frames",
    "median_px",
    "mean_px",
    "p90_px",
    "max_px",
)
_PAIRED_KEYS: tuple[str, ...] = (
    "metric",
    "baseline",
    "candidate",
    "delta_px",
    "ci_low",
    "ci_high",
    "n_pairs",
    "n_frames",
)
_GRID_KEYS: tuple[str, ...] = ("width", "height", "scale_x", "scale_y")
_ARTIFACT_KEYS: tuple[str, ...] = ("path", "sha256")
_REFERENCE_KEYS: tuple[str, ...] = ("path", "sha256", "protocol", "seed_sha256", "label_kind")

Predictions = Mapping[tuple[int, str], Mapping[str, tuple[float, float]]]


# ── Readers ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HandJoint:
    """One labeled joint. Every flag is ``None`` when the document does not state it."""

    xy: tuple[float, float]
    visibility: int | None
    outside: bool | None
    occluded: bool | None


@dataclass(frozen=True)
class HandShape:
    """One hand in one source frame, keyed by :data:`JOINT_NAMES` members (partial shapes allowed)."""

    frame: int
    side: str
    joints: Mapping[str, HandJoint]


@dataclass(frozen=True)
class HandDocument:
    """A reference document: the grid it declares, and the hands it places."""

    path: Path
    grid: tuple[int, int] | None
    shapes: tuple[HandShape, ...]


def _flag(
    element: ET.Element, name: str, *, path: Path, frame: int, side: str, joint: str
) -> bool | None:
    raw = element.get(name)
    if raw is None:
        return None
    text = raw.strip()
    if text == "1":
        return True
    if text == "0":
        return False
    raise ValueError(
        f"{path}: frame {frame} side {side!r} joint {joint!r} has non-boolean {name}={raw!r}"
    )


def _joint(
    element: ET.Element, *, path: Path, frame: int, side: str, joint: str
) -> HandJoint:
    raw = element.get("points")
    parts = (raw or "").split(",")
    if len(parts) != 2:
        raise ValueError(
            f"{path}: frame {frame} side {side!r} joint {joint!r} has no single 'x,y' point "
            f"(found {raw!r})"
        )
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"{path}: frame {frame} side {side!r} joint {joint!r} has non-numeric coordinates "
            f"{raw!r}"
        ) from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(
            f"{path}: frame {frame} side {side!r} joint {joint!r} has non-finite coordinates "
            f"{raw!r}"
        )
    visibility: int | None = None
    for attribute in element.findall("attribute"):
        if (attribute.get("name") or "").strip() != VISIBILITY_ATTRIBUTE:
            continue
        text = (attribute.text or "").strip()
        if text not in VISIBILITY_VALUES:
            raise ValueError(
                f"{path}: frame {frame} side {side!r} joint {joint!r} has visibility {text!r}, "
                f"expected one of {VISIBILITY_VALUES}"
            )
        visibility = int(text)
        break
    return HandJoint(
        xy=(x, y),
        visibility=visibility,
        outside=_flag(element, "outside", path=path, frame=frame, side=side, joint=joint),
        occluded=_flag(element, "occluded", path=path, frame=frame, side=side, joint=joint),
    )


def _shape(
    element: ET.Element, *, path: Path, frame: int, side: str
) -> HandShape | None:
    joints: dict[str, HandJoint] = {}
    for point in element.findall("points"):
        joint = (point.get("label") or "").strip()
        if joint not in JOINT_NAMES:
            raise ValueError(
                f"{path}: frame {frame} side {side!r} has a joint labelled {joint!r}; expected "
                f"one of the 21 names in crinis.hand.keypoint_spec.JOINT_NAMES"
            )
        if joint in joints:
            raise ValueError(
                f"{path}: frame {frame} side {side!r} joint {joint!r} appears twice in one shape"
            )
        joints[joint] = _joint(point, path=path, frame=frame, side=side, joint=joint)
    if not joints:
        return None
    return HandShape(frame=frame, side=side, joints=joints)


def _side(label: str | None) -> str | None:
    return _LABEL_SIDES.get((label or "").strip())


def _add_shape(
    shapes: list[HandShape], shape: HandShape, *, path: Path, seen: set[tuple[int, str]]
) -> None:
    key = (shape.frame, shape.side)
    if key in seen:
        joint = next(iter(shape.joints))
        raise ValueError(
            f"{path}: frame {shape.frame} side {shape.side!r} has more than one shape "
            f"(joint {joint!r}): one side in one frame is one hand"
        )
    seen.add(key)
    shapes.append(shape)


def _image_frame(name: str | None, element_id: str | None, position: int) -> int:
    """The source frame an images-export filename encodes: its **last** integer run.

    ``b03_frame_0045.png`` is frame 45, not 3045 - the same rule
    :func:`crinis.labeling.cvat._image_frame` uses. A name that carries no digits falls back to the
    element's own ``id`` and then to its document position, and the caller is told which happened by
    nothing more than the resulting number: the export simply cannot say more.
    """
    digits = _FRAME_DIGITS_RE.findall(name or "")
    if digits:
        return int(digits[-1])
    raw = (element_id or "").strip()
    if raw.isdigit():
        return int(raw)
    return position


def _skeleton_frame(skeleton: ET.Element, *, path: Path, side: str) -> int:
    raw = skeleton.get("frame")
    if raw is None:
        for point in skeleton.findall("points"):
            raw = point.get("frame")
            if raw is not None:
                break
    if raw is None:
        raise ValueError(f"{path}: track {side!r} has a skeleton with no frame attribute")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: skeleton frame {raw!r} is not an integer") from exc


def _declared_grid(root: ET.Element, path: Path) -> tuple[int, int] | None:
    """The pixel grid the document declares, or ``None`` when it declares none.

    An images export writes ``width``/``height`` on every ``<image>``; a video export may carry one
    ``<original_size>``. A non-positive pair is CVAT's way of saying "unknown" and declares nothing.
    Two *different* positive sizes in one document are a contradiction, not a grid.
    """
    sizes: set[tuple[int, int]] = set()
    for image in root.findall("image"):
        raw_width, raw_height = image.get("width"), image.get("height")
        if not raw_width or not raw_height:
            continue
        try:
            size = (int(raw_width), int(raw_height))
        except ValueError as exc:
            raise ValueError(f"{path}: non-numeric image size {raw_width}x{raw_height}") from exc
        if size[0] <= 0 or size[1] <= 0:
            continue
        sizes.add(size)
    if len(sizes) > 1:
        raise ValueError(
            f"{path}: images declare contradictory sizes {sorted(sizes)}; one document describes "
            "one grid, so no scale can be derived from it"
        )
    if sizes:
        return sizes.pop()
    for node in root.iter("original_size"):
        raw_width, raw_height = node.findtext("width"), node.findtext("height")
        if not raw_width or not raw_height:
            continue
        try:
            size = (int(raw_width), int(raw_height))
        except ValueError as exc:
            raise ValueError(
                f"{path}: non-numeric original size {raw_width}x{raw_height}"
            ) from exc
        if size[0] > 0 and size[1] > 0:
            return size
    return None


def read_hand_xml(path: str | Path) -> HandDocument:
    """Read a CVAT hand export - images or video - into the canonical document.

    Both nestings are accepted: ``<image name=...><skeleton label=...><points .../></skeleton>``
    and ``<track label=...><skeleton frame=...><points .../></skeleton></track>``. A shape whose
    label is not ``hand_left``/``hand_right``/``hand_unknown`` is not a hand and is skipped.
    Two shapes for one ``(frame, side)`` always raise, even when their joints are disjoint: one
    side in one frame is one hand, so a second shape is a duplicate rather than a complement.
    """
    source = Path(path)
    root = ET.parse(source).getroot()
    shapes: list[HandShape] = []
    seen: set[tuple[int, str]] = set()
    grid = _declared_grid(root, source)
    images = root.findall("image")
    if images:
        for position, image in enumerate(images):
            frame = _image_frame(image.get("name"), image.get("id"), position)
            for skeleton in image.findall("skeleton"):
                side = _side(skeleton.get("label"))
                if side is None:
                    continue
                shape = _shape(skeleton, path=source, frame=frame, side=side)
                if shape is not None:
                    _add_shape(shapes, shape, path=source, seen=seen)
    elif root.find(".//track") is not None:
        for track in root.iter("track"):
            side = _side(track.get("label"))
            if side is None:
                continue
            for skeleton in track.findall("skeleton"):
                frame = _skeleton_frame(skeleton, path=source, side=side)
                shape = _shape(skeleton, path=source, frame=frame, side=side)
                if shape is not None:
                    _add_shape(shapes, shape, path=source, seen=seen)
    else:
        raise ValueError(
            f"{source}: not a CVAT annotation export (no <image> or <track> children)"
        )
    shapes.sort(key=lambda shape: (shape.frame, shape.side))
    return HandDocument(path=source, grid=grid, shapes=tuple(shapes))


def read_hand_json(path: str | Path) -> HandDocument:
    """Read a crinis corrections JSON into the canonical document.

    ``{"annotations": [{"kind": "points", "label": ..., "frame": ..., "keypoints": [[x, y], ...],
    "visibility": [...]}]}``. The point list is in :data:`JOINT_NAMES` order; a short list is a
    partial shape. A missing ``visibility`` list leaves every joint unknown. The format carries no
    image size, so the grid is ``None`` and the caller must pass an explicit scale.
    """
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{source}: unreadable corrections JSON: {exc}") from exc
    annotations = document.get("annotations") if isinstance(document, Mapping) else None
    if not isinstance(annotations, list):
        raise ValueError(f"{source}: no 'annotations' array; it is not a crinis corrections file")
    shapes: list[HandShape] = []
    seen: set[tuple[int, str]] = set()
    for entry in annotations:
        if not isinstance(entry, Mapping) or entry.get("kind") != "points":
            continue
        side = _side(str(entry.get("label") or ""))
        if side is None:
            continue
        frame = entry.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"{source}: annotation frame {frame!r} is not an integer")
        points = entry.get("keypoints")
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            raise ValueError(
                f"{source}: frame {frame} side {side!r} has no 'keypoints' array"
            )
        visibility = entry.get("visibility")
        joints: dict[str, HandJoint] = {}
        for index, point in enumerate(points):
            if index >= len(JOINT_NAMES):
                break
            joint = JOINT_NAMES[index]
            if (
                not isinstance(point, Sequence)
                or isinstance(point, (str, bytes))
                or len(point) < 2
            ):
                raise ValueError(
                    f"{source}: frame {frame} side {side!r} joint {joint!r} has no x,y pair"
                )
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source}: frame {frame} side {side!r} joint {joint!r} has non-numeric "
                    f"coordinates {point!r}"
                ) from exc
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(
                    f"{source}: frame {frame} side {side!r} joint {joint!r} has non-finite "
                    f"coordinates {point!r}"
                )
            declared: int | None = None
            if (
                isinstance(visibility, Sequence)
                and not isinstance(visibility, (str, bytes))
                and index < len(visibility)
            ):
                raw = visibility[index]
                if isinstance(raw, bool) or not isinstance(raw, int) or str(raw) not in VISIBILITY_VALUES:
                    raise ValueError(
                        f"{source}: frame {frame} side {side!r} joint {joint!r} has visibility "
                        f"{raw!r}, expected one of {VISIBILITY_VALUES}"
                    )
                declared = int(raw)
            joints[joint] = HandJoint(
                xy=(x, y), visibility=declared, outside=None, occluded=None
            )
        if not joints:
            continue
        _add_shape(
            shapes,
            HandShape(frame=frame, side=side, joints=joints),
            path=source,
            seen=seen,
        )
    shapes.sort(key=lambda shape: (shape.frame, shape.side))
    return HandDocument(path=source, grid=None, shapes=tuple(shapes))


def read_hands(path: str | Path) -> HandDocument:
    """Read either supported reference format, chosen by suffix (``.xml`` or ``.json``)."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".xml":
        return read_hand_xml(source)
    if suffix == ".json":
        return read_hand_json(source)
    raise ValueError(
        f"{source}: expected a CVAT .xml export or a crinis .json corrections file, got "
        f"{suffix or 'no suffix'!r}"
    )


def read_hands2d(
    path: str | Path,
) -> tuple[dict[tuple[int, str], dict[str, tuple[float, float]]], int, int]:
    """One ``hands2d`` artifact as ``(frame, side) -> joint`` maps, honouring ``side_known``.

    Returns ``(predictions, frames, unknown_side_predictions)``. The third number is the count of
    filled slots whose side the model did not establish - every ``side_known == False`` slot - and it
    is returned separately because the map can hold at most one ``(frame, "unknown")`` entry: an
    artifact can fill both slots with score guesses, and two unknown hands on one frame are not two
    sides to match against. Such slots are never keyed ``left``/``right``, so no caller can pair a
    guess with a labelled hand.

    An artifact with **no** ``side_known`` array - every run written before the identity slice - keeps
    the previous rule: the slot index is the side. That is what its readers already assumed, and
    reinterpreting old slots as unknown now would silently rewrite the history of every archived
    measurement.

    Empty slots (``conf`` absent or non-positive) are not predictions; a zero-filled slot is "no hand
    found", not a hand at the image origin.
    """
    from crinis.io.artifacts import load_artifact

    target = Path(path)
    if not target.is_file():
        raise ValueError(f"the hands2d artifact {target} does not exist")
    arrays = load_artifact(
        target,
        artifact="hands2d prediction",
        required={
            "kpts2d_px": "the predicted 21 pixel joints for both hand slots",
            "conf": "which hand slots the detector actually found",
        },
        stage="eval.evidence",
    )
    keypoints = np.asarray(arrays["kpts2d_px"], dtype=np.float64)
    confidence = np.asarray(arrays["conf"], dtype=np.float64)
    if keypoints.ndim != 4 or keypoints.shape[2:] != (len(JOINT_NAMES), 2):
        raise ValueError(
            f"kpts2d_px in {target} has shape {keypoints.shape}, expected "
            f"(T, {len(HAND_SLOTS)}, {len(JOINT_NAMES)}, 2)"
        )
    if confidence.shape != keypoints.shape[:2]:
        raise ValueError(
            f"conf in {target} has shape {confidence.shape}, expected {keypoints.shape[:2]}"
        )
    known: np.ndarray | None = None
    if "side_known" in arrays:
        known = np.asarray(arrays["side_known"])
        if known.shape != confidence.shape:
            raise ValueError(
                f"side_known in {target} has shape {known.shape}, expected {confidence.shape}; "
                "a provenance array that does not line up with the slots cannot be applied"
            )

    predictions: dict[tuple[int, str], dict[str, tuple[float, float]]] = {}
    unknown = 0
    for frame in range(int(keypoints.shape[0])):
        for slot, hand in enumerate(HAND_SLOTS):
            found = confidence[frame, slot]
            if not np.isfinite(found) or found <= 0.0:
                continue
            side = hand
            if known is not None and not bool(known[frame, slot]):
                side = "unknown"
                unknown += 1
            predictions[(frame, side)] = {
                joint: (float(x), float(y))
                for joint, (x, y) in zip(JOINT_NAMES, keypoints[frame, slot], strict=True)
            }
    return predictions, int(keypoints.shape[0]), unknown


def grid_scales(
    label_grid: tuple[int, int] | None, prediction_grid: tuple[int, int] | None
) -> tuple[float, float]:
    """Uniform x/y scale that moves label pixels onto the prediction grid.

    Anisotropic scaling is refused: two different factors turn one crop into another image, and a
    distance between them is a crop mismatch wearing a pixel error's clothes. A caller whose label
    file declares no grid must pass an explicit scale; guessing one would be the exact defect this
    function exists to stop.
    """
    if label_grid is None:
        raise ValueError(
            "the label document declares no image size, so no scale can be derived; pass an "
            "explicit uniform label scale instead of guessing one"
        )
    if prediction_grid is None:
        raise ValueError("the prediction declares no image size, so no scale can be derived")
    label_width, label_height = label_grid
    prediction_width, prediction_height = prediction_grid
    if min(label_width, label_height, prediction_width, prediction_height) <= 0:
        raise ValueError(
            f"declared grids {label_grid[0]}x{label_grid[1]} and "
            f"{prediction_grid[0]}x{prediction_grid[1]} are not positive"
        )
    scale_x = prediction_width / label_width
    scale_y = prediction_height / label_height
    if not (math.isfinite(scale_x) and math.isfinite(scale_y)):
        raise ValueError(
            f"label grid {label_width}x{label_height} and prediction grid "
            f"{prediction_width}x{prediction_height} produce non-finite scales"
        )
    if abs(scale_x / scale_y - 1.0) > 1e-9:
        raise ValueError(
            f"label grid {label_width}x{label_height} and prediction grid "
            f"{prediction_width}x{prediction_height} need independent scales x={scale_x:.6f} "
            f"y={scale_y:.6f}; that is a crop, not a resize. Resample the labels to a uniform grid "
            "or pass explicit uniform scales."
        )
    return scale_x, scale_y


# ── Scoring ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JointError:
    """One scored joint: the reference position and the prediction's distance from it."""

    frame: int
    side: str
    joint: str
    error_px: float
    fraction: float | None


@dataclass(frozen=True)
class EvidenceCounts:
    """Everything the score had to leave out, counted rather than dropped."""

    n_frames: int
    n_hands: int
    n_joints: int
    missing_hands: int
    extra_hands: int
    unknown_side_predictions: int
    unknown_visibility_joints: int
    unscored_frames: int


@dataclass(frozen=True)
class Scored:
    """The primary rows, the all-accepted sensitivity rows, and the counts."""

    rows: tuple[JointError, ...]
    all_accepted: tuple[JointError, ...]
    counts: EvidenceCounts


def hand_extent(joints: Mapping[str, tuple[float, float]]) -> float:
    """The largest distance between any two placed joints: the hand's pixel size."""
    values = list(joints.values())
    extent = 0.0
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            distance = math.hypot(first[0] - second[0], first[1] - second[1])
            if distance > extent:
                extent = distance
    return extent


def score_hands(
    predictions: Predictions,
    labels: HandDocument,
    *,
    scale_x: float,
    scale_y: float,
) -> Scored:
    """Score *predictions* against *labels* on the rescaled label grid.

    ``predictions`` is keyed by ``(source frame, side)`` and must already have dropped empty
    detector slots - a slot the model did not fill is not a prediction at the image origin.
    """
    if not (math.isfinite(scale_x) and math.isfinite(scale_y)):
        raise ValueError(f"scales must be finite, got x={scale_x!r} y={scale_y!r}")
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError(f"scales must be positive, got x={scale_x!r} y={scale_y!r}")
    primary: list[JointError] = []
    all_accepted: list[JointError] = []
    label_frames: set[int] = set()
    label_keys: set[tuple[int, str]] = set()
    scored_frames: set[int] = set()
    missing_hands = 0
    unknown_visibility = 0
    for shape in labels.shapes:
        label_frames.add(shape.frame)
        label_keys.add((shape.frame, shape.side))
        unknown_visibility += sum(
            1 for joint in shape.joints.values() if joint.visibility is None
        )
        scaled = {
            joint: (value.xy[0] * scale_x, value.xy[1] * scale_y)
            for joint, value in shape.joints.items()
        }
        extent = hand_extent(scaled)
        prediction = predictions.get((shape.frame, shape.side))
        if prediction is None:
            # A hand the run did not find is a miss. Scoring its joints as zero error would turn
            # that miss into a perfect prediction.
            missing_hands += 1
            continue
        for joint in JOINT_NAMES:
            if joint not in shape.joints:
                continue
            point = prediction.get(joint)
            if point is None:
                continue
            predicted_x, predicted_y = point
            if not (math.isfinite(predicted_x) and math.isfinite(predicted_y)):
                continue
            truth_x, truth_y = scaled[joint]
            error = float(math.hypot(predicted_x - truth_x, predicted_y - truth_y))
            fraction = None if extent <= 0.0 else error / extent
            row = JointError(
                frame=shape.frame, side=shape.side, joint=joint, error_px=error, fraction=fraction
            )
            all_accepted.append(row)
            if shape.joints[joint].visibility in (0, 1):
                continue
            primary.append(row)
            scored_frames.add(shape.frame)
    extra_hands = 0
    unknown_side = 0
    for frame, side in predictions:
        if frame not in label_frames:
            continue
        if side == "unknown":
            unknown_side += 1
        if (frame, side) not in label_keys:
            extra_hands += 1
    counts = EvidenceCounts(
        n_frames=len(label_frames),
        n_hands=len(label_keys),
        n_joints=sum(len(shape.joints) for shape in labels.shapes),
        missing_hands=missing_hands,
        extra_hands=extra_hands,
        unknown_side_predictions=unknown_side,
        unknown_visibility_joints=unknown_visibility,
        unscored_frames=len(label_frames - scored_frames),
    )
    return Scored(rows=tuple(primary), all_accepted=tuple(all_accepted), counts=counts)


# ── Summaries ────────────────────────────────────────────────────────────────


def _percentile_90(values: NDArray[np.float64]) -> float:
    return float(np.percentile(values, 90))


def _hand_summary(rows: Sequence[JointError], side: str) -> dict[str, Any]:
    selected = [row for row in rows if row.side == side]
    errors = np.asarray([row.error_px for row in selected], dtype=np.float64)
    summary: dict[str, Any] = {
        "n_joints": int(errors.size),
        "n_hands": len({(row.frame, row.side) for row in selected}),
        "n_frames": len({row.frame for row in selected}),
        "median_px": None,
        "mean_px": None,
        "p90_px": None,
        "max_px": None,
    }
    if errors.size:
        summary.update(
            {
                "median_px": float(np.median(errors)),
                "mean_px": float(errors.mean()),
                "p90_px": _percentile_90(errors),
                "max_px": float(errors.max()),
            }
        )
    return summary


def summarise_rows(rows: Sequence[JointError]) -> dict[str, Any]:
    """Build one metric object from scored rows; empty input is all-null with no reason yet."""
    errors = np.asarray([row.error_px for row in rows], dtype=np.float64)
    fractions = np.asarray(
        [row.fraction for row in rows if row.fraction is not None], dtype=np.float64
    )
    coverage: dict[str, float | None] = {f"{threshold:g}": None for threshold in COVERAGE_THRESHOLDS}
    if fractions.size:
        for threshold in COVERAGE_THRESHOLDS:
            coverage[f"{threshold:g}"] = float(np.mean(fractions <= threshold))
    summary: dict[str, Any] = {
        "n_joints": int(errors.size),
        "n_hands": len({(row.frame, row.side) for row in rows}),
        "n_frames": len({row.frame for row in rows}),
        "median_px": None,
        "mean_px": None,
        "p90_px": None,
        "max_px": None,
        "coverage": coverage,
        "by_hand": {side: _hand_summary(rows, side) for side in SIDES},
        "absent_reason": None,
    }
    if errors.size:
        summary.update(
            {
                "median_px": float(np.median(errors)),
                "mean_px": float(errors.mean()),
                "p90_px": _percentile_90(errors),
                "max_px": float(errors.max()),
            }
        )
    return summary


def absent_metric(reason: str) -> dict[str, Any]:
    """The all-null metric object a document carries when nothing could be measured."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("an absent metric needs a non-empty reason")
    summary = summarise_rows(())
    summary["absent_reason"] = reason
    return summary


def document_metrics(
    rows: Sequence[JointError],
    all_accepted: Sequence[JointError],
    *,
    absent_reason: str,
) -> dict[str, dict[str, Any]]:
    """The two metric objects every hand-evidence document carries.

    The primary metric is absent (with a reason) when no observable joint was predicted; the
    all-accepted sensitivity is absent when no placed joint had a prediction at all.
    """
    metrics = {
        METRIC_HAND: summarise_rows(rows) if rows else absent_metric(absent_reason),
    }
    if all_accepted:
        metrics[METRIC_HAND_ALL_ACCEPTED] = summarise_rows(all_accepted)
    else:
        metrics[METRIC_HAND_ALL_ACCEPTED] = absent_metric(
            f"{absent_reason} (all-accepted sensitivity)"
        )
    return metrics


# ── Uncertainty ──────────────────────────────────────────────────────────────


def _statistic_value(values: NDArray[np.float64], statistic: str) -> float:
    if statistic == "median_px":
        return float(np.median(values))
    if statistic == "p90_px":
        return float(np.percentile(values, 90))
    raise ValueError(f"unknown statistic {statistic!r}; expected one of {PAIRED_STATISTICS}")


def _frame_clusters(rows: Sequence[JointError]) -> list[NDArray[np.float64]]:
    """One array per source frame: a draw takes every joint of the frames it selects."""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.frame, []).append(row.error_px)
    return [
        np.asarray(grouped[frame], dtype=np.float64) for frame in sorted(grouped)
    ]


def cluster_bootstrap_median(
    rows: Sequence[JointError],
    *,
    statistic: str = "median_px",
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    """Percentile 95% interval of *statistic*, resampling **source frames** with replacement.

    Frames are the unit because joints within one frame are correlated (a wrist error moves the
    whole finger chain, and both hands share lighting and blur). Resampling joints would understate
    the interval; resampling hands would still split one frame's shared error.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    clusters = _frame_clusters(rows)
    if not clusters:
        raise ValueError("the bootstrap needs at least one scored row")
    rng = np.random.default_rng(seed)
    count = len(clusters)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picked = rng.integers(0, count, size=count)
        draws[index] = _statistic_value(
            np.concatenate([clusters[choice] for choice in picked]), statistic
        )
    low, high = np.percentile(draws, _PERCENTILES)
    return float(low), float(high)


def paired_delta(
    rows_baseline: Sequence[JointError],
    rows_candidate: Sequence[JointError],
    *,
    scope: str,
    statistic: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Paired ``candidate - baseline`` delta for one scope, with a frame-cluster interval.

    The caller intersects the two row sets on the exact ``(frame, side, joint)`` key first, so a key
    missing or invalid on either side leaves both sides identically - never imputed. A negative
    ``delta_px`` means the candidate is closer to the reference.
    """
    if statistic not in PAIRED_STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}; expected one of {PAIRED_STATISTICS}")
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    if len(rows_baseline) != len(rows_candidate):
        raise ValueError(
            f"{scope}: paired rows must be aligned, got {len(rows_baseline)} baseline and "
            f"{len(rows_candidate)} candidate rows; intersect on (frame, side, joint) first"
        )
    if not rows_baseline:
        raise ValueError(f"{scope}: no paired rows to compare")
    baseline_by_frame: dict[int, list[float]] = {}
    candidate_by_frame: dict[int, list[float]] = {}
    for baseline, candidate in zip(rows_baseline, rows_candidate, strict=True):
        if (baseline.frame, baseline.side, baseline.joint) != (
            candidate.frame,
            candidate.side,
            candidate.joint,
        ):
            raise ValueError(
                f"{scope}: rows are not aligned at frame {baseline.frame} {baseline.side} "
                f"{baseline.joint}; intersect on (frame, side, joint) first"
            )
        baseline_by_frame.setdefault(baseline.frame, []).append(baseline.error_px)
        candidate_by_frame.setdefault(candidate.frame, []).append(candidate.error_px)
    frames = sorted(baseline_by_frame)
    baseline_values = np.asarray(
        [value for frame in frames for value in baseline_by_frame[frame]], dtype=np.float64
    )
    candidate_values = np.asarray(
        [value for frame in frames for value in candidate_by_frame[frame]], dtype=np.float64
    )
    baseline_statistic = _statistic_value(baseline_values, statistic)
    candidate_statistic = _statistic_value(candidate_values, statistic)
    rng = np.random.default_rng(seed)
    count = len(frames)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picked = rng.integers(0, count, size=count)
        baseline_draw = np.concatenate(
            [np.asarray(baseline_by_frame[frames[choice]], dtype=np.float64) for choice in picked]
        )
        candidate_draw = np.concatenate(
            [np.asarray(candidate_by_frame[frames[choice]], dtype=np.float64) for choice in picked]
        )
        draws[index] = _statistic_value(candidate_draw, statistic) - _statistic_value(
            baseline_draw, statistic
        )
    low, high = np.percentile(draws, _PERCENTILES)
    return {
        "metric": statistic,
        "baseline": baseline_statistic,
        "candidate": candidate_statistic,
        "delta_px": candidate_statistic - baseline_statistic,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": len(rows_baseline),
        "n_frames": count,
    }


# ── Serialisation ────────────────────────────────────────────────────────────


def utc_now() -> str:
    """The document's ``created_utc`` stamp: seconds, UTC, no offset suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_jsonable(value: object) -> Any:  # noqa: ANN401 - recursive over arbitrary JSON-shaped values
    """Round every float to six decimals and refuse non-finite numbers on the way out."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"refusing to serialise non-finite number {number!r}")
        return round(number, 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"refusing to serialise {type(value).__name__} into a hand-evidence document")


# ── Validation ───────────────────────────────────────────────────────────────


def _fail(message: str) -> NoReturn:
    raise ValueError(f"{SCHEMA}: {message}")


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object, got {type(value).__name__}")
    return cast("Mapping[str, Any]", value)


def _sequence(value: object, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{where} must be an array, got {type(value).__name__}")
    return value


def _count(value: object, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{where} must be an integer >= {minimum}, got {value!r}")
    return value


def _number(value: object, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where} must be a number or null, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{where} is not finite")
    return number


def _hash(value: object, where: str, *, allow_null: bool = False) -> str | None:
    if value is None and allow_null:
        return None
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        _fail(f"{where} must be 64 lowercase hex characters, got {value!r}")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where} must be a non-empty string, got {value!r}")
    return value


def _path(value: object, where: str) -> str:
    text = _text(value, where)
    if "\x00" in text:
        _fail(f"{where} contains a NUL byte")
    if ".." in Path(text).parts:
        _fail(f"{where} escapes with a '..' segment: {text!r}")
    return text


def _exact_keys(value: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    present = set(value)
    expected = set(keys)
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        _fail(f"{where} keys are wrong (missing {missing}, unexpected {extra})")


def _validate_metric(name: str, value: object) -> None:
    metric = _mapping(value, f"metrics[{name!r}]")
    _exact_keys(metric, _METRIC_KEYS, f"metrics[{name!r}]")
    for key in ("n_joints", "n_hands", "n_frames"):
        _count(metric[key], f"metrics[{name!r}].{key}")
    numbers = [metric[key] for key in ("median_px", "mean_px", "p90_px", "max_px")]
    for key in ("median_px", "mean_px", "p90_px", "max_px"):
        _number(metric[key], f"metrics[{name!r}].{key}")
    coverage = _mapping(metric["coverage"], f"metrics[{name!r}].coverage")
    _exact_keys(coverage, [f"{threshold:g}" for threshold in COVERAGE_THRESHOLDS],
                f"metrics[{name!r}].coverage")
    for key in coverage:
        _number(coverage[key], f"metrics[{name!r}].coverage[{key!r}]")
    by_hand = _mapping(metric["by_hand"], f"metrics[{name!r}].by_hand")
    _exact_keys(by_hand, SIDES, f"metrics[{name!r}].by_hand")
    for side in SIDES:
        summary = _mapping(by_hand[side], f"metrics[{name!r}].by_hand[{side!r}]")
        _exact_keys(summary, _HAND_KEYS, f"metrics[{name!r}].by_hand[{side!r}]")
        for key in ("n_joints", "n_hands", "n_frames"):
            _count(summary[key], f"metrics[{name!r}].by_hand[{side!r}].{key}")
        for key in ("median_px", "mean_px", "p90_px", "max_px"):
            _number(summary[key], f"metrics[{name!r}].by_hand[{side!r}].{key}")
    reason = metric["absent_reason"]
    if reason is not None:
        _text(reason, f"metrics[{name!r}].absent_reason")
    if all(number is None for number in numbers) and reason is None:
        _fail(
            f"metrics[{name!r}] has no numbers and no absent_reason; a metric that cannot be "
            "computed must say why"
        )


def _validate_paired(value: object) -> None:
    paired = _mapping(value, "paired_delta")
    if not paired:
        _fail("paired_delta must carry at least one scope")
    for scope, entry in paired.items():
        if scope not in PAIRED_SCOPES:
            _fail(f"paired_delta scope {scope!r} is not one of {sorted(PAIRED_SCOPES)}")
        item = _mapping(entry, f"paired_delta[{scope!r}]")
        _exact_keys(item, _PAIRED_KEYS, f"paired_delta[{scope!r}]")
        metric = item["metric"]
        if metric not in PAIRED_STATISTICS:
            _fail(f"paired_delta[{scope!r}].metric {metric!r} is not one of {PAIRED_STATISTICS}")
        if metric != PAIRED_SCOPES[scope]:
            _fail(
                f"paired_delta[{scope!r}].metric is {metric!r}; the scope name requires "
                f"{PAIRED_SCOPES[scope]!r}"
            )
        baseline = _number(item["baseline"], f"paired_delta[{scope!r}].baseline")
        candidate = _number(item["candidate"], f"paired_delta[{scope!r}].candidate")
        delta = _number(item["delta_px"], f"paired_delta[{scope!r}].delta_px")
        low = _number(item["ci_low"], f"paired_delta[{scope!r}].ci_low")
        high = _number(item["ci_high"], f"paired_delta[{scope!r}].ci_high")
        if baseline is None or candidate is None or delta is None or low is None or high is None:
            _fail(f"paired_delta[{scope!r}] must carry finite numbers for every value")
        if abs(delta - (candidate - baseline)) > 1e-6:
            _fail(
                f"paired_delta[{scope!r}].delta_px {delta!r} is not candidate - baseline "
                f"({candidate!r} - {baseline!r})"
            )
        if low > high:
            _fail(f"paired_delta[{scope!r}] has ci_low {low!r} above ci_high {high!r}")
        _count(item["n_pairs"], f"paired_delta[{scope!r}].n_pairs", minimum=1)
        _count(item["n_frames"], f"paired_delta[{scope!r}].n_frames", minimum=1)


def _reject_nonfinite(value: object, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        if isinstance(value, Mapping):
            for key, item in value.items():
                _reject_nonfinite(item, f"{where}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                _reject_nonfinite(item, f"{where}[{index}]")
        return
    if not math.isfinite(float(value)):
        _fail(f"{where} is not finite")


def validate_evidence(document: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless *document* is a complete, internally consistent evidence object.

    Validation is total: unknown keys, hashes that are not 64 lowercase hex, negative counts, a
    metric missing a key (or carrying one more), an all-null metric with no reason, a ``by_hand``
    outside left/right/unknown, a non-finite number anywhere, a path escaping with ``..`` or a NUL,
    a ``grid`` with a non-positive scale, and a paired delta that does not equal
    ``candidate - baseline`` are all refused rather than written and interpreted later.
    """
    if not isinstance(document, Mapping):
        _fail(f"the document must be an object, got {type(document).__name__}")
    present = set(document)
    missing = [key for key in _REQUIRED_KEYS if key not in present]
    if missing:
        _fail(f"missing required keys {missing}")
    unexpected = sorted(present - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unexpected:
        _fail(f"unexpected keys {unexpected}")
    if document["schema"] != SCHEMA:
        _fail(f"schema must be {SCHEMA!r}, got {document['schema']!r}")
    if document["kind"] not in KINDS:
        _fail(f"kind {document['kind']!r} is not one of {KINDS}")
    _hash(document["source_take_sha256"], "source_take_sha256")
    _hash(document["run_sha256"], "run_sha256")
    artifacts = _sequence(document["prediction_artifacts"], "prediction_artifacts")
    for index, entry in enumerate(artifacts):
        artifact = _mapping(entry, f"prediction_artifacts[{index}]")
        _exact_keys(artifact, _ARTIFACT_KEYS, f"prediction_artifacts[{index}]")
        _path(artifact["path"], f"prediction_artifacts[{index}].path")
        _hash(artifact["sha256"], f"prediction_artifacts[{index}].sha256")
    references = _sequence(document["references"], "references")
    for index, entry in enumerate(references):
        reference = _mapping(entry, f"references[{index}]")
        _exact_keys(reference, _REFERENCE_KEYS, f"references[{index}]")
        _path(reference["path"], f"references[{index}].path")
        _hash(reference["sha256"], f"references[{index}].sha256")
        _text(reference["protocol"], f"references[{index}].protocol")
        _hash(reference["seed_sha256"], f"references[{index}].seed_sha256", allow_null=True)
        if reference["label_kind"] not in LABEL_KINDS:
            _fail(
                f"references[{index}].label_kind {reference['label_kind']!r} is not one of "
                f"{LABEL_KINDS}"
            )
    grid = _mapping(document["grid"], "grid")
    _exact_keys(grid, _GRID_KEYS, "grid")
    _count(grid["width"], "grid.width", minimum=1)
    _count(grid["height"], "grid.height", minimum=1)
    for key in ("scale_x", "scale_y"):
        scale = _number(grid[key], f"grid.{key}")
        if scale is None or scale <= 0.0:
            _fail(f"grid.{key} must be a positive number, got {grid[key]!r}")
    for key in (
        "n_frames",
        "n_hands",
        "n_joints",
        "missing_hands",
        "extra_hands",
        "unknown_side_predictions",
        "unknown_visibility_joints",
        "unscored_frames",
    ):
        _count(document[key], key)
    metrics = _mapping(document["metrics"], "metrics")
    allowed = {METRIC_HAND, METRIC_HAND_ALL_ACCEPTED}
    unknown_metrics = sorted(set(metrics) - allowed)
    if unknown_metrics:
        _fail(f"unknown metric names {unknown_metrics}; expected a subset of {sorted(allowed)}")
    if METRIC_HAND not in metrics:
        _fail(f"metrics must include {METRIC_HAND!r}")
    for name, entry in metrics.items():
        _validate_metric(name, entry)
    limitations = _sequence(document["limitations"], "limitations")
    for index, entry in enumerate(limitations):
        _text(entry, f"limitations[{index}]")
    created = document["created_utc"]
    if not isinstance(created, str) or not _UTC_RE.match(created):
        _fail(f"created_utc must look like 'YYYY-MM-DDTHH:MM:SSZ', got {created!r}")
    try:
        datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{SCHEMA}: created_utc {created!r} is not a real timestamp") from exc
    if "paired_delta" in document:
        _validate_paired(document["paired_delta"])
    _reject_nonfinite(document, "document")


def evidence_document(
    *,
    kind: str,
    source_take_sha256: str,
    run_sha256: str,
    prediction_artifacts: Sequence[Mapping[str, str]],
    references: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
    counts: EvidenceCounts,
    metrics: Mapping[str, Mapping[str, Any]],
    limitations: Sequence[str],
    created_utc: str | None = None,
    paired: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble and validate a ``crinis.hand-evidence/1`` document.

    The document-level ``n_frames``/``n_hands``/``n_joints`` are the primary metric's own scored
    denominators, so the headline can never disagree with the statistic beneath it; what was *not*
    scored lives in the exclusion counts, which is why they are separate keys.
    """
    primary = metrics.get(METRIC_HAND)
    if not isinstance(primary, Mapping):
        raise ValueError(f"metrics must carry {METRIC_HAND!r} to build an evidence document")
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": kind,
        "source_take_sha256": source_take_sha256,
        "run_sha256": run_sha256,
        "prediction_artifacts": [dict(entry) for entry in prediction_artifacts],
        "references": [dict(entry) for entry in references],
        "grid": dict(grid),
        "n_frames": primary["n_frames"],
        "n_hands": primary["n_hands"],
        "n_joints": primary["n_joints"],
        "missing_hands": counts.missing_hands,
        "extra_hands": counts.extra_hands,
        "unknown_side_predictions": counts.unknown_side_predictions,
        "unknown_visibility_joints": counts.unknown_visibility_joints,
        "unscored_frames": counts.unscored_frames,
        "metrics": {name: dict(entry) for name, entry in metrics.items()},
        "limitations": list(limitations),
        "created_utc": created_utc or utc_now(),
    }
    if paired is not None:
        document["paired_delta"] = {scope: dict(entry) for scope, entry in paired.items()}
    validate_evidence(document)
    return document


def write_evidence(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Validate *document*, then write it as sorted, indented JSON with a trailing newline."""
    validate_evidence(document)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_jsonable(document), indent=1, sort_keys=True) + "\n"
    target.write_text(payload, encoding="utf-8")
    return target
