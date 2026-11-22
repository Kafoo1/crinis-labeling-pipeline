"""CVAT for video 1.1 export and correction ingest.

Export reads one episode of a LeRobot v3 dataset and writes a task package:

``annotations.xml``
    Per-frame object boxes (tracks keyed by ``observation.objects.track_id``) and hand
    keypoints (21 points per observed hand, from ``observation.hand.landmarks2d``).  Every
    track is ``source="auto"``: CVAT's own marker for a pre-annotation.
``manifest.json``
    The source video, the frame range, and the exact feature columns each annotation came from.

A dataset that lacks a required column fails with an ``ArtifactError`` naming that column.
Emitting an empty ``annotations.xml`` would tell a labeler "this clip has no objects", which is
a different and much more expensive statement than "this export cannot read the objects".

Skeleton images exports (``write_skeleton_xml``) carry a per-joint ``visibility`` attribute on every
``<points>`` child: 2 when the joint lies inside the frame, 1 when it does not. Ingest reads it back
into ``CvatAnnotation.visibility`` and leaves it ``None`` when the attribute is absent, so "unknown"
is never promoted to "visible" - the same discipline as provenance.

Ingest parses the (possibly hand-edited) XML into ``corrections.json``.  Provenance is
conservative by construction: a shape is ``human`` only when the file explicitly says so
(``source="manual"`` or a ``provenance`` attribute of ``human``).  Everything else -- a missing
``source``, ``source="auto"``, or an attribute saying ``pipeline`` -- stays ``pipeline``.  This
is the same discipline as the evaluator refusing to score unmatched predictions as valid.

``<tag>`` elements are read as ``kind="tag"`` annotations from both nestings CVAT produces: inside an
``<image>`` (the write path this repo's segment task uses, because CVAT 2.80.1's loader only parses a
tag while an image is open) and inside a ``<track>`` with a ``frame`` attribute.  ``outside="1"``, CVAT's
own interval terminator, is dropped rather than promoted to a boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from xml.etree import ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from crinis.errors import ArtifactError, InputError
from crinis.hand.keypoint_spec import JOINT_NAMES, NUM_JOINTS
from crinis.io.storage import atomic_write_bytes, atomic_write_json

#: An array whose dtype is not known statically (parquet columns carry their own).  Spelled as
#: ``NDArray[np.generic]`` rather than a bare ``np.ndarray`` so mypy strict has no missing
#: ``type-arg`` to report; the project already carries hundreds of those and this module adds none.
Array = NDArray[np.generic]

#: The CVAT XML version this module writes and expects.
CVAT_VERSION = "1.1"

#: Columns an export cannot work without, in the order the error names them.
OBJECT_COLUMNS: tuple[str, ...] = (
    "observation.objects.bbox",
    "observation.objects.count",
)
HAND_COLUMNS: tuple[str, ...] = (
    "observation.hand.landmarks2d",
    "observation.hand.observed",
)
CONTACT_COLUMN = "observation.contact.any"

#: Optional columns that enrich the annotations when present.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "observation.objects.track_id",
    "observation.objects.class_id",
    CONTACT_COLUMN,
    "annotation.description",
    "frame_index",
)

REQUIRED_COLUMNS: tuple[str, ...] = (*OBJECT_COLUMNS, *HAND_COLUMNS)

#: The egohand profile's 2-D joints: 2 hands x 21 points x (u, v), in image pixels.
EGOHAND_JOINTS2D = "observation.hand_joints_2d"
EGOHAND_JOINTS2D_WIDTH = 84

#: Columns the projected pre-annotation needs on top of the required set.
PROJECTED_COLUMNS: tuple[str, ...] = (
    "observation.hand.landmarks3d",
    "observation.camera.intrinsics",
)

#: Where a `points` shape's 21 joints come from. `landmarks2d` is what the export has always written;
#: `projected` uses our own 3D joints, which exist on frames the 2D lane misses (measured 5/150 on
#: black-gloved footage against 128 frames of 3D); `off` writes no hand shapes at all.
HANDS_FROM: tuple[str, ...] = ("landmarks2d", "projected", "off")

OBJECT_LABEL = "object"

#: The two anatomical hand slots, as CVAT label names, in artifact order. These stay exactly two: every
#: ``[2, ...]`` hand array in this project - 3-D joints, depth, contact, MANO - has one column per side,
#: and no 3-D code may learn a third slot from a 2-D label.
HAND_SIDES: tuple[str, str] = ("hand_left", "hand_right")

#: What a CVAT *export* may label a hand. ``hand_unknown`` exists because the detector writes a slot
#: whose side its own class did not establish (``hands.unlabelled_slots: score``), and forcing that
#: guess into ``hand_left``/``hand_right`` is how a wrong side became a training label. It is a label
#: and a review signal, never a third anatomical slot.
HAND_UNKNOWN_LABEL = "hand_unknown"
HAND_LABELS: tuple[str, str, str] = (*HAND_SIDES, HAND_UNKNOWN_LABEL)

#: Shape attributes every skeleton label declares, so a labeler can read - and correct - the provenance
#: the artifact carries instead of guessing it. ``source`` and ``assignment`` say where the record came
#: from; ``side_known`` says whether the label above can be trusted as an anatomy; ``track_id`` is the
#: association track; ``structural_status`` is the clip's own soundness verdict.
SOURCE_ATTRIBUTE = "source"
SOURCE_VALUES: tuple[str, str] = ("primary", "fallback")
SIDE_KNOWN_ATTRIBUTE = "side_known"
ASSIGNMENT_ATTRIBUTE = "assignment"
ASSIGNMENT_LABEL_VALUES: tuple[str, str, str] = (
    "model_side",
    "temporal_association",
    "score_slot",
)
TRACK_ID_ATTRIBUTE = "track_id"
STRUCTURAL_STATUS_ATTRIBUTE = "structural_status"
STRUCTURAL_STATUS_VALUES: tuple[str, str, str] = ("sound", "unsound", "unmeasurable")
STRUCTURAL_REASONS_ATTRIBUTE = "structural_reasons"

#: The shape attributes a skeleton may carry, as ``(name, input_type, default, values|None)``. Written
#: into the label declaration and, value by value, onto a shape that actually has one.
SKELETON_SHAPE_ATTRIBUTES: tuple[tuple[str, str, str, str | None], ...] = (
    (SOURCE_ATTRIBUTE, "select", "primary", " ".join(SOURCE_VALUES)),
    (SIDE_KNOWN_ATTRIBUTE, "checkbox", "False", "False True"),
    (ASSIGNMENT_ATTRIBUTE, "select", "score_slot", " ".join(ASSIGNMENT_LABEL_VALUES)),
    (TRACK_ID_ATTRIBUTE, "text", "-1", None),
    (STRUCTURAL_STATUS_ATTRIBUTE, "select", "unmeasurable", " ".join(STRUCTURAL_STATUS_VALUES)),
    (STRUCTURAL_REASONS_ATTRIBUTE, "text", "", None),
)


#: Per-hand gate verdict written on every pre-annotated hand shape, and how many of its signals
#: fired. `not_evaluated` is what a shape says when the gate could not run, so a labeler never reads
#: "no data" as "no_signal". The gate itself lives in `crinis.eval.hand_gate`; this module only
#: writes its answer, which is why the vocabulary is duplicated as strings and not imported (the
#: gate imports this module's projection, so a module-level import back would be a cycle).
GATE_ATTRIBUTE = "gate"
GATE_SIGNALS_ATTRIBUTE = "gate_signals"
GATE_VALUES: tuple[str, ...] = ("no_signal", "flagged", "not_evaluated")
GATE_NOT_EVALUATED = "not_evaluated"
#: MediaPipe's 21-joint hand topology pairs, used only to describe the points label.
HAND_JOINTS = 21

#: Per-joint visibility, written by skeleton exports and read back by ``parse_cvat``. 2 means the
#: joint lies inside the frame, 1 means it does not (the labeler cannot see it). 0 is accepted on
#: ingest as CVAT's own "not visible" value but is never generated from geometry.
VISIBILITY_ATTRIBUTE = "visibility"
VISIBILITY_VALUES: tuple[str, ...] = ("0", "1", "2")
VISIBILITY_INSIDE = 2
VISIBILITY_OUTSIDE = 1

_COLORS = {
    OBJECT_LABEL: "#ff7f0e",
    "hand_left": "#1f77b4",
    "hand_right": "#d62728",
    # Grey on purpose: an unknown-side hand is neither of the two sides, and a colour borrowed from
    # one of them would read as a classification the model never made.
    HAND_UNKNOWN_LABEL: "#7f7f7f",
}

_VIDEO_KEY = "observation.images.head"
_DEFAULT_VIDEO_TEMPLATE = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_SHARD_RE = re.compile(r"chunk-(\d+)[/\\]file-(\d+)\.parquet$")
#: The last integer run in an images-export ``<image name="frame_0007.png">``.
_FRAME_DIGITS_RE = re.compile(r"\d+")


# ── Public result types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CvatExport:
    """What an export produced, in counts a caller can print without re-reading the XML."""

    annotations: Path
    manifest: Path
    episode_index: int | None
    frames: int
    boxes: int
    keypoints: int
    tracks: int
    source_video: str | None


@dataclass(frozen=True)
class CvatAnnotation:
    """One box or keypoint shape from a CVAT XML document."""

    kind: str  # "box" | "points" | "tag"
    label: str
    frame: int
    track_id: int
    source: str  # "pipeline" | "human"
    attributes: Mapping[str, str] = field(default_factory=dict)
    box: tuple[float, float, float, float] | None = None
    keypoints: tuple[tuple[float, float], ...] = ()
    #: One value per joint, in ``keypoints`` order: 2 inside the frame, 1 outside it. ``None``
    #: when the shape carried no visibility attribute at all - absence is never promoted to a guess.
    visibility: tuple[int, ...] | None = None

    def to_dict(self, *, task: str, reviewer: str | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "task": task,
            "reviewer": reviewer,
            "kind": self.kind,
            "label": self.label,
            "frame": self.frame,
            "track_id": self.track_id,
            "source": self.source,
            "attributes": dict(self.attributes),
        }
        if self.box is not None:
            payload["box"] = [round(value, 6) for value in self.box]
        if self.keypoints:
            payload["keypoints"] = [
                [round(x, 6), round(y, 6)] for x, y in self.keypoints
            ]
        if self.visibility is not None:
            payload["visibility"] = list(self.visibility)
        return payload


@dataclass(frozen=True)
class CvatDocument:
    """A parsed CVAT XML document."""

    path: Path
    name: str
    size: int
    labels: tuple[str, ...]
    annotations: tuple[CvatAnnotation, ...]
    reviewer: str | None


@dataclass(frozen=True)
class CvatCorrections:
    """The written corrections file and its provenance split."""

    path: Path
    task: str
    reviewer: str | None
    annotations: int
    pipeline: int
    human: int


# ── Dataset reading ──────────────────────────────────────────────────────────


def _artifact_error(message: str) -> ArtifactError:
    return ArtifactError("CRN-2001", message)


def _read_info(root: Path) -> dict[str, object]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise _artifact_error(
            f"dataset {root} has no meta/info.json; it is not a LeRobot v3 dataset"
        )
    try:
        raw = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _artifact_error(f"dataset {root} has an unreadable meta/info.json: {exc}") from exc
    if not isinstance(raw, dict):
        raise _artifact_error(f"dataset {root} meta/info.json is not a JSON object")
    return cast(dict[str, object], raw)


def _shards(root: Path) -> list[Path]:
    shards = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not shards:
        raise _artifact_error(f"dataset {root} has no data shards under data/chunk-*/*.parquet")
    return shards


def _columns_from_schema(shard: Path) -> tuple[str, ...]:
    try:
        return tuple(pq.read_schema(shard).names)
    except OSError as exc:
        raise _artifact_error(f"cannot read the parquet schema of {shard}: {exc}") from exc


def _episode_indices(shard: Path) -> tuple[int, ...]:
    names = _columns_from_schema(shard)
    if "episode_index" not in names:
        return (0,)
    column = pq.read_table(shard, columns=["episode_index"]).column("episode_index")
    values = {int(value) for value in column.to_pylist() if value is not None}
    return tuple(sorted(values)) or (0,)


def _select_episode(root: Path, episode: int | None) -> tuple[int, Path]:
    available: dict[int, Path] = {}
    for shard in _shards(root):
        for index in _episode_indices(shard):
            available.setdefault(index, shard)
    if not available:
        raise _artifact_error(f"dataset {root} contains no episodes")
    if episode is None:
        if len(available) != 1:
            raise _artifact_error(
                f"dataset {root} contains episodes {sorted(available)}; pass --episode to choose one"
            )
        return next(iter(available.items()))
    if episode not in available:
        raise _artifact_error(
            f"dataset {root} has no episode {episode}; available: {sorted(available)}"
        )
    return episode, available[episode]


def _derive_canonical_columns(columns: dict[str, Array]) -> dict[str, Array]:
    """Fill the canonical columns an egohand table carries in its own form.

    Two differences, both structural rather than missing data:

    * ``observation.hand.landmarks2d`` -- the egohand profile stores the same points flat, as
      ``(T, 84)`` = 2 hands x 21 points x (u, v), under :data:`EGOHAND_JOINTS2D`.
    * the object columns -- the lane's ``objects_pose_lane`` is ``off`` there, so the table has no
      object columns at all. A dataset whose hands are present and whose objects are absent says
      "no objects in this footage", which is a real annotation, not an empty file.

    A table that carries neither form still fails the required-column check.
    """
    if "observation.hand.landmarks2d" not in columns:
        flat = columns.get(EGOHAND_JOINTS2D)
        if flat is not None:
            array = np.asarray(flat)
            if array.ndim == 2 and array.shape[1] == EGOHAND_JOINTS2D_WIDTH:
                columns["observation.hand.landmarks2d"] = array.reshape(len(array), 2, 21, 2)
    if "observation.objects.bbox" not in columns and "observation.hand.observed" in columns:
        frames = len(np.asarray(columns["observation.hand.observed"]))
        columns.setdefault(
            "observation.objects.bbox", np.zeros((frames, 0, 4), dtype=np.float64)
        )
        columns.setdefault("observation.objects.count", np.zeros(frames, dtype=np.int64))
    return columns


def _read_shard(shard: Path, columns: Sequence[str]) -> dict[str, Array]:
    on_disk = set(_columns_from_schema(shard))
    present = set(on_disk)
    # The egohand form supplies the canonical columns structurally; see _derive_canonical_columns.
    egohand = EGOHAND_JOINTS2D in on_disk
    if egohand:
        present.add("observation.hand.landmarks2d")
        present.update(OBJECT_COLUMNS)
    for name in REQUIRED_COLUMNS:
        if name not in present:
            raise _artifact_error(
                f"dataset column {name!r} is missing from {shard}; the CVAT export needs it "
                f"and will not emit an empty annotation file in its place"
            )
    wanted = [name for name in columns if name in on_disk]
    if egohand and EGOHAND_JOINTS2D not in wanted:
        wanted.append(EGOHAND_JOINTS2D)
    table = pq.read_table(shard, columns=wanted)
    derived = {name: np.asarray(table.column(name).to_pylist()) for name in table.column_names}
    return _derive_canonical_columns(derived)


def _filter_episode(
    columns: Mapping[str, Array], episode: int
) -> dict[str, Array]:
    raw = columns.get("episode_index")
    if raw is None:
        return dict(columns)
    mask = np.asarray(raw).reshape(-1) == episode
    return {name: np.asarray(values)[mask] for name, values in columns.items()}


def _task_text(root: Path, columns: Mapping[str, Array]) -> str | None:
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            table = pq.read_table(tasks_path, columns=["task"])
        except (OSError, KeyError):
            table = None
        if table is not None:
            values = table.column("task").to_pylist()
            if values:
                return str(values[0])
    description = columns.get("annotation.description")
    if description is not None and len(description):
        text = str(np.asarray(description).reshape(-1)[0])
        if text:
            return text
    return None


def _video_relpath(root: Path, shard: Path, info: Mapping[str, object]) -> str | None:
    match = _SHARD_RE.search(shard.as_posix())
    if match is None:
        return None
    template = str(info.get("video_path") or _DEFAULT_VIDEO_TEMPLATE)
    try:
        relative = template.format(
            video_key=_VIDEO_KEY,
            chunk_index=int(match.group(1)),
            file_index=int(match.group(2)),
        )
    except (KeyError, IndexError, ValueError):
        return None
    return relative


def _count_column(values: Array) -> Array:
    return np.asarray(values).reshape(-1).astype(np.int64)


def _rows(values: Array, index: int) -> Array:
    return cast(Array, np.asarray(values)[index])


def _box_is_valid(box: Array) -> bool:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        return False
    xtl, ytl, xbr, ybr = (float(value) for value in values)
    return xbr > xtl and ybr > ytl


def _keypoints_are_valid(points: Array) -> bool:
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (HAND_JOINTS, 2) or not np.all(np.isfinite(values)):
        return False
    span = max(
        float(values[:, 0].max() - values[:, 0].min()),
        float(values[:, 1].max() - values[:, 1].min()),
    )
    return span > 0.5


# ── XML writing ──────────────────────────────────────────────────────────────


def _project_joints(joints: Array, intrinsics: Array) -> Array | None:
    """Project 21 camera-frame joints to pixels, or ``None`` when they cannot be projected.

    ``observation.hand.landmarks3d`` is stored in the **camera** frame - measured at 14.88 px median
    agreement with the 2D landmarks, against 71.38 px when the stored pose is applied - so the
    intrinsics alone place it on the image. A joint behind the camera, or a non-finite one, makes the
    whole hand unprojectable: half a skeleton is worse than none, because a labeler cannot tell which
    half to trust.
    """
    values = np.asarray(joints, dtype=np.float64).reshape(HAND_JOINTS, 3)
    if not np.isfinite(values).all():
        return None
    fx, fy, cx, cy = (float(v) for v in np.asarray(intrinsics, dtype=np.float64).reshape(-1)[:4])
    z = values[:, 2]
    if (z <= 1e-6).any():
        return None
    return np.stack([fx * values[:, 0] / z + cx, fy * values[:, 1] / z + cy], axis=1)


def _sub(
    parent: ET.Element, tag: str, text: str | None = None, **attributes: object
) -> ET.Element:
    element = ET.SubElement(
        parent, tag, {key: str(value) for key, value in attributes.items()}
    )
    if text is not None:
        element.text = text
    return element


def _attribute_definition(
    parent: ET.Element,
    *,
    name: str,
    input_type: str,
    default: str,
    values: str | None = None,
    mutable: bool = False,
) -> None:
    attribute = _sub(parent, "attribute")
    _sub(attribute, "name", name)
    _sub(attribute, "mutable", "True" if mutable else "False")
    _sub(attribute, "input_type", input_type)
    _sub(attribute, "default_value", default)
    if values is not None:
        _sub(attribute, "values", values)


def _label_definition(
    labels: ET.Element,
    *,
    name: str,
    shape_type: str,
    include_track_id: bool,
    include_class_id: bool,
    include_contact: bool,
    include_gate: bool = False,
) -> None:
    label = _sub(labels, "label")
    _sub(label, "name", name)
    _sub(label, "color", _COLORS.get(name, "#808080"))
    _sub(label, "type", shape_type)
    attributes = _sub(label, "attributes")
    if include_track_id:
        _attribute_definition(
            attributes, name="track_id", input_type="number", default="-1"
        )
    if include_class_id:
        _attribute_definition(
            attributes, name="class_id", input_type="number", default="-1"
        )
    if include_contact:
        _attribute_definition(
            attributes,
            name="contact",
            input_type="checkbox",
            default="False",
            values="False True",
        )
    if include_gate:
        _attribute_definition(
            attributes,
            name=GATE_ATTRIBUTE,
            input_type="select",
            default=GATE_NOT_EVALUATED,
            values=" ".join(GATE_VALUES),
        )
        _attribute_definition(
            attributes,
            name=GATE_SIGNALS_ATTRIBUTE,
            input_type="text",
            default="0",
        )
    _attribute_definition(
        attributes,
        name="provenance",
        input_type="select",
        default="pipeline",
        values="pipeline human",
    )


def _shape_attributes(parent: ET.Element, values: Mapping[str, str]) -> None:
    for name, value in values.items():
        attribute = _sub(parent, "attribute", value)
        attribute.set("name", name)


def _format_points(points: Array) -> str:
    return ";".join(f"{float(x):.3f},{float(y):.3f}" for x, y in np.asarray(points))


def _write_xml(
    *,
    out_path: Path,
    task_name: str,
    frames: int,
    object_tracks: Mapping[int, list[tuple[int, Array, int, bool | None]]],
    hand_tracks: Mapping[int, list[tuple[int, Array]]],
    has_contact: bool,
    has_track_ids: bool,
    hand_gates: Mapping[tuple[int, str], tuple[str, int]] | None = None,
) -> tuple[int, int]:
    """Write the XML and return ``(boxes, keypoints)``.

    ``hand_gates`` maps ``(frame, hand label)`` to ``(verdict, flagged signal count)``; when it is
    given, every hand shape carries both values and the label declares them. It is optional so the
    dataset export - which has no run artifacts to gate - writes exactly the document it always did.
    """
    root = ET.Element("annotations")
    _sub(root, "version", CVAT_VERSION)

    meta = _sub(root, "meta")
    task = _sub(meta, "task")
    _sub(task, "id", "0")
    _sub(task, "name", task_name)
    _sub(task, "size", str(frames))
    _sub(task, "mode", "annotation")
    _sub(task, "overlap", "0")
    _sub(task, "bugtracker", "")
    _sub(task, "created", datetime.now(timezone.utc).isoformat())
    _sub(task, "updated", datetime.now(timezone.utc).isoformat())
    _sub(task, "start_frame", "0")
    _sub(task, "stop_frame", str(max(frames - 1, 0)))
    _sub(task, "frame_filter", "")
    segments = _sub(task, "segments")
    segment = _sub(segments, "segment")
    _sub(segment, "id", "0")
    _sub(segment, "start", "0")
    _sub(segment, "stop", str(max(frames - 1, 0)))
    _sub(segment, "url", "")
    owner = _sub(task, "owner")
    _sub(owner, "username", "")
    _sub(owner, "email", "")
    labels = _sub(task, "labels")
    _label_definition(
        labels,
        name=OBJECT_LABEL,
        shape_type="rectangle",
        include_track_id=has_track_ids,
        include_class_id=True,
        include_contact=has_contact,
    )
    for name in HAND_SIDES:
        _label_definition(
            labels,
            name=name,
            shape_type="points",
            include_track_id=False,
            include_class_id=False,
            include_contact=False,
            include_gate=hand_gates is not None,
        )
    _sub(meta, "dumped", datetime.now(timezone.utc).isoformat())

    next_track_id = 0
    boxes = 0
    for track_key in sorted(object_tracks):
        shapes = object_tracks[track_key]
        track = _sub(
            root,
            "track",
            id=str(next_track_id),
            label=OBJECT_LABEL,
            source="auto",
        )
        next_track_id += 1
        for frame, box, class_id, contact in shapes:
            xtl, ytl, xbr, ybr = (float(value) for value in np.asarray(box).reshape(-1))
            element = _sub(
                track,
                "box",
                frame=str(frame),
                xtl=f"{xtl:.3f}",
                ytl=f"{ytl:.3f}",
                xbr=f"{xbr:.3f}",
                ybr=f"{ybr:.3f}",
                occluded="0",
                z_order="0",
                outside="0",
                keyframe="1",
            )
            attributes = {"provenance": "pipeline"}
            if has_track_ids:
                attributes["track_id"] = str(track_key)
            if class_id >= 0:
                attributes["class_id"] = str(class_id)
            if contact is not None:
                attributes["contact"] = "True" if contact else "False"
            _shape_attributes(element, attributes)
            boxes += 1

    keypoints = 0
    for side in sorted(hand_tracks):
        track = _sub(
            root,
            "track",
            id=str(next_track_id),
            label=HAND_SIDES[side],
            source="auto",
        )
        next_track_id += 1
        shapes = hand_tracks[side]
        for index, (frame, points) in enumerate(shapes):
            element = _sub(
                track,
                "points",
                frame=str(frame),
                keyframe="1",
                points=_format_points(points),
                occluded="0",
                z_order="0",
                outside="0",
            )
            attributes = {"provenance": "pipeline"}
            if hand_gates is not None:
                gate, flagged = hand_gates.get(
                    (frame, HAND_SIDES[side]), (GATE_NOT_EVALUATED, 0)
                )
                attributes[GATE_ATTRIBUTE] = gate
                attributes[GATE_SIGNALS_ATTRIBUTE] = str(flagged)
            _shape_attributes(element, attributes)
            keypoints += 1

            # Close the track unless the very next frame is also a detection. Without this a video
            # task *interpolates* between keyframes, and the detections here are sparse - on clip_0002
            # the right hand is found on 34 of 150 frames with gaps up to 52 - so CVAT drew a skeleton
            # gliding from one hand position to the next across every gap. That is what "the labels are
            # flying and unrelated to the video" was: nothing to do with the GPU, and nothing to do with
            # interpolation being wrong in principle - the export simply never said the hand was gone.
            following = shapes[index + 1][0] if index + 1 < len(shapes) else None
            if following != frame + 1 and frame + 1 < frames:
                _sub(
                    track,
                    "points",
                    frame=str(frame + 1),
                    keyframe="1",
                    points=_format_points(points),
                    occluded="0",
                    z_order="0",
                    outside="1",
                )

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="utf-8")
    atomic_write_bytes(out_path, b'<?xml version="1.0" encoding="utf-8"?>\n' + body + b"\n")
    return boxes, keypoints


def joint_visibility(
    points: Sequence[tuple[float, float]], width: int, height: int
) -> tuple[int, ...]:
    """One visibility per joint from geometry: 2 inside the frame, 1 outside it.

    A joint at ``width - 1`` is still inside; a non-finite coordinate is outside, because the
    labeler cannot click a position that does not exist. The frame size is a parameter rather than
    a constant: 1280x720 and 1920x1080 clips both exist, and a hardcoded size would mark a real
    1080p joint invisible (or an off-frame 720p joint visible).
    """
    values: list[int] = []
    for x, y in points:
        finite = bool(np.isfinite(x) and np.isfinite(y))
        inside = (
            finite
            and 0.0 <= float(x) <= float(width) - 1.0
            and 0.0 <= float(y) <= float(height) - 1.0
        )
        values.append(VISIBILITY_INSIDE if inside else VISIBILITY_OUTSIDE)
    return tuple(values)


#: The shape-level attribute names a skeleton may carry; a value not declared here is refused rather
#: than written into a document a labeler's CVAT would silently ignore.
_SKELETON_ATTRIBUTE_NAMES: frozenset[str] = frozenset(
    name for name, _input_type, _default, _values in SKELETON_SHAPE_ATTRIBUTES
)


@dataclass(frozen=True)
class SkeletonShape:
    """One hand skeleton on one frame, with the provenance a labeler needs to judge it.

    ``label`` is one of :data:`HAND_LABELS`; a shape whose side the model did not establish is
    ``hand_unknown`` and carries ``side_known=False`` rather than being written as the slot's guess.
    ``joints`` is 21 ``(x, y)`` in :data:`JOINT_NAMES` order. ``attributes`` are the shape-level
    provenance attributes (:data:`SKELETON_SHAPE_ATTRIBUTES`); absent keys are simply not written,
    never defaulted here, because the writer must not claim a value the caller did not supply.
    """

    label: str
    joints: tuple[tuple[float, float], ...]
    attributes: Mapping[str, str] = field(default_factory=dict)


def _skeleton_label_definition(labels: ET.Element, label_name: str) -> None:
    """Declare one hand label: its shape attributes and the 21 per-joint visibility sublabels."""
    label = _sub(labels, "label")
    _sub(label, "name", label_name)
    _sub(label, "color", _COLORS.get(label_name, "#808080"))
    _sub(label, "type", "skeleton")
    attributes = _sub(label, "attributes")
    for name, input_type, default, values in SKELETON_SHAPE_ATTRIBUTES:
        _attribute_definition(
            attributes, name=name, input_type=input_type, default=default, values=values
        )
    for joint in JOINT_NAMES:
        sublabel = _sub(label, "sublabel")
        _sub(sublabel, "name", joint)
        _sub(sublabel, "type", "points")
        point_attributes = _sub(sublabel, "attributes")
        _attribute_definition(
            point_attributes,
            name=VISIBILITY_ATTRIBUTE,
            input_type="select",
            default=str(VISIBILITY_INSIDE),
            values=" ".join(VISIBILITY_VALUES),
            mutable=True,
        )


def write_skeleton_xml(
    path: Path,
    *,
    task_name: str,
    order: Sequence[int],
    hands: Mapping[int, Sequence[SkeletonShape]],
    width: int,
    height: int,
    prefix: str = "",
) -> Path:
    """Write a CVAT-for-images skeleton document with per-joint visibility and shape provenance.

    ``hands`` maps a frame to a **sequence** of shapes, not to a mapping keyed by label: a frame can
    hold two hands whose side is unknown, and a dict keyed by label would silently overwrite one with
    the other - the exact ambiguity ``hand_unknown`` exists to preserve. Every label declares the
    per-joint ``visibility`` select and the six shape attributes; a shape's attribute is written only
    when the caller supplied it, so nothing is invented. The frame size is written on every
    ``<image>`` and used for the geometry, so the same function serves 720p and native 1080p tasks.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("annotations")
    _sub(root, "version", CVAT_VERSION)
    meta = _sub(root, "meta")
    task = _sub(meta, "task")
    _sub(task, "name", task_name)
    _sub(task, "size", str(len(order)))
    labels = _sub(task, "labels")
    for label_name in HAND_LABELS:
        _skeleton_label_definition(labels, label_name)

    for position, frame in enumerate(order):
        image = _sub(
            root,
            "image",
            id=str(position),
            name=f"{prefix}frame_{frame:04d}.png",
            width=str(width),
            height=str(height),
        )
        for shape in hands.get(frame, ()):
            if shape.label not in HAND_LABELS:
                raise InputError(
                    "CRN-1001",
                    f"frame {frame} declares label {shape.label!r}, expected one of {list(HAND_LABELS)}",
                )
            points = tuple((float(x), float(y)) for x, y in shape.joints)
            if len(points) != NUM_JOINTS:
                raise InputError(
                    "CRN-1001",
                    f"frame {frame} {shape.label} has {len(points)} joints, expected {NUM_JOINTS}",
                )
            unknown = [name for name in shape.attributes if name not in _SKELETON_ATTRIBUTE_NAMES]
            if unknown:
                raise InputError(
                    "CRN-1001",
                    f"frame {frame} {shape.label} carries undeclared shape attribute(s) "
                    f"{sorted(unknown)}; the label declares {sorted(_SKELETON_ATTRIBUTE_NAMES)}",
                )
            skeleton = _sub(image, "skeleton", label=shape.label, keyframe="1", z_order="0")
            _shape_attributes(skeleton, shape.attributes)
            visibility = joint_visibility(points, width, height)
            for joint, (x, y), visible in zip(JOINT_NAMES, points, visibility, strict=True):
                element = _sub(
                    skeleton,
                    "points",
                    label=joint,
                    keyframe="1",
                    occluded="0",
                    z_order="0",
                    outside="0",
                    points=f"{x:.3f},{y:.3f}",
                )
                _sub(element, "attribute", str(visible), name=VISIBILITY_ATTRIBUTE)

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="utf-8")
    atomic_write_bytes(path, b'<?xml version="1.0" encoding="utf-8"?>\n' + body + b"\n")
    return path


def skeleton_label_payload(svg: str) -> list[dict[str, object]]:
    """The CVAT API label payload for the hand skeletons, with visibility and shape provenance.

    Same labels and same attribute spec as :func:`write_skeleton_xml`, including
    ``hand_unknown``: the task-creation call and the annotation import then agree about what a joint
    value means and about what an unlabelled hand is called.
    """
    visibility: dict[str, object] = {
        "name": VISIBILITY_ATTRIBUTE,
        "input_type": "select",
        "values": list(VISIBILITY_VALUES),
        "default_value": str(VISIBILITY_INSIDE),
        "mutable": True,
    }
    shape_attributes: list[dict[str, object]] = []
    for name, input_type, default, values in SKELETON_SHAPE_ATTRIBUTES:
        attribute: dict[str, object] = {
            "name": name,
            "input_type": input_type,
            "default_value": default,
        }
        if values is not None:
            attribute["values"] = values.split(" ")
        shape_attributes.append(attribute)
    return [
        {
            "name": label_name,
            "type": "skeleton",
            "svg": svg,
            "attributes": [dict(attribute) for attribute in shape_attributes],
            "sublabels": [
                {"name": joint, "type": "points", "attributes": [dict(visibility)]}
                for joint in JOINT_NAMES
            ],
        }
        for label_name in HAND_LABELS
    ]


# ── Export ───────────────────────────────────────────────────────────────────


def export_cvat(
    dataset: Path,
    out_dir: Path,
    *,
    episode: int | None = None,
    hands_from: str = "landmarks2d",
) -> CvatExport:
    """Write a CVAT-for-video task package for one episode of *dataset*.

    Raises ``ArtifactError("CRN-2001")`` when the dataset is not a readable LeRobot v3
    dataset, when the requested episode is absent/ambiguous, or when a required column is
    missing; the message always names the offending path or column.
    """
    dataset = Path(dataset)
    out_dir = Path(out_dir)
    if hands_from not in HANDS_FROM:
        raise InputError(
            "CRN-1001",
            f"hands_from must be one of {list(HANDS_FROM)}, got {hands_from!r}",
        )
    if not dataset.exists():
        raise _artifact_error(f"dataset {dataset} does not exist")

    info = _read_info(dataset)
    episode_index, shard = _select_episode(dataset, episode)
    wanted = [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "episode_index"]
    if hands_from == "projected":
        wanted.extend(PROJECTED_COLUMNS)
    columns = _read_shard(shard, (*wanted,))
    columns = _filter_episode(columns, episode_index)

    frames = len(_count_column(columns["observation.objects.count"]))
    if frames < 1:
        raise _artifact_error(
            f"dataset {dataset} episode {episode_index} contains no frames"
        )
    frame_count = len(_count_column(columns.get("frame_index", np.arange(frames))))
    if frame_count != frames:
        raise _artifact_error(
            f"dataset {dataset} episode {episode_index} has {frames} count rows but "
            f"{frame_count} frame_index rows"
        )

    bbox = np.asarray(columns["observation.objects.bbox"])
    counts = _count_column(columns["observation.objects.count"])
    observed = np.asarray(columns["observation.hand.observed"]).reshape(frames, 2, -1)
    landmarks = np.asarray(columns["observation.hand.landmarks2d"])
    joints3d = (
        np.asarray(columns["observation.hand.landmarks3d"])
        if hands_from == "projected"
        else None
    )
    intrinsics = (
        np.asarray(columns["observation.camera.intrinsics"])
        if hands_from == "projected"
        else None
    )
    pre_skipped = 0

    track_ids = columns.get("observation.objects.track_id")
    class_ids = columns.get("observation.objects.class_id")
    contact = columns.get("observation.contact.any")

    object_tracks: dict[int, list[tuple[int, Array, int, bool | None]]] = {}
    hand_tracks: dict[int, list[tuple[int, Array]]] = {}
    frames_with_boxes = 0
    frames_with_keypoints = 0

    for frame in range(frames):
        present = int(max(0, min(int(counts[frame]), bbox.shape[1])))
        wrote_box = False
        for slot in range(present):
            box = _rows(bbox, frame)[slot]
            if not _box_is_valid(box):
                continue
            track_key = slot
            if track_ids is not None:
                raw_track = int(np.asarray(track_ids).reshape(frames, -1)[frame, slot])
                if raw_track >= 0:
                    track_key = raw_track
            class_id = -1
            if class_ids is not None:
                class_id = int(np.asarray(class_ids).reshape(frames, -1)[frame, slot])
            contact_value: bool | None = None
            if contact is not None:
                contact_value = bool(
                    np.asarray(contact).reshape(frames, -1)[frame, slot]
                )
            object_tracks.setdefault(track_key, []).append(
                (frame, np.asarray(box).copy(), class_id, contact_value)
            )
            wrote_box = True
        frames_with_boxes += int(wrote_box)

        for side in range(2):
            if not bool(observed[frame, side, 0]) or hands_from == "off":
                continue
            if hands_from == "projected":
                points = _project_joints(
                    _rows(np.asarray(joints3d), frame)[side],
                    np.asarray(intrinsics[frame], dtype=np.float64),
                )
                if points is None:
                    pre_skipped += 1
                    continue
            else:
                points = _rows(np.asarray(landmarks), frame)[side]
            if not _keypoints_are_valid(points):
                pre_skipped += 1
                continue
            hand_tracks.setdefault(side, []).append((frame, np.asarray(points).copy()))
            frames_with_keypoints += 1

    task = _task_text(dataset, columns)
    video_rel = _video_relpath(dataset, shard, info)
    source_video = dataset / video_rel if video_rel is not None else None

    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "annotations.xml"
    boxes, keypoints = _write_xml(
        out_path=annotations_path,
        task_name=task or f"{dataset.name} episode {episode_index}",
        frames=frames,
        object_tracks=object_tracks,
        hand_tracks=hand_tracks,
        has_contact=contact is not None,
        has_track_ids=track_ids is not None,
    )

    manifest: dict[str, object] = {
        "format": "cvat",
        "format_version": CVAT_VERSION,
        "dataset": str(dataset),
        "episode_index": episode_index,
        "task": task,
        "fps": info.get("fps"),
        "frames": frames,
        "frame_range": [0, max(frames - 1, 0)],
        "source_video": video_rel,
        "source_video_exists": bool(source_video is not None and source_video.exists()),
        "annotations": "annotations.xml",
        "annotation_sources": {
            "objects": {
                "columns": [
                    "observation.objects.bbox",
                    "observation.objects.count",
                    *(
                        ["observation.objects.track_id"]
                        if track_ids is not None
                        else []
                    ),
                    *(
                        ["observation.objects.class_id"]
                        if class_ids is not None
                        else []
                    ),
                ],
                "boxes": boxes,
                "tracks": len(object_tracks),
                "frames_with_boxes": frames_with_boxes,
            },
            "hands": {
                "columns": [
                    *(
                        ["observation.hand.landmarks3d", "observation.camera.intrinsics"]
                        if hands_from == "projected"
                        else ["observation.hand.landmarks2d"]
                    ),
                    "observation.hand.observed",
                ],
                "source": hands_from,
                "pre_annotations_skipped": pre_skipped,
                "keypoints": keypoints,
                "tracks": len(hand_tracks),
                "frames_with_keypoints": frames_with_keypoints,
                "joints_per_hand": HAND_JOINTS,
            },
            "contact": {
                "columns": [CONTACT_COLUMN] if contact is not None else [],
                "used": contact is not None,
            },
        },
        "labels": {
            OBJECT_LABEL: "rectangle",
            **dict.fromkeys(HAND_SIDES, "points"),
        },
        "provenance": {
            "default": "pipeline",
            "human_marker": "CVAT track/shape source=manual, or attribute provenance=human",
            "reviewer_source": "meta/task/owner/username",
            "rule": (
                "ingest records pipeline unless the XML explicitly marks a shape human; "
                "a missing source is never promoted"
            ),
        },
    }
    manifest_path = out_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    return CvatExport(
        annotations=annotations_path,
        manifest=manifest_path,
        episode_index=episode_index,
        frames=frames,
        boxes=boxes,
        keypoints=keypoints,
        tracks=len(object_tracks) + len(hand_tracks),
        source_video=video_rel,
    )


# ── Ingest ───────────────────────────────────────────────────────────────────


def _xml_error(path: Path, message: str) -> InputError:
    return InputError(
        "CRN-1001",
        f"CVAT XML {path}: {message}",
        remediation=(
            "Export the corrections from CVAT as 'CVAT for images 1.1' or 'CVAT for video 1.1' "
            "XML and check the file is the edited copy of the exported annotations.xml."
        ),
    )


def _resolve_source(*markers: str | None) -> str:
    """Return ``human`` only for an explicit human marker; everything else is ``pipeline``."""
    for marker in markers:
        if marker is None:
            continue
        if marker.strip().lower() in {"manual", "human"}:
            return "human"
    return "pipeline"


def _shape_attributes_from_xml(element: ET.Element) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for attribute in element.findall("attribute"):
        name = attribute.get("name")
        if name:
            attributes[name] = (attribute.text or "").strip()
    return attributes


def _parse_box(element: ET.Element) -> tuple[float, float, float, float] | None:
    try:
        return (
            float(element.get("xtl", "nan")),
            float(element.get("ytl", "nan")),
            float(element.get("xbr", "nan")),
            float(element.get("ybr", "nan")),
        )
    except ValueError:
        return None


def _parse_points(element: ET.Element) -> tuple[tuple[float, float], ...]:
    raw = (element.get("points") or "").strip()
    if not raw:
        return ()
    points: list[tuple[float, float]] = []
    for pair in raw.split(";"):
        coordinates = pair.split(",")
        if len(coordinates) != 2:
            return ()
        try:
            points.append((float(coordinates[0]), float(coordinates[1])))
        except ValueError:
            return ()
    return tuple(points)


def _skeleton_issue(element: ET.Element) -> str | None:
    """Why *element* is not a complete one-point-per-joint skeleton, or ``None`` when it is."""
    seen: set[str] = set()
    for child in element.findall("points"):
        name = (child.get("label") or "").strip()
        if name not in JOINT_NAMES:
            return f"unknown joint name {name!r}"
        if name in seen:
            return f"duplicate joint {name!r}"
        if len(_parse_points(child)) != 1:
            return f"joint {name!r} has no single 'x,y' point"
        seen.add(name)
    missing = [name for name in JOINT_NAMES if name not in seen]
    if missing:
        return (
            f"{len(seen)} of {NUM_JOINTS} joints, missing {', '.join(missing[:3])}"
            f"{', ...' if len(missing) > 3 else ''}"
        )
    return None


def _parse_skeleton(element: ET.Element) -> tuple[tuple[float, float], ...] | None:
    """Return a skeleton's joints in ``JOINT_NAMES`` order, or ``None`` when it is malformed."""
    if _skeleton_issue(element) is not None:
        return None
    by_name = {
        (child.get("label") or "").strip(): _parse_points(child)[0]
        for child in element.findall("points")
    }
    return tuple(by_name[name] for name in JOINT_NAMES)


def _parse_skeleton_visibility(element: ET.Element) -> tuple[int, ...] | None:
    """Per-joint visibility in ``JOINT_NAMES`` order, or ``None`` when any joint lacks one.

    The attribute lives on each joint's ``<points>`` child. A shape with no visibility attributes
    yields ``None``; a partial or malformed set also yields ``None`` rather than filling the gaps
    with a guess, because "unknown" and "visible" are different statements.
    """
    by_joint: dict[str, int] = {}
    for child in element.findall("points"):
        joint = (child.get("label") or "").strip()
        raw = ""
        for attribute in child.findall("attribute"):
            if (attribute.get("name") or "").strip() == VISIBILITY_ATTRIBUTE:
                raw = (attribute.text or "").strip()
                break
        if raw not in VISIBILITY_VALUES:
            return None
        by_joint[joint] = int(raw)
    if not all(name in by_joint for name in JOINT_NAMES):
        return None
    return tuple(by_joint[name] for name in JOINT_NAMES)


def _skeleton_is_absence(element: ET.Element) -> bool:
    """Whether a skeleton is an absence marker: every joint carries ``outside="1"``.

    CVAT writes ``outside="1"`` per joint on the ``<points>`` children; the ``<skeleton>`` element
    itself carries no ``outside``. A skeleton whose joints are all outside is a closed track /
    deleted hand, not an observation, and is dropped exactly like a shape-level ``outside="1"``.
    A skeleton with only *some* joints outside is partial visibility, not absence, and keeps
    flowing through the visibility path. Reads the same per-joint children as
    :func:`_parse_skeleton_visibility`; no second convention.
    """
    children = element.findall("points")
    if not children:
        return False
    return all(child.get("outside") == "1" for child in children)


def _image_frame(name: str | None, position: int) -> int:
    """The frame an images-export filename encodes, falling back to the element's position."""
    digits = _FRAME_DIGITS_RE.findall(name or "")
    if not digits:
        return position
    return int(digits[-1])


def _annotation_from_shape(
    element: ET.Element,
    *,
    label: str,
    track_id: int,
    track_source: str | None,
    frame: int | None = None,
) -> CvatAnnotation | None:
    if element.get("outside") == "1":
        return None
    attributes = _shape_attributes_from_xml(element)
    source = _resolve_source(
        track_source,
        element.get("source"),
        attributes.get("provenance"),
    )
    try:
        shape_frame = int(element.get("frame", "0")) if frame is None else frame
    except ValueError:
        return None
    if element.tag == "tag":
        # A tag is a boundary marker: CVAT stores it inside a track (``frame=``) or inside an
        # ``<image>`` (the frame comes from the filename). ``outside="1"`` is CVAT's own interval
        # terminator and is dropped above, so only the human-authored boundary survives.
        return CvatAnnotation(
            kind="tag",
            label=label,
            frame=shape_frame,
            track_id=track_id,
            source=source,
            attributes=attributes,
        )
    if element.tag == "box":
        box = _parse_box(element)
        if box is None:
            return None
        return CvatAnnotation(
            kind="box",
            label=label,
            frame=shape_frame,
            track_id=track_id,
            source=source,
            attributes=attributes,
            box=box,
        )
    if element.tag == "points":
        points = _parse_points(element)
        if not points:
            return None
        return CvatAnnotation(
            kind="points",
            label=label,
            frame=shape_frame,
            track_id=track_id,
            source=source,
            attributes=attributes,
            keypoints=points,
        )
    if element.tag == "skeleton":
        if _skeleton_is_absence(element):
            return None
        keypoints = _parse_skeleton(element)
        if keypoints is None:
            return None
        # CVAT marks an edited joint, not the skeleton, so a manual child promotes the shape.
        source = _resolve_source(
            source,
            *(child.get("source") for child in element.findall("points")),
        )
        return CvatAnnotation(
            kind="points",
            label=label,
            frame=shape_frame,
            track_id=track_id,
            source=source,
            attributes=attributes,
            keypoints=keypoints,
            visibility=_parse_skeleton_visibility(element),
        )
    return None


def _collect_shape(
    element: ET.Element,
    *,
    label: str,
    track_id: int,
    track_source: str | None,
    frame: int | None,
    problems: list[str],
) -> CvatAnnotation | None:
    """Build one annotation, recording a malformed skeleton instead of dropping it."""
    if element.tag == "skeleton":
        issue = _skeleton_issue(element)
        if issue is not None:
            shape_frame = frame
            if shape_frame is None:
                try:
                    shape_frame = int(element.get("frame", "0"))
                except ValueError:
                    shape_frame = 0
            problems.append(f"{label or '<unlabeled>'} at frame {shape_frame}: {issue}")
            return None
    return _annotation_from_shape(
        element,
        label=label,
        track_id=track_id,
        track_source=track_source,
        frame=frame,
    )


def parse_cvat(path: Path) -> CvatDocument:
    """Parse a CVAT-for-video XML document into annotations with explicit provenance."""
    path = Path(path)
    if not path.exists():
        raise _artifact_error(f"CVAT XML {path} does not exist")
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        raise _xml_error(path, f"is not valid XML: {exc}") from exc
    root = tree.getroot()
    if root.tag != "annotations":
        raise _xml_error(path, f"root element is {root.tag!r}, expected 'annotations'")

    task = root.find("meta/task")
    name = _text_of(task, "name") or path.stem
    try:
        size = int(_text_of(task, "size") or "0")
    except ValueError:
        size = 0
    reviewer = _text_of(task, "owner/username") or None
    labels = tuple(
        label_name
        for label in root.findall("meta/task/labels/label")
        if (label_name := (label.findtext("name") or "").strip())
    )

    annotations: list[CvatAnnotation] = []
    problems: list[str] = []
    for track in root.findall("track"):
        label = track.get("label", "")
        track_source = track.get("source")
        try:
            track_id = int(track.get("id", "-1"))
        except ValueError:
            track_id = -1
        for element in track:
            annotation = _collect_shape(
                element,
                label=label,
                track_id=track_id,
                track_source=track_source,
                frame=None,
                problems=problems,
            )
            if annotation is not None:
                annotations.append(annotation)

    # CVAT for images 1.1 nests shapes under <image name="frame_0007.png">; take the frame from
    # the filename so an images export lands on the same frames a video export would.
    for position, image in enumerate(root.findall("image")):
        image_frame = _image_frame(image.get("name"), position)
        for element in image:
            if element.tag not in {"box", "points", "skeleton", "tag"}:
                continue
            annotation = _collect_shape(
                element,
                label=element.get("label", ""),
                track_id=-1,
                track_source=None,
                frame=image_frame,
                problems=problems,
            )
            if annotation is not None:
                annotations.append(annotation)

    # Image-less exports keep shapes directly under <annotations>.
    for element in root:
        if element.tag not in {"box", "points", "tag"}:
            continue
        annotation = _collect_shape(
            element,
            label=element.get("label", ""),
            track_id=-1,
            track_source=None,
            frame=None,
            problems=problems,
        )
        if annotation is not None:
            annotations.append(annotation)

    if problems:
        more = len(problems) - 5
        detail = "; ".join(problems[:5])
        if more > 0:
            detail = f"{detail}; and {more} more"
        raise _xml_error(path, f"has {len(problems)} malformed skeleton shape(s): {detail}")

    annotations.sort(key=lambda item: (item.frame, item.label, item.track_id))
    return CvatDocument(
        path=path,
        name=name,
        size=size,
        labels=labels,
        annotations=tuple(annotations),
        reviewer=reviewer,
    )


def _text_of(parent: ET.Element | None, path: str) -> str:
    if parent is None:
        return ""
    return (parent.findtext(path) or "").strip()


def _read_manifest(tasks_dir: Path) -> dict[str, object]:
    manifest_path = tasks_dir / "manifest.json"
    if not manifest_path.exists():
        raise _artifact_error(
            f"labeling package {tasks_dir} has no manifest.json; run `crinis label export` first"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _artifact_error(f"labeling package {manifest_path} is unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise _artifact_error(f"labeling package {manifest_path} is not a JSON object")
    return cast(dict[str, object], raw)


def ingest_cvat(
    tasks_dir: Path, corrections: Path, *, out: Path | None = None
) -> CvatCorrections:
    """Parse a reviewed CVAT XML and write ``corrections.json`` with per-shape provenance.

    ``corrections`` may be the unedited export; every annotation then says ``pipeline``.
    A labeler's edit becomes ``human`` only through CVAT's ``source="manual"`` or an explicit
    ``provenance=human`` attribute -- absence of evidence is never promoted.
    """
    tasks_dir = Path(tasks_dir)
    corrections = Path(corrections)
    manifest = _read_manifest(tasks_dir)
    document = parse_cvat(corrections)

    task = str(manifest.get("task") or document.name or "unlabeled")
    reviewer = document.reviewer
    annotations = [
        annotation.to_dict(task=task, reviewer=reviewer)
        for annotation in document.annotations
    ]
    pipeline = sum(1 for a in document.annotations if a.source == "pipeline")
    human = sum(1 for a in document.annotations if a.source == "human")

    payload: dict[str, object] = {
        "format": "crinis-corrections",
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "dataset": manifest.get("dataset"),
        "episode_index": manifest.get("episode_index"),
        "source_video": manifest.get("source_video"),
        "reviewer": reviewer,
        "source_xml": str(corrections),
        "annotations": annotations,
        "totals": {
            "annotations": len(annotations),
            "pipeline": pipeline,
            "human": human,
        },
    }
    out_path = Path(out) if out is not None else tasks_dir / "corrections.json"
    atomic_write_json(out_path, payload)
    return CvatCorrections(
        path=out_path,
        task=task,
        reviewer=reviewer,
        annotations=len(annotations),
        pipeline=pipeline,
        human=human,
    )


#: Artifact layout of a run, relative to its root.
RUN_ARTIFACTS = {
    "hands3d": "artifacts/hands3d/hands_world.npz",
    "hands2d": "artifacts/hands2d/hands2d.npz",
    "camera": "artifacts/camera/camera_pose.npz",
    "objects": "artifacts/objects/objects.npz",
    "frames": "artifacts/ingest/frames/frames.mp4",
    "index": "artifacts/ingest/frames/frames_index.parquet",
}


def _gate_attributes(
    run_dir: Path,
) -> tuple[dict[tuple[int, str], tuple[str, int]], dict[str, object]]:
    """Per-shape gate values and the manifest block, from the run's own artifacts.

    The import is local because ``crinis.eval.hand_gate`` imports :func:`_project_joints` from this
    module, so a module-level import here would close the cycle. The gate is additive: an export
    must still happen when it cannot run, so a failure downgrades every shape to ``not_evaluated``
    and records the reason in the manifest instead of aborting the task.
    """
    from crinis.eval.hand_gate import gate_run, summary

    try:
        verdicts = gate_run(run_dir)
    except Exception as error:  # the gate is additive; the export itself is the product
        return {}, {"evaluated": False, "reason": f"{type(error).__name__}: {error}"}
    attributes = {
        (verdict.frame, verdict.side): (verdict.state, len(verdict.flags))
        for verdict in verdicts
    }
    counts = cast(dict[str, dict[str, int]], summary(verdicts)["counts"])
    return attributes, {"evaluated": True, **counts}



def _intrinsics_without_trajectory(run_dir: Path) -> tuple[Array, str]:
    """Intrinsics for projecting hands when the run has no solved camera trajectory.

    Projecting a 3-D hand needs the intrinsics and nothing else - the pose never enters this projection -
    and ``hands3d`` computed its metres against exactly one focal, which it records. Requiring
    ``camera_pose.npz`` made a labelling task impossible on any machine that cannot run the camera stage
    (no VGGT checkout, no HAWoR SLAM dump), which blocked labelling of every such clip. Both numbers come
    from artifacts that already exist; the caller records in the manifest that these are *configured*
    intrinsics rather than a solved trajectory, so nothing downstream mistakes them for one.
    """
    meta_path = run_dir / (RUN_ARTIFACTS["hands3d"] + ".meta.json")
    try:
        counters = json.loads(meta_path.read_text(encoding="utf-8")).get("counters", {})
    except (OSError, ValueError) as exc:
        raise InputError(
            "CRN-1001",
            f"{run_dir} has no camera trajectory and no readable hands3d metadata ({exc}); either run "
            f"the camera stage or re-run hands3d so its focal is recorded",
        ) from exc
    focal = counters.get("focal_px") or counters.get("focal_px_frames")
    if not isinstance(focal, (int, float)) or float(focal) <= 0.0:
        raise InputError(
            "CRN-1001",
            f"{meta_path} does not record a usable focal_px ({focal!r}); run the camera stage instead",
        )

    index_path = run_dir / RUN_ARTIFACTS["index"]
    try:
        raw = (pq.read_schema(index_path).metadata or {})[b"crinis.output_size"]
        width, height = (int(v) for v in json.loads(raw.decode()))
    except Exception as exc:  # an unreadable index means "run the camera stage"
        raise InputError(
            "CRN-1001",
            f"{index_path} does not record the frame size ({exc}); run the camera stage instead",
        ) from exc
    intrinsics = np.asarray(
        [float(focal), float(focal), width / 2.0, height / 2.0], dtype=np.float64
    )
    note = (
        f"configured intrinsics: focal {float(focal):.3f} px at {width}x{height}, principal point at the "
        f"image centre, from hands3d focal_px; no solved camera trajectory in this run"
    )
    return intrinsics, note


def export_cvat_from_run(
    run_dir: Path,
    out_dir: Path,
    *,
    hands_from: str = "projected",
    task_name: str | None = None,
) -> CvatExport:
    """Write a CVAT task from a run's artifacts, without requiring a dataset.

    The dataset path exists for work that is already shippable; the labelling loop needs the opposite
    order - label first, ship later - and `export-lerobot` is not always runnable (measured: the
    current `lerobot` needs two pyav APIs that no Python 3.13 wheel provides). The artifacts a task
    needs already exist after `hands2d`, `hands3d` and `camera`, so this reads them directly.

    ``hands_from="projected"`` is the default here, not ``landmarks2d``: the reason to export from a
    run at all is usually that the 2D lane missed hands, and the 3D lane is what covers them.

    Every hand shape also carries the label-free gate's verdict (``crinis.eval.hand_gate``) as a
    ``gate`` attribute plus a ``gate_signals`` count, so a labeler sees which hands are flagged
    without opening a report; the manifest counts them per side.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    if hands_from not in HANDS_FROM:
        raise InputError(
            "CRN-1001", f"hands_from must be one of {list(HANDS_FROM)}, got {hands_from!r}"
        )
    if not run_dir.is_dir():
        raise InputError("CRN-1001", f"run directory {run_dir} does not exist")

    hands3d_path = run_dir / RUN_ARTIFACTS["hands3d"]
    camera_path = run_dir / RUN_ARTIFACTS["camera"]
    if not hands3d_path.exists():
        raise InputError(
            "CRN-1001",
            f"{hands3d_path} does not exist; the hands3d stage must run before a task can be exported",
        )
    # The camera stage is optional for this path: the projection needs intrinsics only.

    with np.load(hands3d_path) as arrays:
        joints = np.asarray(arrays["joints_world_m"], dtype=np.float64)
        observed = np.asarray(arrays["observed"]).reshape(-1, 2).astype(bool)
    if camera_path.exists():
        with np.load(camera_path) as arrays:
            intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float64).reshape(-1)[:4]
        intrinsics_note = "intrinsics from the run's camera artifact"
    else:
        intrinsics, intrinsics_note = _intrinsics_without_trajectory(run_dir)

    frames = int(observed.shape[0])
    fps = 30
    index_path = run_dir / RUN_ARTIFACTS["index"]
    if index_path.exists():
        metadata = pq.read_schema(index_path).metadata or {}
        fps = int(metadata.get(b"crinis.fps", b"30") or 30)
        frames = min(frames, int(pq.read_metadata(index_path).num_rows))

    hand_tracks: dict[int, list[tuple[int, Array]]] = {}
    skipped = 0
    for frame in range(frames):
        for side in range(2):
            if not bool(observed[frame, side]) or hands_from == "off":
                continue
            points = _project_joints(joints[frame, side], intrinsics)
            if points is None:
                skipped += 1
                continue
            hand_tracks.setdefault(side, []).append((frame, np.asarray(points)))

    object_tracks: dict[int, list[tuple[int, Array, int, bool | None]]] = {}
    object_path = run_dir / RUN_ARTIFACTS["objects"]
    has_track_ids = False
    if object_path.exists():
        with np.load(object_path) as arrays:
            boxes = np.asarray(arrays["bbox"])
            counts = np.asarray(arrays["count"]).reshape(-1)
            track_ids = np.asarray(arrays["track_id"]).reshape(boxes.shape[0], -1)
            class_ids = np.asarray(arrays["class_id"]).reshape(boxes.shape[0], -1)
        has_track_ids = True
        for frame in range(min(frames, boxes.shape[0])):
            for slot in range(int(max(0, min(int(counts[frame]), boxes.shape[1])))):
                box = boxes[frame, slot]
                if not box.any():
                    continue
                track = int(track_ids[frame, slot])
                object_tracks.setdefault(track if track >= 0 else slot, []).append(
                    (frame, np.asarray(box).copy(), int(class_ids[frame, slot]), None)
                )

    name = task_name or f"{run_dir.name} - hands from {hands_from}"
    hand_gates, gate_block = _gate_attributes(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "annotations.xml"
    boxes_written, keypoints_written = _write_xml(
        out_path=annotations_path,
        task_name=name,
        frames=frames,
        object_tracks=object_tracks,
        hand_tracks=hand_tracks,
        has_contact=False,
        has_track_ids=has_track_ids,
        hand_gates=hand_gates,
    )
    video = run_dir / RUN_ARTIFACTS["frames"]
    manifest = {
        "format": "cvat",
        "format_version": CVAT_VERSION,
        "source": "run",
        "run": str(run_dir),
        "task": name,
        "fps": fps,
        "frames": frames,
        "frame_range": [0, max(frames - 1, 0)],
        "intrinsics_note": intrinsics_note,
        "source_video": str(video.relative_to(run_dir)) if video.exists() else None,
        "source_video_exists": video.exists(),
        "annotations": "annotations.xml",
        "annotation_sources": {
            "objects": {
                "columns": ["observation.objects.bbox", "observation.objects.count"]
                if object_path.exists()
                else [],
                "boxes": boxes_written,
                "tracks": len(object_tracks),
            },
            "hands": {
                "columns": ["observation.hand.landmarks3d", "observation.camera.intrinsics"]
                if hands_from == "projected"
                else ["observation.hand.landmarks2d"],
                "source": hands_from,
                "pre_annotations_skipped": skipped,
                "keypoints": keypoints_written,
                "tracks": len(hand_tracks),
                "frames_with_keypoints": len(
                    {frame for track in hand_tracks.values() for frame, _ in track}
                ),
                "joints_per_hand": HAND_JOINTS,
            },
        },
        "labels": {OBJECT_LABEL: "rectangle", **dict.fromkeys(HAND_SIDES, "points")},
        "gate": gate_block,
    }
    manifest_path = out_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    source_video_value = cast(str | None, manifest["source_video"])
    return CvatExport(
        annotations=annotations_path,
        manifest=manifest_path,
        episode_index=None,
        frames=frames,
        boxes=boxes_written,
        keypoints=keypoints_written,
        tracks=len(object_tracks) + len(hand_tracks),
        source_video=source_video_value,
    )
