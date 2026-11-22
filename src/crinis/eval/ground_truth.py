"""External ground truth for the eval lanes, with every absent modality named.

The pipeline's own accuracy numbers are internal-consistency checks until they are compared with
something outside the pipeline. This module loads that *something* and refuses to invent what it does
not contain. Two sources are implemented.

**``nvidia``** - the NVIDIA ``PhysicalAI-Robotics-Manipulation-Kitchen`` subset (task ``open_fridge``,
episode 0), whose measured properties are:

* **Simulation, not real footage.** It was generated in IsaacSim; every number derived from it is a
  simulated-accuracy number and must be labelled as such.
* **CC-BY-4.0**, so commercial use is permitted; the licence travels with the index.
* **RGB is a 512x512 h264 video.**
* **Depth is an 8-bit h264 video, not metres.** It is a lossy, normalised quantity and saturates at the
  8-bit maximum, so only its *ordering* is comparable with a metric prediction; see
  :func:`depth_rank_agreement`.
* **Semantic segmentation is an h264 colour-mask video** with no published colour-to-class map, so
  :func:`palette_labels` marks off-palette pixels explicitly instead of forcing them into a class.
* **``observation.state`` is the object-of-interest transform plus its articulated joint.**
  ``state[0:12]`` is the vectorised 3x4 transform of the "object of interest" and ``state[12]`` is that
  articulated object's joint value (the dataset README says so verbatim). One articulated object per
  episode is described; whether the transform is the body or the moving part is not stated per frame,
  and the translation units are unverified, so :func:`pose_residual` reports a **relative** residual.
* **No camera intrinsics/extrinsics, metric depth or per-frame success labels are published.** Those
  modalities are returned as ``None`` plus a reason and are never replaced by a proxy.

**``user_export``** - labels a human writes over their own footage, in
:data:`USER_LABEL_SCHEMA`. The label file names the clip and its SHA-256, so a label set can never be
scored silently against different footage. It supplies 2-D hand keypoints and 2-D object boxes and
nothing else: there is no 3-D reference, no metric depth and no object pose, and the scorers say so.
An absent label file is not an error: it is an empty reference whose every metric is null with the
reason ``user_labels_absent``.

The module imports numpy and the standard library only. Decoding needs a lazy ``av`` (video) or
``PIL`` (still image) import; when either is unavailable the accessor returns ``(None, reason)``
rather than raising, because an absent decoder is an absent modality to a caller, not a crash. Torch
is never imported here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crinis.errors import ConfigError, GeometryError, InputError

if TYPE_CHECKING:  # pragma: no cover - import only for the public signature
    from crinis.config.schema import Config

#: The NVIDIA index schema. A foreign index (another source's file, or a future version) is refused
#: by name instead of being read as if the fields meant the same thing. ``/2`` changed the meaning of
#: ``observation.state``: ``/1`` recorded it as a robot end-effector pose, which the dataset README
#: contradicts, so a ``/1`` document cannot be read with the corrected object-pose interpretation.
INDEX_SCHEMA = "crinis.ground-truth-index/2"

#: Index schemas whose field meanings differ from :data:`INDEX_SCHEMA`. They are refused by exact
#: name with an instruction to re-run the converter, never silently upgraded.
SUPERSEDED_INDEX_SCHEMAS: tuple[str, ...] = ("crinis.ground-truth-index/1",)

#: The human-fillable label schema. Its format is documented verbatim in
#: ``docs/ground_truth_intake.md``. ``/2`` adds the optional ``attestations`` array and the
#: hand-level ``provenance``/``prelabel_sha256`` tags: an explicit human attestation that a
#: pre-label was reviewed and deliberately left unchanged (provenance code 3).
USER_LABEL_SCHEMA = "crinis.user-labels/2"

#: User-label schemas whose field meanings differ from :data:`USER_LABEL_SCHEMA`. They are refused
#: by exact name with an instruction to re-run the converter, never silently upgraded. ``/1`` has
#: no attestation surface, so a ``/1`` document cannot express a reviewed-unchanged pre-label.
SUPERSEDED_USER_LABEL_SCHEMAS: tuple[str, ...] = ("crinis.user-labels/1",)

#: The only permitted ``attestations[].decision`` value. Absence is owned by
#: ``declared_absent_slots``; there is no second absence mechanism.
ATTESTATION_DECISION = "unchanged_correct"

#: Hand-record ``provenance`` tag for a human-edited shape (the default when the tag is absent).
EDITED_HAND_PROVENANCE = "human_corrected"

#: Hand-record ``provenance`` tag for a pre-label adopted verbatim under a human attestation.
ATTESTED_HAND_PROVENANCE = "human_attested_unchanged"

#: The sources ``evaluation.ground_truth.source`` may name. ``none`` is the default and means every
#: accuracy field stays null; the other two are real external references.
SOURCES: tuple[str, ...] = ("none", "nvidia", "user_export")

#: Every modality a reference can declare. Present ones have an accessor that returns data; missing
#: ones appear in ``missing_modalities`` with a reason, and their accessor returns ``(None, reason)``.
MODALITIES: tuple[str, ...] = (
    "rgb",
    "depth_normalised",
    "semantic_segmentation",
    "robot_state",
    "hands_2d",
    "hands_3d",
    "object_boxes_2d",
    "object_pose",
    "articulated_joint",
    "camera_intrinsics",
    "camera_extrinsics",
    "metric_depth",
    "success_labels",
)

#: The 21 joints the user-export contract orders keypoints by (wrist first, then each finger's
#: metacarpal-to-tip chain), matching the pipeline's ``kpts2d_px`` convention.
NUM_HAND_JOINTS = 21

#: The label :func:`palette_labels` assigns to a pixel farther than ``tolerance`` from every palette
#: entry. Negative so it can never collide with a palette index.
UNMATCHED_LABEL = -1

#: The reason :func:`pose_residual` reports when the caller supplies no pose pair. The NVIDIA
#: reference does publish an object-of-interest transform, so a missing *input* is the only
#: "no pose" case this function names.
POSE_INPUTS_ABSENT = "object_pose_pair_absent"

#: The reason an absent user label file reports for every metric.
USER_LABELS_ABSENT = "user_labels_absent"

#: The geometry error code for an input array that violates the measurement contract.
CONTRACT_CODE = "CRN-4001"

#: The geometry error code for a statistic that is undefined because the evidence is too small or
#: degenerate (fewer than 100 pixels, an empty mask/box union, or constant ranks).
UNDEFINED_CODE = "CRN-4003"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class GroundTruthReference:
    """A loaded external reference: identity, inventory, and per-frame records.

    Construct through :meth:`load`; the constructor takes the already-validated document so tests can
    build one without a file when they need to. ``source='none'`` is a real object with zero frames and
    every modality missing, which is what makes "accuracy stays null when the reference is absent"
    true by construction rather than by convention.
    """

    source: str
    index_path: Path | None
    dataset: Mapping[str, object]
    frames: int
    present_modalities: tuple[str, ...]
    missing_modalities: Mapping[str, str]
    records: Mapping[int, Mapping[str, object]]
    root: Path | None
    video_path: Path | None = None
    video_sha256: str | None = None
    label_sha256: str | None = None
    fps: float | None = None
    clip_frames: int | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, source: str, path: str | Path | None) -> GroundTruthReference:
        """Load the reference named by *source* from *path*.

        ``none`` ignores *path* and returns an empty reference whose missing-modality map names the
        reason. ``nvidia`` resolves *path* (directory or index file) to ``gt-index.json`` and refuses a
        missing, unreadable or foreign index with the exact path and expected schema in the message.
        ``user_export`` resolves *path* (directory or ``labels.json``) and returns an empty reference
        when no label file exists; a label file that exists is validated (schema, every field, video
        hash, frame range) and any failure is a named refusal.

        Raises:
            ConfigError: ``CRN-0001`` when *source* is not one of :data:`SOURCES`.
            InputError: ``CRN-1001`` when an existing index/label file is absent from where an explicit
                path says it is, unreadable, malformed, foreign, or maps to different footage.
        """
        name = str(source)
        if name not in SOURCES:
            raise ConfigError(
                "CRN-0001",
                f"evaluation.ground_truth.source: {name!r} is not recognised; expected one of "
                f"{list(SOURCES)}.",
            )
        if name == "none":
            return cls._empty(
                "none",
                (
                    "ground_truth_source_none: evaluation.ground_truth.source is 'none', so no "
                    "reference is loaded and every accuracy field stays null"
                ),
            )
        if name == "nvidia":
            return cls._load_nvidia_index(path)
        return cls._load_user_export(path)

    @classmethod
    def from_config(
        cls, config: Config, *, expected_source: str | None = None
    ) -> GroundTruthReference:
        """Load the reference declared by a validated ``Config``.

        ``source='none'`` returns the existing empty reference, whose every modality carries the
        ``ground_truth_source_none`` reason, so opting out of external accuracy stays a null result
        rather than an error. A non-``none`` source that differs from *expected_source* is a
        :class:`ConfigError` naming both values, so a scorer cannot score one source's bytes under
        another source's contract.

        Missing data keeps :meth:`load`'s own semantics: a NVIDIA index that is absent is an
        ``InputError``, while an absent user label file is the documented empty reference. This
        method adds no new absence rule.
        """
        setting = config.evaluation.ground_truth
        source = str(setting.source)
        if expected_source is not None and source != "none" and source != expected_source:
            raise ConfigError(
                "CRN-0001",
                f"evaluation.ground_truth.source is {source!r}, but this scorer consumes "
                f"{expected_source!r}. Set source: {expected_source!r} with its path, or pass an "
                "explicit reference path to the scorer.",
            )
        return cls.load(source, setting.path)

    @classmethod
    def _empty(cls, source: str, reason: str) -> GroundTruthReference:
        return cls(
            source=source,
            index_path=None,
            dataset={},
            frames=0,
            present_modalities=(),
            missing_modalities=dict.fromkeys(MODALITIES, reason),
            records={},
            root=None,
        )

    @classmethod
    def _load_nvidia_index(cls, path: str | Path | None) -> GroundTruthReference:
        if path is None:
            raise InputError(
                "CRN-1001",
                "evaluation.ground_truth.source='nvidia' requires evaluation.ground_truth.path to "
                "name the converted gt-index.json or the directory that contains it.",
                stage="eval.ground_truth",
            )
        index_path = Path(path)
        if index_path.is_dir():
            index_path = index_path / "gt-index.json"
        if not index_path.is_file():
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' does not exist; convert the reference first with "
                "`python scripts/gt_nvidia.py convert --reference-dir <dir> --output "
                "<dir>/gt-index.json`.",
                stage="eval.ground_truth",
            )
        try:
            document = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' could not be read as JSON: {exc}",
                stage="eval.ground_truth",
            ) from exc
        if not isinstance(document, dict):
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' is not a JSON object; expected a "
                f"{INDEX_SCHEMA!r} document.",
                stage="eval.ground_truth",
            )
        schema = document.get("schema")
        if schema != INDEX_SCHEMA:
            if schema in SUPERSEDED_INDEX_SCHEMAS:
                raise InputError(
                    "CRN-1001",
                    f"ground-truth index '{index_path}' declares the superseded schema {schema!r}; "
                    f"that version recorded observation.state as robot end-effector state, which the "
                    f"dataset README contradicts. It cannot be read with the corrected object-pose "
                    f"meaning; re-run `python scripts/gt_nvidia.py convert --reference-dir <dir> "
                    f"--output {index_path}` to write {INDEX_SCHEMA!r}.",
                    stage="eval.ground_truth",
                )
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' declares schema {schema!r}; this loader accepts "
                f"only {INDEX_SCHEMA!r}.",
                stage="eval.ground_truth",
            )
        dataset = document.get("dataset")
        if not isinstance(dataset, dict) or not dataset.get("name"):
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' has no dataset block naming the source; a "
                "reference without an identity cannot be cited or licence-checked.",
                stage="eval.ground_truth",
            )
        raw_frames = document.get("frames")
        records_raw = document.get("frame_records")
        if not isinstance(raw_frames, int) or raw_frames < 0 or not isinstance(records_raw, list):
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' needs an integer 'frames' and a 'frame_records' "
                "list; the inventory cannot be read.",
                stage="eval.ground_truth",
            )
        records: dict[int, Mapping[str, object]] = {}
        for position, record in enumerate(records_raw):
            if not isinstance(record, dict):
                raise InputError(
                    "CRN-1001",
                    f"ground-truth index '{index_path}' frame record {position} is not an object.",
                    stage="eval.ground_truth",
                )
            frame_index = record.get("frame_index", position)
            if not isinstance(frame_index, int) or frame_index in records:
                raise InputError(
                    "CRN-1001",
                    f"ground-truth index '{index_path}' frame record {position} has a missing, "
                    "non-integer or duplicate frame_index.",
                    stage="eval.ground_truth",
                )
            records[frame_index] = record
        if len(records) != raw_frames:
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' declares {raw_frames} frames but carries "
                f"{len(records)} frame records; the inventory and the records disagree.",
                stage="eval.ground_truth",
            )
        modalities = document.get("modalities")
        if not isinstance(modalities, dict):
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' has no 'modalities' block; a reference must name "
                "what it supplies and what it does not.",
                stage="eval.ground_truth",
            )
        present_raw = modalities.get("present", [])
        missing_raw = modalities.get("missing", {})
        if not isinstance(present_raw, list) or not isinstance(missing_raw, dict):
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' modalities must hold a 'present' list and a "
                "'missing' object of reasons.",
                stage="eval.ground_truth",
            )
        present = tuple(str(item) for item in present_raw)
        unknown = [item for item in present if item not in MODALITIES]
        if unknown:
            raise InputError(
                "CRN-1001",
                f"ground-truth index '{index_path}' names unknown present modalities {unknown}; "
                f"known modalities are {list(MODALITIES)}.",
                stage="eval.ground_truth",
            )
        missing = {str(key): str(value) for key, value in missing_raw.items()}
        missing.update(
            {
                modality: (
                    f"{modality}_not_published: the index does not list this modality as present"
                )
                for modality in MODALITIES
                if modality not in present and modality not in missing
            }
        )
        return cls(
            source="nvidia",
            index_path=index_path,
            dataset=dict(dataset),
            frames=raw_frames,
            present_modalities=present,
            missing_modalities=missing,
            records=records,
            root=index_path.parent,
        )

    @classmethod
    def _load_user_export(cls, path: str | Path | None) -> GroundTruthReference:
        if path is None:
            return cls._empty("user_export", f"{USER_LABELS_ABSENT}: no label path was configured")
        base = Path(path)
        labels_path = base / "labels.json" if base.is_dir() else base
        if not labels_path.is_file():
            return cls._empty(
                "user_export",
                f"{USER_LABELS_ABSENT}: no label file at '{labels_path}'; every metric is null "
                "until the operator exports labels",
            )
        try:
            document = json.loads(labels_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError(
                "CRN-1001",
                f"user label file '{labels_path}' could not be read as JSON: {exc}",
                stage="eval.ground_truth",
            ) from exc
        if not isinstance(document, dict):
            raise InputError(
                "CRN-1001",
                f"user label file '{labels_path}' is not a JSON object; expected a "
                f"{USER_LABEL_SCHEMA!r} document.",
                stage="eval.ground_truth",
            )
        schema = document.get("schema")
        if schema != USER_LABEL_SCHEMA:
            if schema in SUPERSEDED_USER_LABEL_SCHEMAS:
                raise InputError(
                    "CRN-1001",
                    f"user label file '{labels_path}' declares the superseded schema {schema!r}; "
                    f"that version cannot record a human attestation on an unchanged pre-label. "
                    f"Re-run `python scripts/hand_accuracy.py import-labels` to write "
                    f"{USER_LABEL_SCHEMA!r}.",
                    stage="eval.ground_truth",
                )
            raise InputError(
                "CRN-1001",
                f"user label file '{labels_path}' declares schema {schema!r}; this loader accepts "
                f"only {USER_LABEL_SCHEMA!r}.",
                stage="eval.ground_truth",
            )
        video_name = document.get("video")
        if not isinstance(video_name, str) or not video_name.strip():
            raise InputError(
                "CRN-1001",
                f"user label file '{labels_path}' has no 'video' filename; a label set that does not "
                "name its clip cannot be matched to footage.",
                stage="eval.ground_truth",
            )
        label_hash = document.get("video_sha256")
        if not isinstance(label_hash, str) or not _SHA256_RE.match(label_hash):
            raise InputError(
                "CRN-1001",
                f"user label file '{labels_path}' needs a 64-hex 'video_sha256', got "
                f"{label_hash!r}; without it the labels could be scored against another clip.",
                stage="eval.ground_truth",
            )
        fps = _require_positive_number(document.get("fps"), f"{labels_path}: fps")
        raw_labels = document.get("labels")
        if not isinstance(raw_labels, list):
            raise InputError(
                "CRN-1001",
                f"user label file '{labels_path}' needs a 'labels' list.",
                stage="eval.ground_truth",
            )
        labels = _validate_user_labels(raw_labels, labels_path)
        video_path = _resolve_user_video(labels_path.parent, video_name, label_hash)
        hands_by_frame = {
            int(raw["frame_index"]): raw.get("hands", [])
            for raw in raw_labels
            if isinstance(raw, dict) and isinstance(raw.get("frame_index"), int)
        }
        attestations = validate_attestation_fields(
            document.get("attestations"), where=str(labels_path)
        )
        crosscheck_attestations(attestations, hands_by_frame, where=str(labels_path))
        actual_hash = _sha256_file(video_path)
        if actual_hash != label_hash.lower():
            raise InputError(
                "CRN-1001",
                f"user label file '{labels_path}' records video_sha256={label_hash.lower()} but the "
                f"staged footage '{video_path}' hashes to {actual_hash}; refusing to score labels "
                "against different footage.",
                stage="eval.ground_truth",
            )
        clip_frames = _clip_frame_count(labels_path.parent, video_path)
        for label in labels:
            frame_index = _require_int(label["frame_index"], f"{labels_path}: frame_index")
            if clip_frames is not None and frame_index >= clip_frames:
                raise InputError(
                    "CRN-1001",
                    f"user label file '{labels_path}' labels frame {frame_index}, but the staged "
                    f"footage '{video_path.name}' has {clip_frames} frames (last valid index "
                    f"{clip_frames - 1}).",
                    stage="eval.ground_truth",
                )
        for entry in attestations:
            frame_index = int(entry["frame_index"])
            if clip_frames is not None and frame_index >= clip_frames:
                raise InputError(
                    "CRN-1011",
                    f"user label file '{labels_path}' attests frame {frame_index}, but the staged "
                    f"footage '{video_path.name}' has {clip_frames} frames (last valid index "
                    f"{clip_frames - 1}).",
                    stage="eval.ground_truth",
                )
        records = {
            _require_int(label["frame_index"], f"{labels_path}: frame_index"): label
            for label in labels
        }
        present: list[str] = ["rgb"]
        if any(label["hands"] for label in labels):
            present.append("hands_2d")
        if any(label["objects"] for label in labels):
            present.append("object_boxes_2d")
        missing: dict[str, str] = {}
        for modality in MODALITIES:
            if modality in present:
                continue
            if modality == "hands_2d":
                missing[modality] = (
                    f"{USER_LABELS_ABSENT}: the label file has no hand entries, so 2-D keypoint "
                    "error is null rather than zero"
                )
            elif modality == "object_boxes_2d":
                missing[modality] = (
                    f"{USER_LABELS_ABSENT}: the label file has no object-box entries, so box IoU "
                    "is null rather than zero"
                )
            elif modality == "hands_3d":
                missing[modality] = (
                    "hands_3d_not_published: the user-export contract records 2-D keypoints only; "
                    "no 3-D reference is derived from them"
                )
            elif modality == "object_pose":
                missing[modality] = (
                    "object_pose_not_published: the user-export contract records 2-D boxes only; "
                    "no object 6-DoF pose is derived from them"
                )
            elif modality == "metric_depth":
                missing[modality] = (
                    "metric_depth_not_published: the user-export contract carries no un-normalised "
                    "float depth; image pixels are not a depth reference"
                )
            elif modality in {"camera_intrinsics", "camera_extrinsics"}:
                missing[modality] = (
                    f"{modality}_not_published: the user-export contract carries no calibration; a "
                    "pixel-error score needs no intrinsics and no metric claim is made"
                )
            else:
                missing[modality] = (
                    f"{modality}_not_published: the user-export contract does not supply this "
                    "modality"
                )
        dataset = {
            "name": "user_export",
            "video": video_name,
            "video_sha256": actual_hash,
            "license": "operator-owned local reference; not redistributed",
            "simulation": False,
        }
        return cls(
            source="user_export",
            index_path=labels_path,
            dataset=dataset,
            frames=int(clip_frames if clip_frames is not None else len(records)),
            present_modalities=tuple(present),
            missing_modalities=missing,
            records=records,
            root=labels_path.parent,
            video_path=video_path,
            video_sha256=actual_hash,
            label_sha256=_sha256_file(labels_path),
            fps=float(fps),
            clip_frames=clip_frames,
        )

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def licence(self) -> str | None:
        """The dataset licence exactly as the index records it, or ``None`` when absent."""
        value = self.dataset.get("license", self.dataset.get("licence"))
        return str(value) if value is not None else None

    @property
    def simulation(self) -> bool:
        """Whether the reference is simulated. Defaults to ``False`` only because an unknown source
        is not evidence of simulation; the NVIDIA index records ``true`` explicitly."""
        return bool(self.dataset.get("simulation", False))

    # ------------------------------------------------------------------ #
    # Accessors: absent modality returns (None, reason), never raises
    # ------------------------------------------------------------------ #

    def _record(self, frame_index: int) -> tuple[Mapping[str, object] | None, str | None]:
        record = self.records.get(int(frame_index))
        if record is None:
            return None, (
                f"frame_not_labelled: frame {frame_index} is not present in this "
                f"{self.frames}-frame reference"
            )
        return record, None

    def _missing_reason(self, modality: str) -> str:
        return self.missing_modalities.get(
            modality,
            f"{modality}_not_published: the reference does not supply this modality",
        )

    def _record_for(
        self, modality: str, frame_index: int
    ) -> tuple[Mapping[str, object] | None, str | None]:
        """The frame record, or the modality's absence reason when the reference has no records."""
        if not self.records:
            return None, self._missing_reason(modality)
        return self._record(frame_index)

    def _resolve(self, value: object) -> Path | None:
        if not isinstance(value, str) or self.root is None:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root / candidate

    def _decode_record(
        self,
        frame_index: int,
        *,
        modality: str,
        path_key: str,
        ordinal_key: str,
        mode: str,
    ) -> tuple[np.ndarray | None, str | None]:
        record, reason = self._record_for(modality, frame_index)
        if record is None:
            return None, reason
        path_value = record.get(path_key)
        if path_value is None:
            return None, self._missing_reason(modality)
        path = self._resolve(path_value)
        if path is None:
            return None, f"index_malformed: {path_key} is not a path string"
        if not path.is_file():
            return None, f"file_missing: {path}"
        ordinal = record.get(ordinal_key, frame_index)
        if not isinstance(ordinal, int):
            return None, f"index_malformed: {ordinal_key} is not an integer"
        return _decode_frame(path, ordinal, mode=mode)

    def rgb(self, frame_index: int) -> tuple[np.ndarray | None, str | None]:
        """``(H, W, 3)`` uint8 RGB frame, or ``(None, reason)`` when RGB is absent or undecodable."""
        if self.source == "user_export":
            if self.video_path is None:
                return None, self._missing_reason("rgb")
            return _decode_frame(self.video_path, int(frame_index), mode="rgb")
        return self._decode_record(
            frame_index,
            modality="rgb",
            path_key="rgb_path",
            ordinal_key="rgb_frame",
            mode="rgb",
        )

    def depth_normalised(self, frame_index: int) -> tuple[np.ndarray | None, str | None]:
        """``(H, W)`` float32 depth in the reference's own [0, 1] 8-bit units, not metres.

        A video that could not be decoded, or a reference without the depth stream, returns
        ``(None, reason)``. Use :func:`depth_rank_agreement` for this modality; its scale is not
        metric and its values are lossy.
        """
        image, reason = self._decode_record(
            frame_index,
            modality="depth_normalised",
            path_key="depth_path",
            ordinal_key="depth_frame",
            mode="gray",
        )
        if image is None:
            return None, reason
        return np.asarray(image, dtype=np.float32) / np.float32(255.0), None

    def segmentation(self, frame_index: int) -> tuple[np.ndarray | None, str | None]:
        """``(H, W, 3)`` uint8 colour-mask frame, or ``(None, reason)``.

        The colours are codec-coded and no colour-to-class map is published, so treat the array as a
        palette image and use :func:`palette_labels`.
        """
        return self._decode_record(
            frame_index,
            modality="semantic_segmentation",
            path_key="seg_path",
            ordinal_key="seg_frame",
            mode="rgb",
        )

    def metric_depth(self, frame_index: int) -> tuple[np.ndarray | None, str | None]:
        """``(H, W)`` float depth in metres from a per-frame ``.npy`` file, or ``(None, reason)``.

        The NVIDIA reference has no metric depth: its stream is 8-bit normalised video. A commercial
        intake can supply ``metric_depth_path`` per frame; this accessor reads only that explicit,
        un-normalised float array and never rescales a video into metres.
        """
        record, reason = self._record_for("metric_depth", frame_index)
        if record is None:
            return None, reason
        path = self._resolve(record.get("metric_depth_path"))
        if path is None:
            return None, self._missing_reason("metric_depth")
        if not path.is_file():
            return None, f"file_missing: {path}"
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            return None, f"decode_failed: {path}: {exc}"
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            return None, (
                f"index_malformed: metric depth {path} must be a 2-D float array, got "
                f"{array.dtype} {array.shape}"
            )
        return np.asarray(array, dtype=np.float32), None

    def hands_2d(self, frame_index: int) -> tuple[list[Mapping[str, object]] | None, str | None]:
        """Labelled 2-D hands for one frame, or ``(None, reason)``."""
        record, reason = self._record_for("hands_2d", frame_index)
        if record is None:
            return None, reason
        hands = record.get("hands")
        if not isinstance(hands, list):
            return None, self._missing_reason("hands_2d")
        if not hands:
            return None, self._missing_reason("hands_2d")
        return [dict(hand) for hand in hands], None

    def object_boxes_2d(
        self, frame_index: int
    ) -> tuple[list[Mapping[str, object]] | None, str | None]:
        """Labelled 2-D object boxes for one frame, or ``(None, reason)``."""
        record, reason = self._record_for("object_boxes_2d", frame_index)
        if record is None:
            return None, reason
        boxes = record.get("objects")
        if not isinstance(boxes, list):
            return None, self._missing_reason("object_boxes_2d")
        if not boxes:
            return None, self._missing_reason("object_boxes_2d")
        return [dict(box) for box in boxes], None

    def hands_3d(self, frame_index: int) -> tuple[np.ndarray | None, str | None]:
        """No 3-D reference exists for either source, so this always returns ``(None, reason)``."""
        return None, self._missing_reason("hands_3d")

    def object_pose(self, frame_index: int) -> tuple[dict[str, object] | None, str | None]:
        """Inline object pose from an index, or ``(None, reason)`` when it is not supplied.

        The NVIDIA index carries the dataset's ``observation.state[0:12]`` as
        ``{"rotation": 3x3, "translation": [3], "frame": ...,
        "units": "simulator_units_not_verified_metres"}``: the transform of the episode's single
        articulated object of interest. Whether that transform is the object's body or its moving
        part is not stated per frame, and the units are unverified. A commercial index may carry the
        same shape with metres and a named frame. Neither is converted into anything else: a robot
        state is never accepted as an object pose. The user export records 2-D boxes only.
        """
        record, reason = self._record_for("object_pose", frame_index)
        if record is None:
            return None, reason
        pose = record.get("object_pose")
        if not isinstance(pose, dict):
            return None, self._missing_reason("object_pose")
        return dict(pose), None

    def articulated_joint(
        self, frame_index: int
    ) -> tuple[dict[str, object] | None, str | None]:
        """Articulated-object joint value from an index, or ``(None, reason)``.

        The NVIDIA index records ``observation.state[12]`` as
        ``{"value": scalar, "units": "undocumented", "feature": "observation.state[12]"}``: the joint
        value of the episode's articulated object of interest. The dataset does not state radians or
        metres, so the value is never converted into a metric quantity.
        """
        record, reason = self._record_for("articulated_joint", frame_index)
        if record is None:
            return None, reason
        joint = record.get("articulated_joint")
        if not isinstance(joint, dict) or "value" not in joint:
            return None, self._missing_reason("articulated_joint")
        return dict(joint), None

    def robot_state(self, frame_index: int) -> tuple[dict[str, object] | None, str | None]:
        """Published robot/action vector from an index, or ``(None, reason)``.

        The NVIDIA index records the dataset's 34-D ``action`` vector per frame as
        ``{"values": [34 floats], "names": [34 names], "dimension": 34, "source": "action",
        "units": "undocumented", ...}``. The publisher does not state whether these are commanded
        or observed joint values, so the accessor returns the vector and its published names and
        makes no sensor-feedback or end-effector-pose claim. A vector whose declared dimension or
        name count disagrees with its values is refused by name rather than truncated or padded.
        """
        record, reason = self._record_for("robot_state", frame_index)
        if record is None:
            return None, reason
        state = record.get("robot_state")
        if not isinstance(state, dict):
            return None, self._missing_reason("robot_state")
        values = state.get("values")
        if not isinstance(values, list) or not values:
            return None, "robot_state_malformed: values must be a non-empty list"
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return None, "robot_state_malformed: values are not numeric"
        if array.ndim != 1 or not np.isfinite(array).all():
            return None, "robot_state_malformed: values must be a finite 1-D vector"
        dimension = state.get("dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            return None, "robot_state_malformed: dimension must be an integer"
        if array.size != dimension:
            return None, (
                f"robot_state_malformed: dimension says {dimension}, vector has {array.size}"
            )
        names = state.get("names")
        if not isinstance(names, list) or len(names) != array.size:
            found = len(names) if isinstance(names, list) else None
            return None, (
                f"robot_state_malformed: names count {found} does not match vector length "
                f"{array.size}"
            )
        if not all(isinstance(name, str) for name in names):
            return None, "robot_state_malformed: every name must be a string"
        return dict(state), None

    def camera_intrinsics(self) -> tuple[dict[str, object] | None, str | None]:
        """Published intrinsics from the index, or ``(None, reason)``."""
        intrinsics = self.dataset.get("camera_intrinsics")
        if not isinstance(intrinsics, dict):
            return None, self._missing_reason("camera_intrinsics")
        return dict(intrinsics), None

    def validity(self, frame_index: int) -> tuple[np.ndarray | None, str | None]:
        """Per-pixel validity supplied by a commercial intake, or ``(None, reason)``.

        NVIDIA publishes no per-frame validity mask; the depth video's saturation is a property of the
        codec, not a validity label, so it is not inferred here.
        """
        record, reason = self._record_for("success_labels", frame_index)
        if record is None:
            return None, reason
        path = self._resolve(record.get("validity_path"))
        if path is None:
            return None, self._missing_reason("success_labels")
        if not path.is_file():
            return None, f"file_missing: {path}"
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            return None, f"decode_failed: {path}: {exc}"
        return np.asarray(array, dtype=bool), None


# ---------------------------------------------------------------------------
# User-export validation
# ---------------------------------------------------------------------------


def validate_attestation_fields(
    raw: object, *, where: str, frame_limit: int | None = None
) -> list[dict[str, object]]:
    """Validate the structural contract of a ``crinis.user-labels/2`` ``attestations`` array.

    Shared by the import converter, the stage loader and the score loader so the three layers
    cannot drift. ``None`` (an absent array) normalizes to ``[]``. Each refusal names its own code:
    ``CRN-1010`` malformed entry/side/decision/duplicate, ``CRN-1011`` frame outside the axis,
    ``CRN-1017`` missing operator/timestamp/tolerance, ``CRN-1018`` malformed pre-label digest.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InputError(
            "CRN-1010",
            f"{where}.attestations must be a list, got {type(raw).__name__}.",
            stage="eval.ground_truth",
        )
    entries: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for position, entry in enumerate(raw):
        item = f"{where}.attestations[{position}]"
        if not isinstance(entry, Mapping):
            raise InputError("CRN-1010", f"{item} is not an object.", stage="eval.ground_truth")
        frame = entry.get("frame_index")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise InputError(
                "CRN-1010", f"{item}.frame_index must be an integer, got {frame!r}.",
                stage="eval.ground_truth",
            )
        if frame < 0 or (frame_limit is not None and frame >= frame_limit):
            bound = f"0..{frame_limit - 1}" if frame_limit is not None else "non-negative"
            raise InputError(
                "CRN-1011",
                f"{item} addresses frame {frame}, outside the declared frame axis ({bound}).",
                stage="eval.ground_truth",
            )
        side = entry.get("side")
        if side not in ("left", "right"):
            raise InputError(
                "CRN-1010", f"{item}.side must be 'left' or 'right', got {side!r}.",
                stage="eval.ground_truth",
            )
        decision = entry.get("decision")
        if decision != ATTESTATION_DECISION:
            raise InputError(
                "CRN-1010",
                f"{item}.decision must be {ATTESTATION_DECISION!r} (the only attestation decision; "
                f"absence is declared via declared_absent_slots), got {decision!r}.",
                stage="eval.ground_truth",
            )
        operator = entry.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            raise InputError(
                "CRN-1017",
                f"{item}.operator must be a non-empty string naming the reviewing human, got "
                f"{operator!r}.",
                stage="eval.ground_truth",
            )
        reviewed = entry.get("reviewed_at_utc")
        if not isinstance(reviewed, str) or not reviewed.strip():
            raise InputError(
                "CRN-1017",
                f"{item}.reviewed_at_utc must be a non-empty UTC timestamp string, got "
                f"{reviewed!r}.",
                stage="eval.ground_truth",
            )
        tolerance = entry.get("tolerance_px")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not np.isfinite(float(tolerance))
            or float(tolerance) <= 0.0
        ):
            raise InputError(
                "CRN-1017",
                f"{item}.tolerance_px must be a finite positive number of pixels, got "
                f"{tolerance!r}.",
                stage="eval.ground_truth",
            )
        sha = entry.get("prelabel_sha256")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            raise InputError(
                "CRN-1018",
                f"{item}.prelabel_sha256 must be 64 hexadecimal characters, got {sha!r}; an "
                "attestation must name the pre-label bytes it accepted.",
                stage="eval.ground_truth",
            )
        key = (frame, str(side))
        if key in seen:
            raise InputError(
                "CRN-1010",
                f"{item} repeats the attestation for frame {frame} side {side!r}; one attestation "
                "per frame/side is representable.",
                stage="eval.ground_truth",
            )
        seen.add(key)
        entries.append(
            {
                "frame_index": frame,
                "side": str(side),
                "decision": ATTESTATION_DECISION,
                "operator": operator,
                "reviewed_at_utc": reviewed,
                "tolerance_px": float(tolerance),
                "prelabel_sha256": sha.lower(),
            }
        )
    return entries


def crosscheck_attestations(
    attestations: Sequence[Mapping[str, object]],
    hands_by_frame: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    where: str,
) -> dict[tuple[int, str], Mapping[str, object]]:
    """Reconcile attestations with the document's hand records, one code per contradiction.

    ``CRN-1015`` an attestation and a human-edited shape for the same frame/side; ``CRN-1016`` an
    attestation with no materialized attested hand record (or an attested hand record with no
    attestation); ``CRN-1018`` the hand's ``prelabel_sha256`` differs from the attestation's.
    Returns ``(frame_index, side) -> attestation``.
    """
    by_slot: dict[tuple[int, str], Mapping[str, object]] = {
        (int(entry["frame_index"]), str(entry["side"])): entry for entry in attestations
    }
    materialized: set[tuple[int, str]] = set()
    for frame, hands in hands_by_frame.items():
        for slot, hand in enumerate(hands):
            item = f"{where}: frame {frame} hands[{slot}]"
            side = str(hand.get("side"))
            key = (int(frame), side)
            tag = hand.get("provenance")
            if tag == ATTESTED_HAND_PROVENANCE:
                entry = by_slot.get(key)
                if entry is None:
                    raise InputError(
                        "CRN-1016",
                        f"{item} is tagged {ATTESTED_HAND_PROVENANCE!r} but no attestation names "
                        f"frame {frame} side {side!r}; an attested hand must carry its human "
                        "sign-off.",
                        stage="eval.ground_truth",
                    )
                hand_sha = hand.get("prelabel_sha256")
                if not isinstance(hand_sha, str) or hand_sha.lower() != entry["prelabel_sha256"]:
                    raise InputError(
                        "CRN-1018",
                        f"{item}.prelabel_sha256 {hand_sha!r} does not match the attestation's "
                        f"{entry['prelabel_sha256']!r}; the adopted coordinates and the sign-off "
                        "must name the same pre-label bytes.",
                        stage="eval.ground_truth",
                    )
                materialized.add(key)
            elif tag in (None, EDITED_HAND_PROVENANCE):
                if key in by_slot:
                    raise InputError(
                        "CRN-1015",
                        f"{item} is a human-edited shape but an attestation also names frame "
                        f"{frame} side {side!r}; a slot is either edited or attested-unchanged, "
                        "never both.",
                        stage="eval.ground_truth",
                    )
            else:
                raise InputError(
                    "CRN-1010",
                    f"{item}.provenance must be {EDITED_HAND_PROVENANCE!r} or "
                    f"{ATTESTED_HAND_PROVENANCE!r}, got {tag!r}.",
                    stage="eval.ground_truth",
                )
    for key in by_slot:
        if key not in materialized:
            raise InputError(
                "CRN-1016",
                f"{where} attests frame {key[0]} side {key[1]!r} but no hand record tagged "
                f"{ATTESTED_HAND_PROVENANCE!r} materializes the accepted pre-label; the document "
                "must be self-contained.",
                stage="eval.ground_truth",
            )
    return by_slot


def _require_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError("CRN-1001", f"{label} must be a number, got {value!r}.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise InputError("CRN-1001", f"{label} must be finite and positive, got {value!r}.")
    return parsed


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError("CRN-1001", f"{label} must be an integer, got {value!r}.")
    return value


def _validate_user_labels(
    raw_labels: list[object], labels_path: Path
) -> tuple[Mapping[str, object], ...]:
    """Validate every required field of a user label export, naming the frame and index on failure."""
    labels: list[Mapping[str, object]] = []
    seen_frames: set[int] = set()
    for position, raw in enumerate(raw_labels):
        if not isinstance(raw, dict):
            raise InputError(
                "CRN-1001",
                f"{labels_path}: labels[{position}] is not an object.",
            )
        frame_index = _require_int(
            raw.get("frame_index"), f"{labels_path}: labels[{position}].frame_index"
        )
        if frame_index < 0:
            raise InputError(
                "CRN-1001",
                f"{labels_path}: labels[{position}].frame_index={frame_index} must be non-negative.",
            )
        if frame_index in seen_frames:
            raise InputError(
                "CRN-1001",
                f"{labels_path}: frame {frame_index} is labelled more than once; each frame may "
                "appear once so a residual has one reference.",
            )
        seen_frames.add(frame_index)
        time_s = raw.get("time_s")
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
            raise InputError(
                "CRN-1001",
                f"{labels_path}: labels[{position}].time_s must be a number, got {time_s!r}.",
            )
        if not np.isfinite(float(time_s)) or float(time_s) < 0.0:
            raise InputError(
                "CRN-1001",
                f"{labels_path}: labels[{position}].time_s must be finite and non-negative, got "
                f"{time_s!r}.",
            )
        hands = raw.get("hands")
        objects = raw.get("objects")
        if not isinstance(hands, list) or not isinstance(objects, list):
            raise InputError(
                "CRN-1001",
                f"{labels_path}: labels[{position}] needs 'hands' and 'objects' arrays (either may "
                "be empty).",
            )
        validated_hands = [
            _validate_hand(hand, labels_path, frame_index, slot)
            for slot, hand in enumerate(hands)
        ]
        sides = [str(hand["side"]) for hand in validated_hands]
        if len(set(sides)) != len(sides):
            raise InputError(
                "CRN-1001",
                f"{labels_path}: frame {frame_index} labels the same side more than once "
                f"({sides}); use at most one left and one right hand.",
            )
        validated_objects = [
            _validate_object(box, labels_path, frame_index, slot)
            for slot, box in enumerate(objects)
        ]
        tracks = [
            _require_int(box["track_id"], f"{labels_path}: frame {frame_index} objects.track_id")
            for box in validated_objects
        ]
        if len(set(tracks)) != len(tracks):
            raise InputError(
                "CRN-1001",
                f"{labels_path}: frame {frame_index} labels the same object track_id more than once "
                f"({tracks}).",
            )
        labels.append(
            {
                "frame_index": frame_index,
                "time_s": float(time_s),
                "hands": validated_hands,
                "objects": validated_objects,
            }
        )
    return tuple(labels)


def _validate_hand(
    hand: object, labels_path: Path, frame_index: int, slot: int
) -> Mapping[str, object]:
    where = f"{labels_path}: frame {frame_index} hands[{slot}]"
    if not isinstance(hand, dict):
        raise InputError("CRN-1001", f"{where} is not an object.")
    side = hand.get("side")
    if side not in ("left", "right"):
        raise InputError("CRN-1001", f"{where}.side must be 'left' or 'right', got {side!r}.")
    points = hand.get("kpts2d_px")
    if not isinstance(points, list) or len(points) != NUM_HAND_JOINTS:
        raise InputError(
            "CRN-1001",
            f"{where}.kpts2d_px must have {NUM_HAND_JOINTS} [x, y] entries, got "
            f"{len(points) if isinstance(points, list) else type(points).__name__}.",
        )
    parsed_points: list[list[float]] = []
    for joint, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            raise InputError(
                "CRN-1001",
                f"{where}.kpts2d_px[{joint}] must be [x, y], got {point!r}.",
            )
        values = []
        for axis, value in enumerate(point):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InputError(
                    "CRN-1001",
                    f"{where}.kpts2d_px[{joint}][{axis}] must be a number, got {value!r}.",
                )
            if not np.isfinite(float(value)):
                raise InputError(
                    "CRN-1001",
                    f"{where}.kpts2d_px[{joint}][{axis}] must be finite, got {value!r}.",
                )
            values.append(float(value))
        parsed_points.append(values)
    visible = hand.get("visible")
    if not isinstance(visible, list) or len(visible) != NUM_HAND_JOINTS:
        raise InputError(
            "CRN-1001",
            f"{where}.visible must have {NUM_HAND_JOINTS} booleans, got "
            f"{len(visible) if isinstance(visible, list) else type(visible).__name__}.",
        )
    if any(not isinstance(flag, bool) for flag in visible):
        raise InputError("CRN-1001", f"{where}.visible must contain only true/false values.")
    result: dict[str, object] = {"side": side, "kpts2d_px": parsed_points, "visible": visible}
    tag = hand.get("provenance")
    if tag is not None:
        if tag not in (EDITED_HAND_PROVENANCE, ATTESTED_HAND_PROVENANCE):
            raise InputError(
                "CRN-1010",
                f"{where}.provenance must be {EDITED_HAND_PROVENANCE!r} or "
                f"{ATTESTED_HAND_PROVENANCE!r}, got {tag!r}.",
            )
        result["provenance"] = tag
        if tag == ATTESTED_HAND_PROVENANCE:
            sha = hand.get("prelabel_sha256")
            if not isinstance(sha, str) or not _SHA256_RE.match(sha):
                raise InputError(
                    "CRN-1018",
                    f"{where}.prelabel_sha256 must be 64 hexadecimal characters, got {sha!r}; an "
                    "attested hand must name the pre-label bytes it adopted.",
                )
            result["prelabel_sha256"] = sha.lower()
    return result


def _validate_object(
    box: object, labels_path: Path, frame_index: int, slot: int
) -> Mapping[str, object]:
    where = f"{labels_path}: frame {frame_index} objects[{slot}]"
    if not isinstance(box, dict):
        raise InputError("CRN-1001", f"{where} is not an object.")
    track_id = _require_int(box.get("track_id"), f"{where}.track_id")
    class_name = box.get("class")
    if not isinstance(class_name, str) or not class_name.strip():
        raise InputError("CRN-1001", f"{where}.class must be a non-empty string.")
    bbox = box.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise InputError(
            "CRN-1001", f"{where}.bbox_xyxy must have four numbers [x0, y0, x1, y1]."
        )
    values: list[float] = []
    for axis, value in enumerate(bbox):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError(
                "CRN-1001", f"{where}.bbox_xyxy[{axis}] must be a number, got {value!r}."
            )
        if not np.isfinite(float(value)):
            raise InputError(
                "CRN-1001", f"{where}.bbox_xyxy[{axis}] must be finite, got {value!r}."
            )
        values.append(float(value))
    if values[2] < values[0] or values[3] < values[1]:
        raise InputError(
            "CRN-1001",
            f"{where}.bbox_xyxy must be ordered [x0<=x1, y0<=y1], got {values}.",
        )
    return {"track_id": track_id, "class": class_name, "bbox_xyxy": values}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_user_video(parent: Path, video_name: str, label_hash: str) -> Path:
    candidate = parent / video_name
    if candidate.is_file():
        return candidate
    receipt_path = parent / "receipt.json"
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = None
        if isinstance(receipt, dict):
            for key in ("path", "video_path"):
                value = receipt.get(key)
                if not isinstance(value, str):
                    continue
                resolved = Path(value)
                if not resolved.is_absolute():
                    resolved = parent / resolved
                if resolved.is_file() and (
                    resolved.name == Path(video_name).name
                    or str(receipt.get("sha256", "")).lower() == label_hash.lower()
                ):
                    return resolved
    raise InputError(
        "CRN-1001",
        f"user labels name video {video_name!r}, but '{candidate}' does not exist and no sibling "
        f"receipt.json resolves it to staged footage; stage the clip and run "
        "`scripts/gt_user.py prepare` before scoring.",
        stage="eval.ground_truth",
    )


def _clip_frame_count(parent: Path, video_path: Path) -> int | None:
    """The clip's frame count from the prepare receipt, else the container, else ``None``."""
    receipt_path = parent / "receipt.json"
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = None
        if isinstance(receipt, dict):
            frames = receipt.get("frames")
            if isinstance(frames, int) and frames > 0:
                return frames
    try:
        av = importlib.import_module("av")
    except ImportError:
        return None
    try:
        with av.open(str(video_path)) as container:
            frames = int(container.streams.video[0].frames)
    except Exception:
        return None
    return frames if frames > 0 else None


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode_frame(
    path: Path,
    ordinal: int,
    *,
    mode: str,
) -> tuple[np.ndarray | None, str | None]:
    """Decode one frame from a video or still-image path without raising.

    ``mode='gray'`` returns ``(H, W)`` uint8, ``mode='rgb'`` returns ``(H, W, 3)`` uint8. A video is
    decoded sequentially because h264 has no reliable random access without decoding the preceding
    frames; the reference clips are short, and correctness beats a seek that lands on the wrong frame.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            return None, f"decode_failed: {path}: {exc}"
        return np.asarray(array, dtype=np.uint8), None
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return _decode_image(path, mode=mode)
    try:
        av = importlib.import_module("av")
    except ImportError:
        return None, (
            "decoder_unavailable: PyAV (av) is not installed, so the reference video cannot be "
            "decoded; install the dev or modern extra, or run this on the inference box"
        )
    if not path.is_file():
        return None, f"file_missing: {path}"
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for position, frame in enumerate(container.decode(stream)):
                if position != ordinal:
                    continue
                if mode == "gray":
                    return np.ascontiguousarray(frame.to_ndarray(format="gray")), None
                return np.ascontiguousarray(frame.to_ndarray(format="rgb24")), None
            return None, f"frame_missing: {path} has no frame {ordinal}"
    except Exception as exc:  # an accessor returns a reason, never raises
        return None, f"decode_failed: {path}: {exc}"


def _decode_image(path: Path, *, mode: str) -> tuple[np.ndarray | None, str | None]:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError:
        return None, (
            "decoder_unavailable: Pillow is not installed, so the reference image cannot be "
            "decoded; install the dev extra"
        )
    try:
        with image_module.open(path) as handle:
            array = np.asarray(handle.convert("L" if mode == "gray" else "RGB"))
    except (OSError, ValueError) as exc:
        return None, f"decode_failed: {path}: {exc}"
    return np.ascontiguousarray(array), None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, the tie-corrected input to Spearman's rho.

    Ties receive the mean of the positions they span, which is the textbook correction; without it a
    plateau would be ordered arbitrarily by the sort and inflate or deflate the correlation.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < ordered.shape[0]:
        stop = start + 1
        while stop < ordered.shape[0] and ordered[stop] == ordered[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def depth_rank_agreement(
    predicted_m: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    """Spearman rank correlation between a metric prediction and the reference depth.

    The NVIDIA depth reference is an 8-bit h264 video: its values are normalised, lossy and saturated,
    so their *magnitude* is not metres and comparing metres to them would produce a number with no
    meaning. Rank correlation is scale-invariant, which is the one comparison this reference supports.

    ``valid`` is the caller's mask of pixels eligible for the comparison. This function additionally
    requires those pixels to be finite; it refuses shape mismatches, non-finite selected values, fewer
    than 100 valid pixels, and inputs with no rank variation (for which rho is undefined). It never
    returns NaN.

    Returns:
        ``{"rho": float, "n": int, "note": str}``.

    Raises:
        GeometryError: ``CRN-4001`` for a shape/finiteness violation, ``CRN-4003`` when the evidence is
            too small or degenerate.
    """
    prediction = np.asarray(predicted_m, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    select = np.asarray(valid, dtype=bool)
    if prediction.shape != expected.shape or prediction.shape != select.shape:
        raise GeometryError(
            CONTRACT_CODE,
            f"depth rank agreement needs matching shapes, got prediction {prediction.shape}, "
            f"reference {expected.shape}, valid {select.shape}.",
            stage="eval.ground_truth",
        )
    selected_prediction = prediction[select]
    selected_reference = expected[select]
    count = int(selected_prediction.shape[0])
    if count < 100:
        raise GeometryError(
            UNDEFINED_CODE,
            f"depth rank agreement needs at least 100 valid pixels, got {count}; a rank "
            "correlation over fewer pixels is not a measurement.",
            stage="eval.ground_truth",
        )
    if not np.isfinite(selected_prediction).all() or not np.isfinite(selected_reference).all():
        raise GeometryError(
            CONTRACT_CODE,
            "depth rank agreement received non-finite values inside the valid mask; exclude them "
            "from `valid` rather than letting them propagate into a correlation.",
            stage="eval.ground_truth",
        )
    rank_prediction = _ranks(selected_prediction)
    rank_reference = _ranks(selected_reference)
    centred_prediction = rank_prediction - rank_prediction.mean()
    centred_reference = rank_reference - rank_reference.mean()
    denominator = float(
        np.sqrt(
            np.dot(centred_prediction, centred_prediction)
            * np.dot(centred_reference, centred_reference)
        )
    )
    if denominator == 0.0:
        raise GeometryError(
            UNDEFINED_CODE,
            "depth rank agreement is undefined because the prediction or the reference has no rank "
            "variation over the valid pixels (a constant map has no ordering).",
            stage="eval.ground_truth",
        )
    rho = float(np.dot(centred_prediction, centred_reference) / denominator)
    return {
        "rho": rho,
        "n": count,
        "note": (
            "Spearman rank correlation over valid pixels. The reference is an 8-bit h264 "
            "normalised depth video, not metres, and it is lossy and saturating, so only the "
            "ordering is compared; the metric prediction is used for its ranks, never for its scale."
        ),
    }


def mask_iou(predicted: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Intersection-over-union of two boolean masks, refusing an empty union.

    An empty union means neither mask marks a pixel; IoU would be ``0/0``. Returning 0.0 would read as
    "both masks are wrong in the same way", which is not what happened, so this raises instead.

    Returns:
        ``{"iou": float, "intersection": int, "union": int, "n_pred": int, "n_ref": int}``.
    """
    prediction = np.asarray(predicted).astype(bool, copy=False)
    expected = np.asarray(reference).astype(bool, copy=False)
    if prediction.shape != expected.shape:
        raise GeometryError(
            CONTRACT_CODE,
            f"mask IoU needs matching shapes, got prediction {prediction.shape} and reference "
            f"{expected.shape}.",
            stage="eval.ground_truth",
        )
    intersection = int(np.count_nonzero(prediction & expected))
    union = int(np.count_nonzero(prediction | expected))
    if union == 0:
        raise GeometryError(
            UNDEFINED_CODE,
            "mask IoU is undefined because the union is empty; neither mask marks a pixel.",
            stage="eval.ground_truth",
        )
    return {
        "iou": intersection / union,
        "intersection": intersection,
        "union": union,
        "n_pred": int(np.count_nonzero(prediction)),
        "n_ref": int(np.count_nonzero(expected)),
    }


def box_iou(predicted_xyxy: np.ndarray, reference_xyxy: np.ndarray) -> dict[str, object]:
    """Intersection-over-union of two ``[x0, y0, x1, y1]`` boxes; empty union is a refusal.

    Boxes are in pixels, in the same image frame. A degenerate box (zero width or height) is allowed
    only if the other box overlaps it; the union must be positive for IoU to mean anything.
    """
    prediction = np.asarray(predicted_xyxy, dtype=np.float64)
    expected = np.asarray(reference_xyxy, dtype=np.float64)
    for label, box in (("predicted", prediction), ("reference", expected)):
        if box.shape != (4,) or not np.isfinite(box).all():
            raise GeometryError(
                CONTRACT_CODE,
                f"{label} box must be a finite [x0, y0, x1, y1], got {box}.",
                stage="eval.ground_truth",
            )
        if box[2] < box[0] or box[3] < box[1]:
            raise GeometryError(
                CONTRACT_CODE,
                f"{label} box must be ordered [x0<=x1, y0<=y1], got {box.tolist()}.",
                stage="eval.ground_truth",
            )
    inter_x0 = max(prediction[0], expected[0])
    inter_y0 = max(prediction[1], expected[1])
    inter_x1 = min(prediction[2], expected[2])
    inter_y1 = min(prediction[3], expected[3])
    intersection = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    area_prediction = float(
        max(0.0, prediction[2] - prediction[0]) * max(0.0, prediction[3] - prediction[1])
    )
    area_reference = float(
        max(0.0, expected[2] - expected[0]) * max(0.0, expected[3] - expected[1])
    )
    union = area_prediction + area_reference - intersection
    if union <= 0.0:
        raise GeometryError(
            UNDEFINED_CODE,
            f"box IoU is undefined because the union is empty (predicted area {area_prediction}, "
            f"reference area {area_reference}).",
            stage="eval.ground_truth",
        )
    return {
        "iou": float(intersection / union),
        "intersection_px2": float(intersection),
        "union_px2": float(union),
        "area_pred_px2": area_prediction,
        "area_ref_px2": area_reference,
    }


def keypoint_error_2d(
    predicted_px: np.ndarray,
    reference_px: np.ndarray,
    visible: np.ndarray,
    *,
    tolerance_px: float,
    predicted_valid: np.ndarray | None = None,
) -> dict[str, object]:
    """Per-joint 2-D keypoint error in pixels, with a *reported* pass tolerance.

    Only joints that are both labelled visible and predicted valid are measured. A joint with no
    reference or no prediction is ``None`` in ``per_joint_px``, never zero: zero would read as a
    perfect measurement of a joint nobody measured.

    The tolerance is the caller's and is echoed in the result; this function does not decide that any
    particular pixel error "passes" in general, it only counts the fraction within the stated band.

    Returns:
        ``{"available": True, "n_joints_measured": int, "n_joints_visible": int,
        "per_joint_px": list[float | None], "mean_px": float, "median_px": float,
        "pass_fraction": float, "tolerance_px": float}``; or ``{"available": False, "reason": ...}``
        when no joint could be measured.

    Raises:
        GeometryError: ``CRN-4001`` for shape violations or a non-positive tolerance.
    """
    prediction = np.asarray(predicted_px, dtype=np.float64)
    reference = np.asarray(reference_px, dtype=np.float64)
    flags = np.asarray(visible, dtype=bool)
    expected = (NUM_HAND_JOINTS, 2)
    if prediction.shape != expected or reference.shape != expected:
        raise GeometryError(
            CONTRACT_CODE,
            f"2-D keypoint error needs {expected} arrays, got prediction {prediction.shape} and "
            f"reference {reference.shape}.",
            stage="eval.ground_truth",
        )
    if flags.shape != (NUM_HAND_JOINTS,):
        raise GeometryError(
            CONTRACT_CODE,
            f"2-D keypoint visibility must be ({NUM_HAND_JOINTS},), got {flags.shape}.",
            stage="eval.ground_truth",
        )
    limit = float(tolerance_px)
    if not np.isfinite(limit) or limit <= 0.0:
        raise GeometryError(
            CONTRACT_CODE,
            f"tolerance_px must be finite and positive, got {tolerance_px!r}.",
            stage="eval.ground_truth",
        )
    if predicted_valid is None:
        valid = flags
    else:
        predicted_flags = np.asarray(predicted_valid, dtype=bool)
        if predicted_flags.shape != (NUM_HAND_JOINTS,):
            raise GeometryError(
                CONTRACT_CODE,
                f"2-D keypoint predicted_valid must be ({NUM_HAND_JOINTS},), got "
                f"{predicted_flags.shape}.",
                stage="eval.ground_truth",
            )
        valid = flags & predicted_flags
    valid = valid & np.isfinite(prediction).all(axis=1) & np.isfinite(reference).all(axis=1)
    distances = np.linalg.norm(prediction - reference, axis=1)
    per_joint: list[float | None] = [None] * NUM_HAND_JOINTS
    measured: list[float] = []
    for joint in range(NUM_HAND_JOINTS):
        if valid[joint]:
            value = float(distances[joint])
            per_joint[joint] = value
            measured.append(value)
    if not measured:
        return {
            "available": False,
            "reason": (
                "no_visible_reference_joints: no joint is both labelled visible and predicted valid, "
                "so 2-D keypoint error is undefined rather than zero"
            ),
            "n_joints_visible": int(flags.sum()),
            "tolerance_px": limit,
        }
    values = np.asarray(measured, dtype=np.float64)
    return {
        "available": True,
        "n_joints_measured": int(values.shape[0]),
        "n_joints_visible": int(flags.sum()),
        "per_joint_px": per_joint,
        "mean_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "pass_fraction": float(np.count_nonzero(values <= limit) / values.shape[0]),
        "tolerance_px": limit,
        "note": (
            "Pixel distance per visible joint. The tolerance is supplied by the caller and reported "
            "with the number; it is not a claim that this band is the right one."
        ),
    }


def palette_labels(
    rgb: np.ndarray,
    palette: Sequence[Sequence[int]],
    tolerance: float,
) -> np.ndarray:
    """Nearest-palette label map with an explicit unmatched sentinel.

    Codec smear makes a colour-mask video contain thousands of colours that are not in the published
    palette. Forcing each of them into the nearest class would hide that corruption inside a plausible
    label, so a pixel whose nearest palette entry is farther than *tolerance* (Euclidean RGB distance)
    becomes :data:`UNMATCHED_LABEL` instead. The function is deliberately independent of any
    colour-to-class table: the palette is the caller's, and labels are palette indices.

    Args:
        rgb: ``(H, W, 3)`` image in 0-255 units.
        palette: sequence of ``(r, g, b)`` entries in 0-255 units.
        tolerance: maximum Euclidean distance for a match; must be finite and non-negative.

    Returns:
        ``(H, W)`` int16 label map in ``[0, len(palette))`` or :data:`UNMATCHED_LABEL`.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise GeometryError(
            CONTRACT_CODE,
            f"palette labelling needs an (H, W, 3) RGB image, got {image.shape}.",
            stage="eval.ground_truth",
        )
    entries = np.asarray(palette, dtype=np.float64)
    if entries.ndim != 2 or entries.shape[1] != 3 or entries.shape[0] == 0:
        raise GeometryError(
            CONTRACT_CODE,
            f"palette labelling needs a non-empty sequence of (r, g, b) entries, got shape "
            f"{entries.shape}.",
            stage="eval.ground_truth",
        )
    limit = float(tolerance)
    if not np.isfinite(limit) or limit < 0.0:
        raise GeometryError(
            CONTRACT_CODE,
            f"palette tolerance must be finite and non-negative, got {tolerance!r}.",
            stage="eval.ground_truth",
        )
    flat = image.reshape(-1, 3).astype(np.float64)
    labels = np.full(flat.shape[0], UNMATCHED_LABEL, dtype=np.int16)
    chunk = 65536
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - entries[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(block.shape[0]), nearest]
        matched = nearest_distance <= limit
        labels[start : start + block.shape[0]] = np.where(matched, nearest, UNMATCHED_LABEL)
    return labels.reshape(image.shape[:2])


def _pose_components(pose: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Split a 3x4 or 4x4 transform into rotation and translation, refusing other shapes.

    The dataset publishes ``observation.state[0:12]`` as a vectorised 3x4, while a commercial intake
    may supply a homogeneous 4x4; both describe the same transform family. Any other shape is a
    contract violation, not a zero residual.
    """
    array = np.asarray(pose, dtype=np.float64)
    if array.shape == (3, 4):
        rotation, translation = array[:, :3], array[:, 3]
    elif array.shape == (4, 4):
        if not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise GeometryError(
                CONTRACT_CODE,
                f"{label} is not homogeneous: its bottom row is {array[3].tolist()}.",
                stage="eval.ground_truth",
            )
        rotation, translation = array[:3, :3], array[:3, 3]
    else:
        raise GeometryError(
            CONTRACT_CODE,
            f"{label} must be a 3x4 or 4x4 transform matrix, got shape {array.shape}.",
            stage="eval.ground_truth",
        )
    if not np.isfinite(array).all():
        raise GeometryError(
            CONTRACT_CODE,
            f"{label} must contain finite values only.",
            stage="eval.ground_truth",
        )
    return rotation, translation


def pose_residual(
    predicted_pose: np.ndarray | None = None,
    reference_pose: np.ndarray | None = None,
    *,
    reference: GroundTruthReference | None = None,
) -> dict[str, object]:
    """Object-pose residual against the reference transform of the object of interest.

    ``observation.state`` is not a robot end-effector state here: the README says
    ``observation.state``'s first 12 values are the vectorised transform matrix of the "object of
    interest", and the index carries that as ``object_pose``. So an object-pose residual is real, with
    two stated limits. Its translation is in the reference's own simulator units, which the dataset
    does not document as metres, so the number is **relative** (``units_unverified: true``): it is
    comparable between poses of the same reference and never a metric distance. Its identity is one
    articulated object per episode, and whether the transform belongs to the body or the moving part
    is not stated per frame. Rotation is a geodesic angle in degrees, which needs no units.

    Either pose may be a 3x4 (the published vectorisation, or a 4x4 with its homogeneous row dropped)
    or a 4x4 homogeneous matrix. A shape outside that family raises :class:`GeometryError`; absent
    inputs return ``available: False`` with :data:`POSE_INPUTS_ABSENT`, never a zero residual.
    """
    if predicted_pose is None or reference_pose is None:
        missing = [
            name
            for name, value in (("predicted_pose", predicted_pose), ("reference_pose", reference_pose))
            if value is None
        ]
        return {
            "available": False,
            "reason": POSE_INPUTS_ABSENT,
            "missing": missing,
            "note": (
                "pose_residual needs both a predicted and a reference object transform; one pose "
                "alone cannot be compared with itself."
            ),
        }
    predicted_rotation, predicted_translation = _pose_components(
        predicted_pose, label="predicted_pose"
    )
    reference_rotation, reference_translation = _pose_components(
        reference_pose, label="reference_pose"
    )
    relative = predicted_rotation @ reference_rotation.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    result: dict[str, object] = {
        "available": True,
        "rotation_error_deg": float(np.degrees(np.arccos(cosine))),
        "translation_error": float(np.linalg.norm(predicted_translation - reference_translation)),
        "units_unverified": True,
        "translation_relative": True,
        "note": (
            "Geodesic rotation error in degrees and Euclidean translation error in the reference's "
            "own units. The NVIDIA reference does not document those units, so the translation error "
            "is a relative figure: not metres, and not comparable with a reference whose scale was "
            "not verified the same way."
        ),
    }
    if reference is not None:
        result["reference"] = {
            "source": reference.source,
            "dataset": reference.dataset.get("name"),
        }
    return result
