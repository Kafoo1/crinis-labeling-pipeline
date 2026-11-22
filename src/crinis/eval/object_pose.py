"""Measurement: object-pose accuracy as a number rather than a claim.

Slice 1 of ``PLANS/object-centric-upgrade.md``, and the plan's own rule applies to this file: nothing
here asserts accuracy, it measures it. The metrics below are the contract - their names appear in
``docs/eval.md`` and in every claim the dataset makes about objects - so each one documents *how* it
aggregates, because a metric without its aggregation stated is ambiguous rather than precise.

Conventions, all of them deliberate:

* **Rotation and translation errors are reported as medians.** A handful of wild frames should not
  decide whether a dataset is good; the fraction of frames that were usable is reported separately
  (``valid_ratio``) so a median over three surviving frames cannot masquerade as a good episode.
* **ADD and ADD-S are means**, because that is the BOP convention and a comparison against published
  numbers is only meaningful if the aggregation matches.
* **Invalid frames are excluded, never interpolated.** A pose the tracker itself marked ``valid=False``
  is not evidence about accuracy, and counting it would flatter or damn the result arbitrarily.
* **A pose without a mesh cannot be scored by ADD**, so ``add``/``add_s`` come back as ``None`` rather
  than as zero. Zero would read as a perfect score.

Everything the geometry needs already exists in ``crinis.geom``; this module adds no new maths beyond
the aggregation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from crinis.errors import GeometryError
from crinis.geom.rotations import quat_conjugate, quat_multiply, quat_to_axis_angle, quat_to_matrix

#: The code for an evaluation input that violates the pose contract. CRN-4001 is this project's
#: "array is not usable" geometry failure, which is exactly what a mismatched pose array is.
CONTRACT_CODE = "CRN-4001"

#: A rotation error above this is treated as a wrap-around artefact rather than a real error.
MAX_ROTATION_DEG = 180.0


@dataclass(frozen=True)
class PoseMetrics:
    """One instance's pose accuracy over one episode, or a pool of instances.

    ``add_mean``/``add_s_mean`` are ``None`` when no mesh was supplied: ADD needs the object's vertices,
    and reporting 0.0 for "not measured" would be a lie a reader cannot detect.
    """

    instance_id: int
    frames_total: int
    frames_valid: int
    valid_ratio: float
    mean_stale_frames: float
    rot_err_deg_median: float
    rot_err_deg_p95: float
    trans_err_mm_median: float
    trans_err_mm_p95: float
    add_mean_m: float | None
    add_s_mean_m: float | None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe, for ``docs/eval.md`` and for run metadata."""
        return {
            "instance_id": self.instance_id,
            "frames_total": self.frames_total,
            "frames_valid": self.frames_valid,
            "valid_ratio": self.valid_ratio,
            "mean_stale_frames": self.mean_stale_frames,
            "rot_err_deg_median": self.rot_err_deg_median,
            "rot_err_deg_p95": self.rot_err_deg_p95,
            "trans_err_mm_median": self.trans_err_mm_median,
            "trans_err_mm_p95": self.trans_err_mm_p95,
            "add_mean_m": self.add_mean_m,
            "add_s_mean_m": self.add_s_mean_m,
        }


def _check_pose_arrays(quat: np.ndarray, trans: np.ndarray, *, label: str) -> None:
    if quat.shape[:-1] != trans.shape[:-1] or quat.shape[-1] != 4 or trans.shape[-1] != 3:
        raise GeometryError(
            CONTRACT_CODE,
            f"{label} quaternions {quat.shape} and translations {trans.shape} are not a pose "
            f"sequence: both need a matching leading shape, 4 and 3 on the last axis",
            stage="eval.object_pose",
        )
    if not np.isfinite(quat).all() or not np.isfinite(trans).all():
        raise GeometryError(
            CONTRACT_CODE,
            f"{label} contains non-finite values; a pose that is not a number cannot be scored",
            stage="eval.object_pose",
        )


def rotation_error_deg(pred_quat: np.ndarray, gt_quat: np.ndarray) -> np.ndarray:
    """Geodesic rotation error per frame, in degrees.

    The relative rotation's angle, so a 5 degree mistake reads as 5 whatever axis it is about.
    """
    relative = quat_multiply(pred_quat, quat_conjugate(gt_quat))
    angle_rad = np.linalg.norm(quat_to_axis_angle(relative), axis=-1)
    return np.asarray(np.degrees(angle_rad))


def translation_error_mm(pred_trans_m: np.ndarray, gt_trans_m: np.ndarray) -> np.ndarray:
    """Euclidean translation error per frame, in millimetres (the input is metres)."""
    return np.asarray(np.linalg.norm(pred_trans_m - gt_trans_m, axis=-1) * 1000.0)


def _transformed(quat: np.ndarray, trans_m: np.ndarray, vertices_m: np.ndarray) -> np.ndarray:
    """Apply a per-frame pose to one vertex set: ``(F, V, 3)``."""
    matrices = quat_to_matrix(quat)  # (F, 3, 3)
    rotated = vertices_m[None, :, :] @ np.swapaxes(matrices, -1, -2)  # (F, V, 3)
    return np.asarray(rotated + trans_m[:, None, :])


def add(
    pred_quat: np.ndarray,
    pred_trans_m: np.ndarray,
    gt_quat: np.ndarray,
    gt_trans_m: np.ndarray,
    vertices_m: np.ndarray,
) -> np.ndarray:
    """Average Distance of model points, per frame, in metres (visible-surface-free form).

    Two poses of the same rigid object are scored by how far the mesh points move, which is the
    quantity a policy actually cares about - unlike a rotation angle, it is symmetric in the object's
    geometry.
    """
    predicted = _transformed(pred_quat, pred_trans_m, vertices_m)
    truth = _transformed(gt_quat, gt_trans_m, vertices_m)
    return np.asarray(np.linalg.norm(predicted - truth, axis=-1).mean(axis=-1))


def add_s(
    pred_quat: np.ndarray,
    pred_trans_m: np.ndarray,
    gt_quat: np.ndarray,
    gt_trans_m: np.ndarray,
    vertices_m: np.ndarray,
) -> np.ndarray:
    """Average Distance of model points with symmetry handling, per frame, in metres.

    ADD-S forgives a pose that is right up to the object's own symmetry, which matters because a mug
    rotated 180 degrees about its handle axis is often the same mug to a task.
    """
    predicted = _transformed(pred_quat, pred_trans_m, vertices_m)
    truth = _transformed(gt_quat, gt_trans_m, vertices_m)
    # (F, V, V) defeats a long clip; the smoke sizes here are small, and the caller can subsample the
    # mesh if it ever bites. Chunking would change the answer only by rounding.
    distances = np.linalg.norm(predicted[:, :, None, :] - truth[:, None, :, :], axis=-1)
    return np.asarray(distances.min(axis=-1).mean(axis=-1))


def pose_errors(
    *,
    pred_quat: np.ndarray,
    pred_trans_m: np.ndarray,
    gt_quat: np.ndarray,
    gt_trans_m: np.ndarray,
    instance_id: int = -1,
    valid: np.ndarray | None = None,
    stale_frames: np.ndarray | None = None,
    vertices_m: np.ndarray | None = None,
) -> PoseMetrics:
    """Score one instance's predicted trajectory against ground truth for one episode.

    ``valid`` is the tracker's own honesty flag; frames it marks false are excluded from every error
    and counted in ``valid_ratio``, which is what stops a median over three frames from looking good.
    """
    pred_quat = np.asarray(pred_quat, dtype=np.float64)
    pred_trans_m = np.asarray(pred_trans_m, dtype=np.float64)
    gt_quat = np.asarray(gt_quat, dtype=np.float64)
    gt_trans_m = np.asarray(gt_trans_m, dtype=np.float64)
    _check_pose_arrays(pred_quat, pred_trans_m, label="predicted")
    _check_pose_arrays(gt_quat, gt_trans_m, label="ground truth")
    if pred_quat.shape != gt_quat.shape:
        raise GeometryError(
            CONTRACT_CODE,
            f"predicted shape {pred_quat.shape} and ground-truth shape {gt_quat.shape} differ; "
            f"two trajectories can only be scored frame by frame",
            stage="eval.object_pose",
        )

    frames_total = int(pred_quat.shape[0])
    keep = np.ones(frames_total, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if keep.shape != (frames_total,):
        raise GeometryError(
            CONTRACT_CODE,
            f"valid mask {keep.shape} does not match {frames_total} frames",
            stage="eval.object_pose",
        )
    frames_valid = int(keep.sum())

    stale = np.asarray(stale_frames, dtype=np.float64) if stale_frames is not None else None
    mean_stale = float(stale[keep].mean()) if stale is not None and frames_valid else 0.0

    if frames_valid == 0:
        # An instance seen in no usable frame is reported as such rather than as 0 degrees of error.
        return PoseMetrics(
            instance_id=instance_id,
            frames_total=frames_total,
            frames_valid=0,
            valid_ratio=0.0,
            mean_stale_frames=mean_stale,
            rot_err_deg_median=float("nan"),
            rot_err_deg_p95=float("nan"),
            trans_err_mm_median=float("nan"),
            trans_err_mm_p95=float("nan"),
            add_mean_m=None,
            add_s_mean_m=None,
        )

    rotation = rotation_error_deg(pred_quat[keep], gt_quat[keep])
    translation = translation_error_mm(pred_trans_m[keep], gt_trans_m[keep])
    add_per_frame = add_s_per_frame = None
    if vertices_m is not None:
        vertices = np.asarray(vertices_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise GeometryError(
                CONTRACT_CODE,
                f"vertices {vertices.shape} are not a (V, 3) mesh",
                stage="eval.object_pose",
            )
        add_per_frame = add(
            pred_quat[keep], pred_trans_m[keep], gt_quat[keep], gt_trans_m[keep], vertices
        )
        add_s_per_frame = add_s(
            pred_quat[keep], pred_trans_m[keep], gt_quat[keep], gt_trans_m[keep], vertices
        )

    return PoseMetrics(
        instance_id=instance_id,
        frames_total=frames_total,
        frames_valid=frames_valid,
        valid_ratio=frames_valid / frames_total,
        mean_stale_frames=mean_stale,
        rot_err_deg_median=float(np.median(rotation)),
        rot_err_deg_p95=float(np.percentile(rotation, 95)),
        trans_err_mm_median=float(np.median(translation)),
        trans_err_mm_p95=float(np.percentile(translation, 95)),
        add_mean_m=float(np.mean(add_per_frame)) if add_per_frame is not None else None,
        add_s_mean_m=float(np.mean(add_s_per_frame)) if add_s_per_frame is not None else None,
    )


def pool_metrics(metrics: Sequence[PoseMetrics], *, instance_id: int = -1) -> PoseMetrics:
    """Aggregate per-instance metrics into one row.

    Weighted by valid frames, not by instance: an instance with three usable frames should not count
    as much as one with three hundred. ADD-family values are pooled only when every input measured
    them, because a mean over a subset silently narrows what the number means.
    """
    if not metrics:
        raise GeometryError(
            CONTRACT_CODE,
            "nothing to pool: the caller handed over an empty metric list",
            stage="eval.object_pose",
        )
    total = sum(item.frames_total for item in metrics)
    valid = sum(item.frames_valid for item in metrics)
    weights = np.asarray([item.frames_valid for item in metrics], dtype=np.float64)
    add_values = [item.add_mean_m for item in metrics]
    add_s_values = [item.add_s_mean_m for item in metrics]
    # ADD is pooled from the instances that were actually *scored*: an instance with no valid frames
    # reports None for a reason that has nothing to do with having a mesh, and letting it null the pool
    # would discard measurements that exist.
    scored = [index for index, weight in enumerate(weights) if weight > 0]
    add_measured = all(add_values[index] is not None for index in scored)
    add_s_measured = all(add_s_values[index] is not None for index in scored)
    # Narrowed to the *scored* instances: `np.average` cannot take a list containing None, and the
    # weights must come from the same instances as the values or the two arrays disagree in length.
    scored_weights = [weights[index] for index in scored]
    add_floats = np.asarray(
        [add_values[index] for index in scored if add_values[index] is not None], dtype=np.float64
    )
    add_s_floats = np.asarray(
        [add_s_values[index] for index in scored if add_s_values[index] is not None], dtype=np.float64
    )

    def _weighted_median(values: list[float]) -> float:
        """Weighted over the instances that had usable frames; the rest are dropped, not averaged.

        Averaging with a zero weight is not neutral: `np.average` computes ``sum(w*x)/sum(w)`` and
        ``0.0 * nan == nan``, so an instance with no valid frames - which honestly reports NaN, see
        `pose_errors` - would poison the pooled row that the CLI prints and writes as the headline
        number, while that same row's `frames_valid` and `valid_ratio` reported real counts and made it
        look populated. Found by reproducing it with this module's own code, in review.
        """
        usable = [(value, weight) for value, weight in zip(values, weights, strict=True) if weight > 0]
        if not usable:
            return float("nan")
        return float(
            np.average([value for value, _ in usable], weights=[weight for _, weight in usable])
        )

    return PoseMetrics(
        instance_id=instance_id,
        frames_total=total,
        frames_valid=valid,
        valid_ratio=(valid / total) if total else 0.0,
        mean_stale_frames=_weighted_median([item.mean_stale_frames for item in metrics]),
        rot_err_deg_median=_weighted_median([item.rot_err_deg_median for item in metrics]),
        rot_err_deg_p95=_weighted_median([item.rot_err_deg_p95 for item in metrics]),
        trans_err_mm_median=_weighted_median([item.trans_err_mm_median for item in metrics]),
        trans_err_mm_p95=_weighted_median([item.trans_err_mm_p95 for item in metrics]),
        add_mean_m=(
            float(np.average(add_floats, weights=scored_weights))
            if add_measured and scored
            else None
        ),
        add_s_mean_m=(
            float(np.average(add_s_floats, weights=scored_weights))
            if add_s_measured and scored
            else None
        ),
    )
