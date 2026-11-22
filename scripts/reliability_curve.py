"""MOAT B1 — a calibrated per-instance reliability curve from the quality surface (§46.2).

The repair pipeline already writes a per-instance quality surface (``per_instance.csv``): every
instance carries ``pose_quality``, ``presence_confidence`` and a bundle of numeric geometry signals.
This script turns that surface into a *reliability curve* a buyer can filter on — but only when human
labels exist to calibrate it. The refusal-first contract is the point:

* **No labels** → the script still runs and emits, per tier (``pose_quality x presence_confidence``),
  the instance count, the corpus share, and the p10/p50/p90 of every numeric signal with a seeded
  bootstrap CI on the median. Every such number is tagged ``evidence: "distribution_only"`` and the
  document carries ``calibrated: false`` with a named reason. **No accuracy fraction is emitted.**
* **Labels** (``crinis.user-labels/2`` via :mod:`crinis.eval.ground_truth`) → per tier and pixel
  tolerance, the fraction of labelled joints within tolerance, a **Wilson 95% interval**, and ``n``.
  A tier below ``--min-n`` is ``insufficient_evidence`` and never gets a bare point estimate.
  ``calibrated: true`` only when at least one tier clears the minimum.

The labelled path needs the pipeline's prediction keypoints to measure a distance against. They are
the repaired output the surface describes, preserved beside it as ``smoothed.npz`` (per-clip key
``<clip>__smoothed``, shape ``(T, 2, 21, 2)`` pixels, slot 0 = left / slot 1 = right). When labels are
supplied but that bundle is absent, the script refuses to invent a distance: ``calibrated: false``
with reason ``predictions_absent``. No prediction artifact is ever written.

Monotonicity is checked on the labelled subset: does ``strong`` really beat ``uncertain`` beat
``absent``? A non-monotonic ordering is a FINDING, printed and recorded, never hidden. (``partial`` is
an ``identity_confidence`` value, not a ``presence_confidence`` one; the ordering uses the actual
``presence_confidence`` domain.)

The CSV is streamed row by row; per-tier signal values accumulate in compact ``array('d')`` buffers,
never a list of 147k dicts. Bootstrap draws are bounded so a large tier cannot allocate a
``(draws, n)`` index bigger than the host can hold.

Usage:
    python scripts/reliability_curve.py --surface .omp/qa/prod/repair/per_instance.csv
    python scripts/reliability_curve.py --surface .../per_instance.csv --labels .../labels.json
"""

from __future__ import annotations

import argparse
import array
import csv
import math
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from crinis.eval.ground_truth import USER_LABEL_SCHEMA, GroundTruthReference
from crinis.eval.label_quality import DEFAULT_SEED, DRAWS, bootstrap_median_ci
from crinis.io.storage import atomic_write_json, sha256_file

#: Output document schema. A foreign consumer can refuse a version it does not understand.
RELIABILITY_SCHEMA = "crinis.reliability-curve/1"

#: The numeric signal columns of the surface, in report order. Empty cells are unmeasured, not zero.
NUMERIC_SIGNALS: tuple[str, ...] = (
    "wrist_step_px",
    "mcp_median_step_px",
    "wrist_ratio",
    "bone_ratio",
    "phalanx_inversion_fraction",
    "mcp_line_residual_norm",
    "mcp_spread_norm",
    "collapsed_bone_fraction",
    "spread_px",
    "bone_cv_worst",
    "frame_bone_deviation",
    "off_frame_fraction",
    "valid_joint_fraction",
    "track_run_frames",
    "detector_score",
    "content_std",
    "content_edge",
)

#: The categorical flag columns reported as per-tier breakdowns.
FLAG_SIGNALS: tuple[str, ...] = ("wrist_flick", "hand_vs_arm")

#: Confidence ordering on ``presence_confidence`` for the monotonicity check, high → low.
PRESENCE_ORDER: tuple[str, ...] = ("strong", "uncertain", "absent")

#: Slot convention shared with the pipeline: axis-1 slot 0 is the left hand, slot 1 the right.
SIDE_TO_SLOT = {"left": 0, "right": 1}

#: Two-sided 95% normal quantile used by the Wilson interval.
WILSON_Z = 1.959963984540054

#: Cap on the bootstrap ``(draws, n)`` element count so a large tier stays within host memory.
_BOOTSTRAP_ELEMENT_CAP = 2_000_000

#: Floor on bootstrap draws for a large tier; the percentile CI stays valid, only coarser.
_BOOTSTRAP_MIN_DRAWS = 100


def tier_key(pose_quality: str, presence_confidence: str) -> str:
    """The canonical tier identifier ``pose_quality|presence_confidence``."""
    return f"{pose_quality}|{presence_confidence}"


def wilson_interval(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """The Wilson 95% score interval for ``successes`` of ``n`` Bernoulli trials.

    Unlike a Wald interval it stays inside ``[0, 1]`` and is defined at ``p̂ = 0`` or ``1``; ``n = 0``
    returns the uninformative ``[0, 1]`` rather than dividing by zero.
    """
    if n <= 0:
        return (0.0, 1.0)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _bootstrap_draws(n: int) -> int:
    """Draws bounded so ``(draws, n)`` stays under the element cap; full convention for small tiers."""
    if n <= 0:
        return DRAWS
    return min(DRAWS, max(_BOOTSTRAP_MIN_DRAWS, _BOOTSTRAP_ELEMENT_CAP // n))


def signal_summary(values: array.array) -> dict[str, object] | None:
    """p10/p50/p90 plus a seeded bootstrap CI on the median for one tier's signal values.

    Returns ``None`` when the signal was never measured in the tier. The median point estimate is the
    full-data median; the CI reuses :func:`bootstrap_median_ci` with a memory-bounded draw count.
    """
    n = len(values)
    if n == 0:
        return None
    data = np.frombuffer(values, dtype=np.float64)
    p10, p50, p90 = (float(v) for v in np.percentile(data, (10.0, 50.0, 90.0)))
    draws = _bootstrap_draws(n)
    _, lo, hi = bootstrap_median_ci(data, seed=DEFAULT_SEED, draws=draws)
    return {
        "n_measured": n,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "median_ci95": [lo, hi],
        "bootstrap_draws": draws,
        "evidence": "distribution_only",
    }


class SurfaceAccumulator:
    """Streaming per-tier accumulator: compact signal buffers, flag counts, and an optional join.

    One pass over the CSV feeds both the distribution report (every clip) and, when labels name a
    clip, a ``(frame, side) -> tier`` lookup used to place each labelled hand in its tier.
    """

    def __init__(self, label_clip: str | None = None) -> None:
        self.label_clip = label_clip
        self.total = 0
        self.counts: Counter[str] = Counter()
        self.signals: dict[str, dict[str, array.array]] = {}
        self.flags: dict[str, dict[str, Counter[str]]] = {}
        self.label_lookup: dict[tuple[int, str], str] = {}
        self.label_clip_rows = 0

    def _tier(self, key: str) -> tuple[dict[str, array.array], dict[str, Counter[str]]]:
        if key not in self.signals:
            self.signals[key] = {name: array.array("d") for name in NUMERIC_SIGNALS}
            self.flags[key] = {name: Counter() for name in FLAG_SIGNALS}
        return self.signals[key], self.flags[key]

    def add(self, row: dict[str, str]) -> None:
        """Accumulate one surface row. Unmeasured (empty/non-numeric) signal cells are skipped."""
        pose = row["pose_quality"]
        presence = row["presence_confidence"]
        key = tier_key(pose, presence)
        self.total += 1
        self.counts[key] += 1
        sig_buf, flag_buf = self._tier(key)
        for name in NUMERIC_SIGNALS:
            raw = row.get(name, "")
            if raw == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                sig_buf[name].append(value)
        for name in FLAG_SIGNALS:
            flag_buf[name][row.get(name, "") or "unmeasured"] += 1
        if self.label_clip is not None and row["clip"] == self.label_clip:
            self.label_clip_rows += 1
            try:
                frame = int(row["frame"])
            except ValueError:
                return
            self.label_lookup[(frame, row["side"])] = key


def accumulate_surface(csv_path: Path, label_clip: str | None = None) -> SurfaceAccumulator:
    """Stream the whole surface into a :class:`SurfaceAccumulator` without materialising the rows."""
    acc = SurfaceAccumulator(label_clip)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            acc.add(row)
    return acc


def _load_predictions(surface_path: Path, clip: str) -> tuple[np.ndarray | None, Path | None, str | None]:
    """The repaired prediction keypoints for *clip* from ``smoothed.npz`` beside the surface.

    Returns ``(keypoints, path, reason)``. Only the named clip's array is read (lazy npz access), so
    the other clips' bytes never enter memory. Absence is a named refusal, not a crash.
    """
    bundle_path = surface_path.parent / "smoothed.npz"
    if not bundle_path.is_file():
        return (
            None,
            None,
            f"predictions_absent: no 'smoothed.npz' beside the surface at '{surface_path.parent}'; "
            "the labelled joints have no prediction to be measured against, so no accuracy is claimed",
        )
    key = f"{clip}__smoothed"
    with np.load(bundle_path, allow_pickle=False) as npz:
        if key not in npz.files:
            return (
                None,
                bundle_path,
                f"predictions_absent: '{bundle_path}' has no key '{key}' for clip '{clip}'; "
                "no accuracy is claimed for a clip the prediction bundle does not cover",
            )
        keypoints = np.asarray(npz[key], dtype=np.float64)
    if keypoints.ndim != 4 or keypoints.shape[2:] != (21, 2):
        return (
            None,
            bundle_path,
            f"predictions_absent: '{key}' has shape {keypoints.shape}, expected (T, 2, 21, 2)",
        )
    return keypoints, bundle_path, None


def compute_labelled_accuracy(
    reference: GroundTruthReference,
    predictions: np.ndarray,
    label_lookup: dict[tuple[int, str], str],
    tolerances: tuple[float, ...],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Per-tier within-tolerance joint fractions against the labelled hands, plus join diagnostics.

    A labelled joint enters the denominator only when its ``visible`` flag is true (an invisible
    joint's ``[0, 0]`` is a placeholder, not a measurement). Hands are matched to the surface tier by
    ``(frame, side)`` and to the prediction by the side's slot.
    """
    within: dict[str, dict[float, int]] = {}
    totals: dict[str, int] = {}
    diagnostics: dict[str, object] = {
        "labelled_frames": len(reference.records),
        "labelled_hands": 0,
        "matched_hands": 0,
        "unmatched_label_hands": 0,
        "out_of_range_frames": 0,
        "visible_joints": 0,
    }
    n_frames = int(predictions.shape[0])
    for frame in sorted(reference.records):
        hands, _reason = reference.hands_2d(frame)
        if hands is None:
            continue
        if frame >= n_frames:
            diagnostics["out_of_range_frames"] = int(diagnostics["out_of_range_frames"]) + 1
            continue
        for hand in hands:
            diagnostics["labelled_hands"] = int(diagnostics["labelled_hands"]) + 1
            side = str(hand["side"])
            key = label_lookup.get((frame, side))
            if key is None:
                diagnostics["unmatched_label_hands"] = int(diagnostics["unmatched_label_hands"]) + 1
                continue
            slot = SIDE_TO_SLOT.get(side)
            if slot is None or slot >= predictions.shape[1]:
                diagnostics["unmatched_label_hands"] = int(diagnostics["unmatched_label_hands"]) + 1
                continue
            diagnostics["matched_hands"] = int(diagnostics["matched_hands"]) + 1
            label_pts = np.asarray(hand["kpts2d_px"], dtype=np.float64)
            visible = np.asarray(hand["visible"], dtype=bool)
            pred_pts = predictions[frame, slot]
            distances = np.linalg.norm(label_pts - pred_pts, axis=1)
            tier_within = within.setdefault(key, dict.fromkeys(tolerances, 0))
            for joint in range(distances.shape[0]):
                if not bool(visible[joint]) or not math.isfinite(float(distances[joint])):
                    continue
                diagnostics["visible_joints"] = int(diagnostics["visible_joints"]) + 1
                totals[key] = totals.get(key, 0) + 1
                for tol in tolerances:
                    if float(distances[joint]) <= tol:
                        tier_within[tol] += 1
    accuracy: dict[str, dict[str, object]] = {}
    for key, total in totals.items():
        per_tol: dict[str, object] = {}
        for tol in tolerances:
            successes = within[key][tol]
            lo, hi = wilson_interval(successes, total)
            per_tol[str(tol)] = {
                "fraction": successes / total,
                "wilson_ci95": [lo, hi],
                "n": total,
                "evidence": "labelled",
            }
        accuracy[key] = per_tol
    return accuracy, diagnostics


def check_monotonicity(
    accuracy: dict[str, dict[str, object]],
    tolerances: tuple[float, ...],
    min_n: int,
) -> list[dict[str, object]]:
    """Find labelled tiers where a higher ``presence_confidence`` does NOT beat a lower one.

    Compared within each ``pose_quality`` along :data:`PRESENCE_ORDER`, at every tolerance, using only
    tiers that clear ``min_n``. Each violation names the pair, the tolerance and both fractions.
    """
    violations: list[dict[str, object]] = []
    pose_qualities = sorted({key.split("|", 1)[0] for key in accuracy})
    for pose in pose_qualities:
        sufficient = [
            level
            for level in PRESENCE_ORDER
            if tier_key(pose, level) in accuracy
            and int(accuracy[tier_key(pose, level)][str(tolerances[0])]["n"]) >= min_n
        ]
        for higher, lower in pairwise(sufficient):
            high_key = tier_key(pose, higher)
            low_key = tier_key(pose, lower)
            for tol in tolerances:
                high_frac = float(accuracy[high_key][str(tol)]["fraction"])
                low_frac = float(accuracy[low_key][str(tol)]["fraction"])
                if high_frac < low_frac:
                    violations.append(
                        {
                            "pose_quality": pose,
                            "higher_confidence": higher,
                            "lower_confidence": lower,
                            "tolerance_px": tol,
                            "higher_fraction": high_frac,
                            "lower_fraction": low_frac,
                            "finding": (
                                f"monotonicity_violation: {pose}/{higher} ({high_frac:.3f}) is below "
                                f"{pose}/{lower} ({low_frac:.3f}) at {tol:g}px on the labelled subset"
                            ),
                        }
                    )
    return violations


def build_document(
    surface_path: Path,
    acc: SurfaceAccumulator,
    tolerances: tuple[float, ...],
    min_n: int,
    labels_path: Path | None,
    reference: GroundTruthReference | None,
    predictions: np.ndarray | None,
    predictions_path: Path | None,
    predictions_reason: str | None,
    accuracy: dict[str, dict[str, object]],
    join_diagnostics: dict[str, object] | None,
    violations: list[dict[str, object]],
) -> dict[str, object]:
    """Assemble the reliability document, enforcing the no-accuracy-without-labels contract."""
    tiers: dict[str, object] = {}
    calibrated_tiers = 0
    for key in sorted(acc.counts):
        pose, presence = key.split("|", 1)
        n = acc.counts[key]
        signals = {
            name: summary
            for name in NUMERIC_SIGNALS
            if (summary := signal_summary(acc.signals[key][name])) is not None
        }
        flags = {name: dict(counter) for name, counter in acc.flags[key].items()}
        tier_acc: object = None
        if reference is not None and predictions is not None:
            if key in accuracy:
                tier_n = int(accuracy[key][str(tolerances[0])]["n"])  # type: ignore[index]
                if tier_n >= min_n:
                    tier_acc = accuracy[key]
                    calibrated_tiers += 1
                else:
                    tier_acc = {
                        "status": "insufficient_evidence",
                        "n": tier_n,
                        "min_n": min_n,
                    }
            else:
                tier_acc = {"status": "insufficient_evidence", "n": 0, "min_n": min_n}
        tiers[key] = {
            "pose_quality": pose,
            "presence_confidence": presence,
            "n": n,
            "share": (n / acc.total) if acc.total else 0.0,
            "evidence": "distribution_only",
            "signals": signals,
            "flags": flags,
            "accuracy": tier_acc,
        }

    if reference is None:
        calibrated = False
        reason = (
            "user_labels_absent: no --labels were supplied, so every number here is a distribution "
            "statistic, not a reliability claim; no accuracy fraction is emitted"
        )
    elif predictions is None:
        calibrated = False
        reason = predictions_reason or "predictions_absent: no prediction bundle for the labelled clip"
    elif calibrated_tiers == 0:
        calibrated = False
        reason = (
            f"insufficient_evidence: labels and predictions are present but no tier reaches "
            f"min_n={min_n} labelled joints, so no tier is calibrated"
        )
    else:
        calibrated = True
        reason = f"calibrated: {calibrated_tiers} tier(s) reach min_n={min_n} labelled joints"

    findings: list[str] = [str(v["finding"]) for v in violations]
    if not calibrated:
        findings.append(f"not_calibrated: {reason}")
    findings.append(
        "distribution_only: signal percentiles describe the corpus, not accuracy; a reliability "
        "fraction exists only where labelled joints clear min_n"
    )

    document: dict[str, object] = {
        "schema": RELIABILITY_SCHEMA,
        "surface": str(surface_path),
        "surface_sha256": sha256_file(surface_path),
        "labels": str(labels_path) if labels_path is not None else None,
        "labels_schema": USER_LABEL_SCHEMA if reference is not None else None,
        "predictions": str(predictions_path) if predictions_path is not None else None,
        "calibrated": calibrated,
        "calibration_reason": reason,
        "min_n": min_n,
        "tolerances_px": list(tolerances),
        "total_instances": acc.total,
        "tiers": tiers,
        "monotonicity": {
            "presence_order": list(PRESENCE_ORDER),
            "checked": reference is not None and predictions is not None,
            "violations": violations,
        },
        "label_join": join_diagnostics,
        "findings": findings,
    }
    return document


def write_tier_csv(path: Path, document: dict[str, object], tolerances: tuple[float, ...]) -> None:
    """Write the flat per-tier table: counts, share, signal medians, and per-tolerance accuracy."""
    tiers = document["tiers"]
    assert isinstance(tiers, dict)
    fields = ["pose_quality", "presence_confidence", "n", "share", "evidence"]
    fields += [f"{name}_p50" for name in NUMERIC_SIGNALS]
    fields += [f"acc_within_{tol:g}px" for tol in tolerances]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(tiers):
            tier = tiers[key]
            row: dict[str, object] = {
                "pose_quality": tier["pose_quality"],
                "presence_confidence": tier["presence_confidence"],
                "n": tier["n"],
                "share": f"{tier['share']:.6f}",
                "evidence": tier["evidence"],
            }
            signals = tier["signals"]
            for name in NUMERIC_SIGNALS:
                summary = signals.get(name)
                row[f"{name}_p50"] = f"{summary['p50']:.6f}" if summary else ""
            acc_block = tier["accuracy"]
            for tol in tolerances:
                cell = ""
                if isinstance(acc_block, dict) and "status" not in acc_block:
                    entry = acc_block.get(str(tol))
                    if isinstance(entry, dict):
                        cell = f"{entry['fraction']:.6f}"
                elif isinstance(acc_block, dict):
                    cell = str(acc_block["status"])
                row[f"acc_within_{tol:g}px"] = cell
            writer.writerow(row)


def _parse_tolerances(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError("--tolerances must be positive finite pixel values, e.g. 5,10,20")
    return values


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on a produced document (calibrated or honestly not)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--surface", required=True, type=Path, help="per-instance quality surface CSV")
    parser.add_argument("--labels", default=None, type=Path, help="crinis.user-labels/2 JSON (optional)")
    parser.add_argument("--out", default=Path("reliability.json"), type=Path, help="output JSON document")
    parser.add_argument("--csv", default=Path("tiers.csv"), type=Path, help="output flat tier table")
    parser.add_argument("--min-n", default=30, type=int, help="minimum labelled joints to calibrate a tier")
    parser.add_argument(
        "--tolerances", default="5,10,20", type=_parse_tolerances, help="comma-separated pixel tolerances"
    )
    args = parser.parse_args(argv)
    tolerances: tuple[float, ...] = args.tolerances
    surface_path: Path = args.surface
    if not surface_path.is_file():
        print(f"error: surface '{surface_path}' does not exist", file=sys.stderr)
        return 2

    reference: GroundTruthReference | None = None
    label_clip: str | None = None
    if args.labels is not None:
        reference = GroundTruthReference.load("user_export", args.labels)
        if reference.video_path is not None:
            label_clip = reference.video_path.stem
        elif reference.index_path is not None:
            label_clip = reference.index_path.stem

    acc = accumulate_surface(surface_path, label_clip)

    predictions: np.ndarray | None = None
    predictions_path: Path | None = None
    predictions_reason: str | None = None
    accuracy: dict[str, dict[str, object]] = {}
    join_diagnostics: dict[str, object] | None = None
    violations: list[dict[str, object]] = []
    if reference is not None and label_clip is not None:
        predictions, predictions_path, predictions_reason = _load_predictions(surface_path, label_clip)
        if predictions is not None:
            accuracy, join_diagnostics = compute_labelled_accuracy(
                reference, predictions, acc.label_lookup, tolerances
            )
            violations = check_monotonicity(accuracy, tolerances, args.min_n)
    elif reference is not None:
        predictions_reason = (
            "predictions_absent: the label file names no clip video, so no prediction bundle can be "
            "selected; no accuracy is claimed"
        )

    document = build_document(
        surface_path,
        acc,
        tolerances,
        args.min_n,
        args.labels,
        reference,
        predictions,
        predictions_path,
        predictions_reason,
        accuracy,
        join_diagnostics,
        violations,
    )
    atomic_write_json(args.out, document)
    write_tier_csv(args.csv, document, tolerances)

    print(f"reliability curve: {acc.total} instances, {len(acc.counts)} tiers → {args.out}")
    print(f"calibrated: {document['calibrated']} — {document['calibration_reason']}")
    for violation in violations:
        print(f"FINDING: {violation['finding']}")
    if not document["calibrated"]:
        print("No accuracy fraction is emitted without calibrated labels (distribution_only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
