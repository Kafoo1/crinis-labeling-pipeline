"""Paired agreement between two CVAT hand-skeleton exports.

The first pass is the baseline (``a``) and the correction is ``b``; every difference is reported as
``b - a`` so a positive x bias means the second annotator put the joint to the right of the first.
Hands are matched by ``(frame, side)``; a hand that exists on only one side is reported as an
unmatched finding rather than silently dropped, because a missing hand is exactly the kind of
disagreement this instrument exists to expose.

Both documents are read through :func:`crinis.eval.evidence.read_hands`, the canonical rich reader:
a duplicate ``(frame, side)`` shape, an unknown joint name, a non-finite point or two contradictory
declared image sizes are rejected here for the same reason they are rejected by the evaluators.
Both CVAT hand formats are accepted per file - images (``<image><skeleton>``) and video
(``<track><skeleton>``) - and joints are compared by their names, so the two files may list them in
any order. Partial skeletons are compared on the joints both passes place; the per-joint table
reports that count instead of a padded zero.

The two exports may be drawn on different pixel grids: the frozen eval labels are 1280x720, while
the second-annotator task is built on native 1920x1080 frames. When both files declare a size, both
are rescaled - uniformly - to the finer grid before differencing. A non-uniform factor would be a
crop rather than a resize, and is refused. If a file declares no size, the pair is compared 1:1 and
the report says so.

What the number is, and what it is not:

* ``INTER-RATER AGREEMENT`` is printed for two reads that are not known to be seeded.
* ``CORRECTION AGREEMENT`` is printed when ``--seed-labels`` is supplied. The two passes were then
  *anchored corrections against the same pre-annotation*, so the statistic bounds how closely one
  person reproduces another's edits; it is not independent inter-rater repeatability, and it is not
  an accuracy benchmark.
* A **video** seed export carries no ``<image>`` and no ``<original_size>``, so it declares no grid
  even when the passes it seeded do. ``--seed-grid WxH`` states that grid explicitly; the seed's
  coordinates are then scaled by one uniform factor like any declared grid, and the same aspect
  refusal applies. Without the flag the historical refusal stands, and a seed that declares its own
  grid that disagrees with ``--seed-grid`` is refused rather than silently overridden.
* The seeded strata are per **joint**: a joint enters ``edited_by_either`` when either pass moved it
  from the seed by more than ``--edited-px`` (default 0.5 px), ``edited_by_both`` when both did, and
  ``unchanged`` when neither did. Untouched joints keep the pre-annotation value in both passes and
  are identical by construction, so letting them into the headline deflates the spread; the rule
  excludes CVAT's sub-pixel export rounding without hiding a real edit.
* Bland-Altman is reported as signed coordinate differences ``B - A`` with the sample SD
  (``ddof=1``) and ``bias +- 1.96 SD`` limits. Fewer than two differences has no estimable spread,
  so its limits are ``None`` - never ``0.0``.
* The interval on the median is a seeded percentile bootstrap that resamples **source frames**, not
  hands or joints: a hand's joints are correlated with each other and both hands in a frame share
  lighting and blur, so only the frame is an independent unit. The spread it reports is the spread
  between these two reads; it is **not a universal measurement floor**.

    python dist/inter_rater.py --a .omp/qa/suite_clip1_labels/annotations.xml \\
        --b .omp/qa/second_annotator/annotations_export.xml \\
        [--seed-labels .omp/qa/second_annotator/annotations_prefilled.xml] \\
        [--out .omp/qa/inter_rater.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from crinis.eval import evidence
from crinis.hand.keypoint_spec import JOINT_NAMES

HAND_LABELS: tuple[str, str] = ("hand_left", "hand_right")
NUM_JOINTS = len(JOINT_NAMES)
SEED = 20261104
RESAMPLES = 4000
BOOTSTRAP_UNIT = "source_frame"

#: A joint is **edited** when a pass moved it from the seed by more than this many pixels. Untouched
#: joints are copied from the pre-annotation, and the CVAT round-trip preserves that to its 3-decimal
#: export precision (~0.005 px), so a sub-half-pixel rule would call every joint edited and deflate
#: the headline towards the all-joint spread. ``0.5`` is the rule the archived 9.60 px measurement
#: used; ``--edited-px`` overrides it, and ``crinis.eval.label_quality`` uses the tighter 0.2 px.
EDITED_PX = 0.5

Hands = dict[tuple[int, str], NDArray[np.float64]]
HandKey = tuple[int, str]
JointArray = NDArray[np.float64]

_JOINT_INDEX = {joint: index for index, joint in enumerate(JOINT_NAMES)}
_FRAME_RE = re.compile(r"frame[_-]?(\d+)")
_DIGITS_RE = re.compile(r"\d+")


def sha256_digest(path: Path) -> str:
    """The SHA-256 of a file's exact bytes. Read-only: the file is opened by ``read_bytes``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_number(name: str, fallback: int) -> int:
    """The frame index encoded in an image name, or *fallback* when the name carries none.

    ``00_frame_0009.png`` -> 9; a name with no ``frame`` marker uses its last integer run, which is
    the convention :mod:`crinis.labeling.cvat` reads for exported images.
    """
    match = _FRAME_RE.search(name)
    if match is not None:
        return int(match.group(1))
    runs = _DIGITS_RE.findall(name.rsplit(".", 1)[0])
    return int(runs[-1]) if runs else fallback


def load_hands(path: Path) -> Hands:
    """Read either CVAT hand format into ``{(frame, hand_label): (21, 2) array}``.

    This is the array form the task builders consume, so it requires all 21 named joints; a partial
    skeleton raises rather than becoming an array padded with zeros. ``compare`` does not go
    through this function - it scores the joints both passes actually place.
    """
    source = Path(path)
    document = evidence.read_hands(source)
    hands: Hands = {}
    for shape in document.shapes:
        missing = [joint for joint in JOINT_NAMES if joint not in shape.joints]
        if missing:
            raise ValueError(
                f"{source}: frame {shape.frame} {evidence.SIDE_LABELS[shape.side]} has "
                f"{len(JOINT_NAMES) - len(missing)} of {len(JOINT_NAMES)} joints "
                f"(missing {', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''})"
            )
        array = np.empty((NUM_JOINTS, 2), dtype=np.float64)
        for joint, value in shape.joints.items():
            array[_JOINT_INDEX[joint]] = value.xy
        hands[(shape.frame, evidence.SIDE_LABELS[shape.side])] = array
    return hands


def _hands_by_side(
    document: evidence.HandDocument,
) -> dict[tuple[int, str], dict[str, tuple[float, float]]]:
    """One document keyed by ``(frame, side)`` with joint names preserved."""
    return {
        (shape.frame, shape.side): {joint: value.xy for joint, value in shape.joints.items()}
        for shape in document.shapes
    }


@dataclass(frozen=True)
class _Pair:
    """One joint present in both passes, on the common grid, as signed ``B - A``."""

    frame: int
    side: str
    joint: str
    dx: float
    dy: float
    distance: float


def _pairs(
    hands_a: Mapping[HandKey, Mapping[str, tuple[float, float]]],
    hands_b: Mapping[HandKey, Mapping[str, tuple[float, float]]],
    keys: Sequence[HandKey],
    *,
    scale_a: float,
    scale_b: float,
) -> list[_Pair]:
    pairs: list[_Pair] = []
    for key in keys:
        joints_a, joints_b = hands_a[key], hands_b[key]
        for joint in JOINT_NAMES:
            if joint not in joints_a or joint not in joints_b:
                continue
            ax, ay = joints_a[joint]
            bx, by = joints_b[joint]
            dx = bx * scale_b - ax * scale_a
            dy = by * scale_b - ay * scale_a
            pairs.append(
                _Pair(
                    frame=key[0],
                    side=key[1],
                    joint=joint,
                    dx=dx,
                    dy=dy,
                    distance=float(math.hypot(dx, dy)),
                )
            )
    return pairs


def _grid_for(
    size_a: tuple[int, int] | None, size_b: tuple[int, int] | None
) -> tuple[float, float, tuple[int, int] | None]:
    """Uniform scales that bring both files onto the finer declared grid.

    ``(1.0, 1.0, None)`` means at least one file declares no size, so the pair is compared 1:1 and
    the report says so. A non-similar aspect is refused by ``grid_scales``: independently scaling x
    and y would turn two different crops into a plausible-looking distance.
    """
    if size_a is None or size_b is None:
        return 1.0, 1.0, None
    target = size_a if size_a[0] * size_a[1] >= size_b[0] * size_b[1] else size_b
    scale_a, _ = evidence.grid_scales(size_a, target)
    scale_b, _ = evidence.grid_scales(size_b, target)
    return scale_a, scale_b, target


def _axis_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Bias, sample SD and 95% limits of agreement for one axis' signed differences."""
    if not values:
        return {"bias_px": None, "sd_px": None, "loa_low_px": None, "loa_high_px": None}
    data = np.asarray(values, dtype=np.float64)
    bias = float(data.mean())
    if data.size < 2:
        return {"bias_px": bias, "sd_px": None, "loa_low_px": None, "loa_high_px": None}
    spread = float(data.std(ddof=1))
    return {
        "bias_px": bias,
        "sd_px": spread,
        "loa_low_px": bias - 1.96 * spread,
        "loa_high_px": bias + 1.96 * spread,
    }


def _per_joint(pairs: Sequence[_Pair]) -> dict[str, dict[str, Any]]:
    """One entry per joint name, in :data:`JOINT_NAMES` order, with n=0 when absent."""
    table: dict[str, dict[str, Any]] = {}
    for joint in JOINT_NAMES:
        selected = [pair for pair in pairs if pair.joint == joint]
        distances = [pair.distance for pair in selected]
        table[joint] = {
            "n": len(selected),
            "median_px": float(np.median(distances)) if distances else None,
            "p90_px": float(np.percentile(distances, 90)) if distances else None,
            "x": _axis_summary([pair.dx for pair in selected]),
            "y": _axis_summary([pair.dy for pair in selected]),
        }
    return table


def _stratum(pairs: Sequence[_Pair], keys: Sequence[HandKey]) -> dict[str, Any]:
    distances = [pair.distance for pair in pairs]
    return {
        "n_hands": len(keys),
        "n_joints": len(distances),
        "median_px": float(np.median(distances)) if distances else None,
        "p90_px": float(np.percentile(distances, 90)) if distances else None,
    }


JointKey = tuple[int, str, str]


def _edited_flags(
    hands_a: Mapping[HandKey, Mapping[str, tuple[float, float]]],
    hands_b: Mapping[HandKey, Mapping[str, tuple[float, float]]],
    seed: Mapping[HandKey, Mapping[str, tuple[float, float]]],
    keys: Sequence[HandKey],
    *,
    scale_a: float,
    scale_b: float,
    seed_scale: float,
    edited_px: float,
) -> dict[JointKey, tuple[bool, bool]]:
    """Whether each pass moved this joint from the seed by more than *edited_px*, per joint.

    The unit is the joint, not the hand: one dragged wrist must not pull its 20 untouched neighbours
    into the edited stratum, because those neighbours are identical in both passes by construction
    and would deflate the spread. A joint missing from either pass or from the seed is absent here
    and is already excluded from every stratum for the same reason it is excluded from scoring.
    """
    flags: dict[JointKey, tuple[bool, bool]] = {}
    for key in keys:
        joints_a = hands_a.get(key)
        joints_b = hands_b.get(key)
        seed_joints = seed.get(key)
        if joints_a is None or joints_b is None or seed_joints is None:
            continue
        for joint in JOINT_NAMES:
            if joint not in joints_a or joint not in joints_b or joint not in seed_joints:
                continue
            ax, ay = joints_a[joint]
            bx, by = joints_b[joint]
            sx, sy = seed_joints[joint]
            moved_a = math.hypot(ax * scale_a - sx * seed_scale, ay * scale_a - sy * seed_scale)
            moved_b = math.hypot(bx * scale_b - sx * seed_scale, by * scale_b - sy * seed_scale)
            flags[(key[0], key[1], joint)] = (moved_a > edited_px, moved_b > edited_px)
    return flags


def _seed_scale(seed_grid: tuple[int, int] | None, target: tuple[int, int] | None) -> float:
    if target is None:
        return 1.0
    if seed_grid is None:
        raise ValueError(
            "the seed document declares no image size, so its coordinates cannot be placed on the "
            "common grid"
        )
    scale, _ = evidence.grid_scales(seed_grid, target)
    return scale


def _seed_grid_used(
    declared: tuple[int, int] | None,
    override: tuple[int, int] | None,
    path: Path,
) -> tuple[int, int] | None:
    """The grid the seed's coordinates are read on.

    ``--seed-grid`` answers the one question a video export cannot: the grid the seed was drawn on.
    It is an override only when the document declares nothing - a disagreeing declaration is a
    contradiction, not a default - so both being present and different is refused.
    """
    if override is not None:
        if min(override) <= 0:
            raise ValueError(f"--seed-grid {override[0]}x{override[1]} is not positive")
        if declared is not None and declared != override:
            raise ValueError(
                f"{path}: the seed document declares {declared[0]}x{declared[1]} but --seed-grid "
                f"says {override[0]}x{override[1]}; they must agree, because the seed cannot be on "
                "two grids"
            )
        return override
    return declared


def compare(
    path_a: Path,
    path_b: Path,
    *,
    seed: int = SEED,
    resamples: int = RESAMPLES,
    seed_labels: Path | None = None,
    seed_grid: tuple[int, int] | None = None,
    edited_px: float = EDITED_PX,
) -> dict[str, Any]:
    """Score ``b`` against ``a``; every reported difference is ``b - a``.

    Raises ``ValueError`` when both paths resolve to the same file, when no hand can be matched, or
    when the documents cannot be placed on one grid. Unmatched hands on either side are returned as
    findings, not an error. ``seed_grid`` supplies the grid of a seed document that declares none
    (a video export); it never overrides a grid the seed itself declares differently. ``edited_px``
    is the distance from the seed above which a joint counts as acted on by that pass.
    """
    if not math.isfinite(edited_px) or edited_px <= 0.0:
        raise ValueError(f"--edited-px must be a finite positive number, got {edited_px!r}")
    a = Path(path_a)
    b = Path(path_b)
    resolved_a = a.resolve()
    resolved_b = b.resolve()
    if resolved_a == resolved_b:
        raise ValueError(
            f"refusing to compare a file with itself: {a} and {b} both resolve to {resolved_a}"
        )
    document_a = evidence.read_hands(a)
    document_b = evidence.read_hands(b)
    scale_a, scale_b, grid = _grid_for(document_a.grid, document_b.grid)
    hands_a = _hands_by_side(document_a)
    hands_b = _hands_by_side(document_b)
    keys_a = set(hands_a)
    keys_b = set(hands_b)
    matched = sorted(keys_a & keys_b)
    unmatched_a = sorted(keys_a - keys_b)
    unmatched_b = sorted(keys_b - keys_a)
    if not matched:
        raise ValueError(
            f"no matched (frame, side) pairs between {a} and {b}: "
            f"{len(unmatched_a)} unmatched in a, {len(unmatched_b)} unmatched in b"
        )

    pairs = _pairs(hands_a, hands_b, matched, scale_a=scale_a, scale_b=scale_b)
    rows = [
        evidence.JointError(
            frame=pair.frame,
            side=pair.side,
            joint=pair.joint,
            error_px=pair.distance,
            fraction=None,
        )
        for pair in pairs
    ]
    summary = evidence.summarise_rows(rows)
    ci_low, ci_high = evidence.cluster_bootstrap_median(
        rows, statistic="median_px", seed=seed, resamples=resamples
    )
    result: dict[str, Any] = {
        "a": str(a),
        "b": str(b),
        "difference": "b - a",
        "axis_sign": {
            "x": "positive means b is to the right of a",
            "y": "positive means b is below a",
        },
        "grid": list(grid) if grid is not None else None,
        "scales": {"a": scale_a, "b": scale_b},
        "declared_sizes": {
            "a": list(document_a.grid) if document_a.grid is not None else None,
            "b": list(document_b.grid) if document_b.grid is not None else None,
        },
        "matched_pairs": [[frame, evidence.SIDE_LABELS[side]] for frame, side in matched],
        "unmatched_a": [[frame, evidence.SIDE_LABELS[side]] for frame, side in unmatched_a],
        "unmatched_b": [[frame, evidence.SIDE_LABELS[side]] for frame, side in unmatched_b],
        "n_hands": len(matched),
        "n_joints": summary["n_joints"],
        "median_px": summary["median_px"],
        "p90_px": summary["p90_px"],
        "mean_px": summary["mean_px"],
        "bland_altman": {
            "x": _axis_summary([pair.dx for pair in pairs]),
            "y": _axis_summary([pair.dy for pair in pairs]),
        },
        "bootstrap": {
            "seed": seed,
            "resamples": resamples,
            "unit": BOOTSTRAP_UNIT,
            "ci_low_px": ci_low,
            "ci_high_px": ci_high,
        },
        "per_joint": _per_joint(pairs),
        "measurement_kind": "inter_rater",
    }
    if seed_labels is not None:
        seed_document = evidence.read_hands(seed_labels)
        seed_hands = _hands_by_side(seed_document)
        seed_grid_used = _seed_grid_used(seed_document.grid, seed_grid, Path(seed_labels))
        seed_scale = _seed_scale(seed_grid_used, grid)
        flags = _edited_flags(
            hands_a,
            hands_b,
            seed_hands,
            matched,
            scale_a=scale_a,
            scale_b=scale_b,
            seed_scale=seed_scale,
            edited_px=edited_px,
        )

        def stratum(keep: Callable[[tuple[bool, bool]], bool]) -> dict[str, Any]:
            selected = [
                pair
                for pair in pairs
                if (pair.frame, pair.side, pair.joint) in flags
                and keep(flags[(pair.frame, pair.side, pair.joint)])
            ]
            hands = sorted({(pair.frame, pair.side) for pair in selected})
            summary = _stratum(selected, hands)
            if selected:
                rows = [
                    evidence.JointError(
                        frame=pair.frame,
                        side=pair.side,
                        joint=pair.joint,
                        error_px=pair.distance,
                        fraction=None,
                    )
                    for pair in selected
                ]
                low, high = evidence.cluster_bootstrap_median(
                    rows, statistic="median_px", seed=seed, resamples=resamples
                )
                summary["ci_low_px"] = low
                summary["ci_high_px"] = high
            else:
                summary["ci_low_px"] = None
                summary["ci_high_px"] = None
            return summary

        result["measurement_kind"] = "correction_repeatability"
        result["seed_labels"] = str(seed_labels)
        result["seed_grid"] = list(seed_grid_used) if seed_grid_used is not None else None
        result["edited_px"] = edited_px
        result["strata"] = {
            "all": _stratum(pairs, matched),
            "edited_by_either": stratum(lambda moved: moved[0] or moved[1]),
            "edited_by_both": stratum(lambda moved: moved[0] and moved[1]),
            "unchanged": stratum(lambda moved: not moved[0] and not moved[1]),
        }
    return result


def _format_pairs(pairs: list[list[int | str]]) -> str:
    return ", ".join(f"frame {frame} {label}" for frame, label in pairs)


def _axis_text(axis: Mapping[str, float | None]) -> str:
    bias = axis["bias_px"]
    if bias is None:
        return "no joints"
    low, high = axis["loa_low_px"], axis["loa_high_px"]
    if low is None or high is None:
        return f"bias {bias:.3f} px, LoA n/a (n<2)"
    return f"bias {bias:.3f} px LoA [{low:.3f}, {high:.3f}]"


def _print_report(result: Mapping[str, Any]) -> None:
    seeded = result["measurement_kind"] == "correction_repeatability"
    print(f"a: {result['a']}")
    print(f"b: {result['b']}")
    if seeded:
        print(f"seed: {result['seed_labels']}")
        seed_grid = result.get("seed_grid")
        if seed_grid is not None:
            print(f"seed grid: {seed_grid[0]}x{seed_grid[1]}")
        print(
            f"edited: a joint moved by more than {result['edited_px']:g} px from the seed "
            "(sub-pixel export rounding is not an edit)"
        )
    grid = result["grid"]
    if grid is None:
        print("grid: compared 1:1 (at least one file declares no frame size)")
    else:
        print(
            f"grid: {grid[0]}x{grid[1]} "
            f"(a scaled {result['scales']['a']:.3f}x, b scaled {result['scales']['b']:.3f}x)"
        )
    print(f"measurement: {result['measurement_kind']}")
    print("CORRECTION AGREEMENT" if seeded else "INTER-RATER AGREEMENT")
    print(f"matched {result['n_hands']} hand(s): {_format_pairs(result['matched_pairs'])}")
    for side in ("unmatched_a", "unmatched_b"):
        pairs = result[side]
        suffix = f": {_format_pairs(pairs)}" if pairs else ""
        print(f"{side}: {len(pairs)} hand(s){suffix}")
    bland = result["bland_altman"]
    print(
        "Bland-Altman (b - a): "
        f"x {_axis_text(bland['x'])} | y {_axis_text(bland['y'])} | n={result['n_joints']}"
    )
    median, p90 = result["median_px"], result["p90_px"]
    if median is not None and p90 is not None:
        print(
            f"spread between these two reads: {median:.3f} px "
            f"[{result['bootstrap']['ci_low_px']:.3f}, "
            f"{result['bootstrap']['ci_high_px']:.3f}] "
            f"(p90 {p90:.3f}, n={result['n_joints']} joints from {result['n_hands']} hands, "
            "bootstrap over source frames) - not a universal measurement floor"
        )
    if seeded:
        for name, stratum in result["strata"].items():
            if stratum["median_px"] is None:
                print(f"  {name:<16} no paired joints")
                continue
            low, high = stratum.get("ci_low_px"), stratum.get("ci_high_px")
            interval = (
                f"  [{low:.3f}, {high:.3f}]" if low is not None and high is not None else ""
            )
            print(
                f"  {name:<16} median {stratum['median_px']:.3f} px{interval}  "
                f"p90 {stratum['p90_px']:.3f}  "
                f"n={stratum['n_joints']} joints from {stratum['n_hands']} hands"
            )
    else:
        print("no seed given: this pair is not labelled as an anchored correction")


def _grid_argument(raw: str) -> tuple[int, int]:
    """Parse ``WxH`` into a positive ``(width, height)``; argparse reports exit 2 otherwise."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", raw)
    if match is None:
        raise argparse.ArgumentTypeError(f"expected WxH such as 1280x720, got {raw!r}")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"grid {raw!r} must be positive")
    return width, height


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="baseline CVAT hand export")
    parser.add_argument("--b", type=Path, required=True, help="correction CVAT hand export")
    parser.add_argument(
        "--seed-labels",
        type=Path,
        default=None,
        help="the pre-annotation both passes were anchored to; makes this CORRECTION AGREEMENT",
    )
    parser.add_argument(
        "--seed-grid",
        type=_grid_argument,
        default=None,
        metavar="WXH",
        help=(
            "grid the seed coordinates live on, for a video export that declares none; refused "
            "when the seed declares a different grid itself"
        ),
    )
    parser.add_argument(
        "--edited-px",
        type=float,
        default=None,
        metavar="PX",
        help=(
            "distance from the seed above which a joint counts as acted on by a pass "
            f"(default {EDITED_PX:g}); sub-pixel CVAT export rounding is not an edit"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="write the same numbers as JSON")
    args = parser.parse_args(argv)

    edited_px = EDITED_PX if args.edited_px is None else args.edited_px
    try:
        result = compare(
            args.a,
            args.b,
            seed_labels=args.seed_labels,
            seed_grid=args.seed_grid,
            edited_px=edited_px,
        )
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"inter_rater: {exc}", file=sys.stderr)
        return 2

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
