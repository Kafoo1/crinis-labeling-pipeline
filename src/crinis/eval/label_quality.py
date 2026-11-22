"""Measure the labeler's own repeatability from two blind passes, with calibrated error bars.

The question "how much more accurate does labelling get us" has no answer until we know the
*ceiling*: how close the same person lands to the same joint twice, without seeing the first
attempt. This module turns paired passes A and B into that ceiling and gives every headline number
a seeded bootstrap interval, so a difference smaller than the interval is not reported as a finding.

Definitions, fixed here because a metric without its aggregation stated is ambiguous:

* **Per-joint agreement** is the 2D Euclidean distance in pixels between the joint in PASS A and the
  same joint in PASS B.
* **ICC(2,k)** comes from a two-way random-effects ANOVA over subjects (joints) x raters (passes),
  with ``k = 2``. It is computed per coordinate - x and y separately - and the reported value is the
  mean of the two. With ``n`` subjects, ``MSR`` the between-subject mean square, ``MSC`` the
  between-rater mean square and ``MSE`` the residual mean square::

      ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n)
      ICC(2,k) = (MSR - MSE) / (MSR + (MSC - MSE)/n)

* **Bland-Altman** per joint: ``bias = mean(A - B)`` for x and y, limits of agreement
  ``bias +- 1.96 * sd`` with the sample standard deviation (``ddof=1``). A sample with fewer than
  two differences has no estimable spread, so its ``sd`` and both limits are ``None`` rather than a
  zero that would read as perfect agreement.
* **Acted-on joints** are the repeatability headline. A joint the labeler left on the pipeline's
  pre-annotation is identical in both passes *by construction*, so counting it would deflate the
  spread (measured: 5.25 px with them, 9.60 px without). A joint counts as placed when either pass
  differs from the pre-annotation by more than :data:`PLACED_TOLERANCE_PX`; untouched joints are
  reported in the totals, never in the headline.
* **Model error per joint** is the mean of the two passes' distances from the pre-annotation,
  ``(|A - P| + |B - P|) / 2``. It is a paired per-joint quantity, so the median and the ratio over
  repeatability both get a bootstrap interval; the old one-off script averaged the two pass medians,
  which is not resamplable as a per-joint statistic.

Every interval is a percentile bootstrap over joints with replacement, :data:`DRAWS` draws and a
fixed seed, so the same XML always produces the same interval. ``numpy`` is the only dependency:
no SciPy, statsmodels or pingouin.

``from_xml`` is a thin adapter over :func:`crinis.eval.evidence.read_hand_xml`, the canonical rich
reader: duplicate ``(frame, side)`` shapes, unknown joint names, non-finite points and contradictory
declared grids are rejected in one place instead of once per consumer. An images task numbers its
frames by position and PASS B stores them in a different order, so the frame is translated back
through the image names from ``data/meta``; ``b03_frame_0045.png`` maps through its last integer
run, 0045. ``from_task`` still reads CVAT directly, because a task id has no exported document.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from crinis.eval.evidence import SIDE_LABELS, read_hand_xml
from crinis.eval.evidence import hand_extent as evidence_hand_extent
from crinis.hand.keypoint_spec import JOINT_NAMES

if TYPE_CHECKING:
    from crinis.labeling.client import CvatClient

#: The CVAT instance the task loader talks to. Credentials come from the environment through the one
#: client; nothing is embedded here.
URL = os.environ.get("CVAT_URL", "http://localhost:8080")


def _client() -> CvatClient:
    """The authenticated client. Built on demand so importing this module needs no token."""
    from crinis.labeling.client import CvatClient

    return CvatClient.from_environment()

#: Bootstrap protocol. Every confidence interval in this module uses these unless a caller overrides.
DEFAULT_SEED = 20261028
DRAWS = 2000
PERCENTILES = (2.5, 97.5)

#: A joint closer than this to the pre-annotation in *both* passes was never moved by the labeler;
#: it is identical in A and B by construction and must not enter the repeatability headline.
PLACED_TOLERANCE_PX = 0.2

#: An edit of more than this (0.01 px) is what the historical guard uses to decide that a pass was
#: labelled at all. Kept separate from the headline threshold so the two questions stay independent.
EDITED_TOLERANCE_PX = 0.01

Hands = dict[tuple[int, str], dict[str, tuple[float, float]]]
PerJoint = dict[str, list[float]]
Pairs = list[tuple[float, float, float, float]]


# ── CVAT reading (lifted unchanged from dist/cvat_repeatability.py) ──────────


def _api(
    method: str,
    path: str,
    token: str | None = None,
    data: bytes | None = None,
) -> Any:  # noqa: ANN401 - CVAT REST answers are heterogeneous JSON, decoded per call site
    request = urllib.request.Request(
        f"{URL}/api/{path.lstrip('/')}",
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Token {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = response.read()
    return json.loads(body) if body else None


def _last_digits(name: str) -> str:
    """The last integer run in a filename: ``'b03_frame_0045.png'`` -> ``'0045'``, not ``'030045'``."""
    runs = re.findall(r"\d+", name)
    return runs[-1] if runs else ""


def from_task(task_id: int) -> Hands:
    """Read one pass straight from CVAT, keyed by the REAL frame number, not the task's image index.

    An images task numbers its frames 0..N by position, and PASS B stores the same frames in a
    different order, so the index must be translated back through the image names from ``data/meta``
    - otherwise every joint is compared against a different frame and the result is nonsense.
    """
    client = _client()
    token = client.token
    parents: dict[int, str] = {}
    names: dict[int, str] = {}
    for label in _api("GET", f"labels?task_id={task_id}", token=token).get("results", []):
        parents[label["id"]] = label["name"]
        for sublabel in label.get("sublabels", []):
            names[sublabel["id"]] = sublabel["name"]
    meta = _api("GET", f"tasks/{task_id}/data/meta", token=token)
    real_frame: dict[int, int] = {}
    for index, image in enumerate(meta.get("frames", [])):
        digits = _last_digits(image.get("name", ""))
        real_frame[index] = int(digits) if digits else index

    annotations = _api("GET", f"tasks/{task_id}/annotations", token=token)
    hands: Hands = {}
    for shape in annotations.get("shapes", []):
        if shape.get("type") != "skeleton":
            continue
        joints: dict[str, tuple[float, float]] = {}
        for element in shape.get("elements", []):
            joint = names.get(element["label_id"])
            points = element.get("points") or []
            if joint and len(points) == 2:
                joints[joint] = (float(points[0]), float(points[1]))
        if joints:
            hands[
                (
                    real_frame.get(
                        int(shape.get("frame", 0)), int(shape.get("frame", 0))
                    ),
                    parents.get(shape["label_id"], ""),
                )
            ] = joints
    if not hands:
        raise SystemExit(
            f"task {task_id} produced no skeleton shapes; pass an exported XML instead, or check "
            "the task id still exists"
        )
    return hands


def from_xml(path: Path) -> Hands:
    """Read one exported CVAT XML, keyed by the frame number encoded in each image filename.

    Reads through :func:`crinis.eval.evidence.read_hand_xml`, so this path and the evaluators agree
    about duplicates, joint names, declared grids and frame numbering. ``hand_unknown`` shapes keep
    their label rather than being folded into a named side.
    """
    document = read_hand_xml(Path(path))
    hands: Hands = {}
    for shape in document.shapes:
        hands[(shape.frame, SIDE_LABELS[shape.side])] = {
            joint: value.xy for joint, value in shape.joints.items()
        }
    return hands


def load(source: str) -> Hands:
    """A path is a file (error if absent); anything else is a CVAT task id."""
    path = Path(source)
    if path.suffix.lower() == ".xml" or path.exists():
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        return from_xml(path)
    if not source.isdigit():
        raise SystemExit(f"expected a task id or an .xml path, got {source!r}")
    return from_task(int(source))


def hand_extent(joints: Mapping[str, tuple[float, float]]) -> float:
    """The largest distance between any two joints of one hand: its pixel size in the frame."""
    return evidence_hand_extent(joints)


def pairwise(
    hands_a: Hands,
    hands_b: Hands,
    spans: Mapping[tuple[int, str], float],
) -> tuple[list[float], PerJoint, list[tuple[int, str]], list[float]]:
    """Return ``(distances_px, per_joint_px, keys, fractions)`` for keys present in both."""
    distances: list[float] = []
    per_joint: PerJoint = {}
    fractions: list[float] = []
    for key in sorted(set(hands_a) & set(hands_b)):
        a, b = hands_a[key], hands_b[key]
        span = spans.get(key) or 1.0
        for joint in JOINT_NAMES:
            if joint not in a or joint not in b:
                continue
            d = ((a[joint][0] - b[joint][0]) ** 2 + (a[joint][1] - b[joint][1]) ** 2) ** 0.5
            distances.append(d)
            per_joint.setdefault(joint, []).append(d)
            fractions.append(d / span)
    return distances, per_joint, sorted(set(hands_a) & set(hands_b)), fractions


# ── Pairing and agreement ────────────────────────────────────────────────────


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def paired_joints(hands_a: Hands, hands_b: Hands) -> Pairs:
    """``(ax, ay, bx, by)`` for every joint present in both passes, in stable order.

    The order is ``sorted(hands)`` x :data:`JOINT_NAMES`, which is what :func:`pairwise` and every
    bootstrap in this module resample from; ``a`` and ``b`` rows stay paired by construction.
    """
    pairs: Pairs = []
    for key in sorted(set(hands_a) & set(hands_b)):
        a, b = hands_a[key], hands_b[key]
        for joint in JOINT_NAMES:
            if joint in a and joint in b:
                pairs.append((a[joint][0], a[joint][1], b[joint][0], b[joint][1]))
    return pairs


# ── ICC(2,k) ─────────────────────────────────────────────────────────────────


def _icc_coordinate(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> tuple[float, float]:
    """One coordinate's ICC(2,1) and ICC(2,k) from the two-way random-effects ANOVA.

    ``a`` and ``b`` are the same ``n`` subjects measured by the two raters/passes. One observation
    per cell, so the residual has ``(n-1)(k-1)`` degrees of freedom with ``k = 2``.
    """
    n = int(a.size)
    if n < 2:
        return float("nan"), float("nan")
    values = np.stack([a, b], axis=1)
    grand = float(values.mean())
    rows = values.mean(axis=1)
    cols = values.mean(axis=0)
    ss_rows = 2.0 * float(((rows - grand) ** 2).sum())
    ss_cols = float(n * ((cols - grand) ** 2).sum())
    ss_total = float(((values - grand) ** 2).sum())
    # Floating point can leave a tiny negative residual for identical passes; zero is the truth.
    ss_error = max(ss_total - ss_rows - ss_cols, 0.0)
    msr = ss_rows / (n - 1)
    msc = ss_cols  # SS_cols / (k - 1) with k = 2
    mse = ss_error / (n - 1)  # SS_error / ((n - 1)(k - 1))
    numerator = msr - mse
    denominator_1 = msr + mse + 2.0 * (msc - mse) / n
    denominator_k = msr + (msc - mse) / n
    if abs(denominator_1) < 1e-12:
        icc_1 = 0.0 if abs(numerator) < 1e-12 else float(np.sign(numerator))
    else:
        icc_1 = numerator / denominator_1
    if abs(denominator_k) < 1e-12:
        # No between-subject variance at all: reliability is undefined, and zero is the honest
        # "nothing here can be distinguished" value rather than a fabricated 1.0.
        icc_k = 0.0 if abs(numerator) < 1e-12 else float(np.sign(numerator))
    else:
        icc_k = numerator / denominator_k
    return float(icc_1), float(icc_k)


def _mean_icc(
    ax: NDArray[np.float64],
    bx: NDArray[np.float64],
    ay: NDArray[np.float64],
    by: NDArray[np.float64],
) -> tuple[float, float]:
    """The reported ICC: the mean of the x and y coordinate ICCs, for (2,1) and (2,k)."""
    icc_1_x, icc_k_x = _icc_coordinate(ax, bx)
    icc_1_y, icc_k_y = _icc_coordinate(ay, by)
    return (
        float(np.nanmean([icc_1_x, icc_1_y])),
        float(np.nanmean([icc_k_x, icc_k_y])),
    )


def icc2k(
    a: Hands,
    b: Hands,
    *,
    seed: int = DEFAULT_SEED,
    draws: int = DRAWS,
) -> dict[str, float]:
    """ICC(2,k) for two passes over the same joints, with a bootstrap 95% interval.

    Returns ``icc_2_1``, ``icc_2_k``, ``ci_lo`` and ``ci_hi``. The interval belongs to
    ``icc_2_k``, the reported headline; it is a percentile bootstrap over subjects (joints) with
    replacement, and the point estimate is the full-sample value.
    """
    pairs = paired_joints(a, b)
    if not pairs:
        raise ValueError("ICC needs at least one joint present in both passes")
    values = np.asarray(pairs, dtype=np.float64)
    icc_1, icc_k = _mean_icc(
        values[:, 0], values[:, 2], values[:, 1], values[:, 3]
    )
    n = int(values.shape[0])
    rng = np.random.default_rng(seed)
    sampled: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, n, size=n)
        sample = values[index]
        _, sampled_k = _mean_icc(
            sample[:, 0], sample[:, 2], sample[:, 1], sample[:, 3]
        )
        if np.isfinite(sampled_k):
            sampled.append(sampled_k)
    if sampled:
        lo, hi = np.percentile(sampled, PERCENTILES)
        ci_lo, ci_hi = float(lo), float(hi)
    else:
        ci_lo = ci_hi = float("nan")
    return {
        "icc_2_1": float(icc_1),
        "icc_2_k": float(icc_k),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


# ── Bland-Altman ─────────────────────────────────────────────────────────────


def bland_altman(
    a: Hands, b: Hands
) -> dict[str, float | tuple[float, float] | int | None]:
    """Per-joint bias and limits of agreement for ``A - B``, one set per coordinate.

    ``sd`` and both limits are ``None`` when fewer than two joints are present in both passes: one
    difference cannot estimate a spread, and writing ``0.0`` would claim perfect agreement.
    """
    values = np.asarray(paired_joints(a, b), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Bland-Altman needs at least one joint present in both passes")
    dx = values[:, 0] - values[:, 2]
    dy = values[:, 1] - values[:, 3]
    bias_x = float(dx.mean())
    bias_y = float(dy.mean())
    loa_x: tuple[float, float] | None = None
    loa_y: tuple[float, float] | None = None
    sd_x: float | None = None
    sd_y: float | None = None
    if dx.size > 1:
        sd_x = float(dx.std(ddof=1))
        sd_y = float(dy.std(ddof=1))
        loa_x = (bias_x - 1.96 * sd_x, bias_x + 1.96 * sd_x)
        loa_y = (bias_y - 1.96 * sd_y, bias_y + 1.96 * sd_y)
    return {
        "bias_x": bias_x,
        "bias_y": bias_y,
        "sd_x": sd_x,
        "sd_y": sd_y,
        "loa_x": loa_x,
        "loa_y": loa_y,
        "n": int(dx.size),
    }


# ── Bootstrap helpers ────────────────────────────────────────────────────────


def bootstrap_median_ci(
    values: Iterable[float],
    seed: int = DEFAULT_SEED,
    draws: int = DRAWS,
) -> tuple[float, float, float]:
    """``(median, ci_lo, ci_hi)`` for *values*, seeded and reproducible."""
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        raise ValueError("bootstrap_median_ci needs at least one value")
    point = float(np.median(data))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, data.size, size=(draws, data.size))
    medians = np.median(data[index], axis=1)
    lo, hi = np.percentile(medians, PERCENTILES)
    return point, float(lo), float(hi)


def bootstrap_ratio_ci(
    numerator: Iterable[float],
    denominator: Iterable[float],
    seed: int = DEFAULT_SEED,
    draws: int = DRAWS,
) -> tuple[float, float, float]:
    """``(ratio, ci_lo, ci_hi)`` for ``median(numerator) / median(denominator)``.

    ``numerator`` and ``denominator`` are paired per joint, so each bootstrap draw resamples the
    same joints from both lists. Draws whose denominator median is zero are discarded; a ratio with
    no measurable noise has no finite interval.
    """
    num = np.asarray(list(numerator), dtype=np.float64)
    den = np.asarray(list(denominator), dtype=np.float64)
    if num.size == 0 or num.size != den.size:
        raise ValueError("bootstrap_ratio_ci needs paired non-empty numerator and denominator")
    num_point = float(np.median(num))
    den_point = float(np.median(den))
    point = num_point / den_point if den_point > 0.0 else float("inf")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, num.size, size=(draws, num.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.median(num[index], axis=1) / np.median(den[index], axis=1)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(ratios, PERCENTILES)
    return point, float(lo), float(hi)


# ── Report ───────────────────────────────────────────────────────────────────


def _p90(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return float(ordered[int(0.9 * (len(ordered) - 1))])


def _empty_icc() -> dict[str, float]:
    nan = float("nan")
    return {"icc_2_1": nan, "icc_2_k": nan, "ci_lo": nan, "ci_hi": nan}


def report(
    a: Hands,
    b: Hands,
    pipeline: Hands,
    *,
    seed: int = DEFAULT_SEED,
    draws: int = DRAWS,
) -> dict[str, object]:
    """Everything the quality report prints, with the acted-on-joint rule applied.

    ``pipeline`` is the pre-annotation both passes started from. A joint is **placed** when either
    pass moved it by more than :data:`PLACED_TOLERANCE_PX`; untouched joints are identical in A and
    B by construction, so they are counted in ``totals`` but never in ``repeatability``.
    """
    paired_keys = sorted(set(a) & set(b))
    common_keys = sorted(set(a) & set(b) & set(pipeline))
    spans = {key: hand_extent(joints) for key, joints in {**a, **b}.items()}

    all_distances, _, _, _ = pairwise(a, b, spans)

    placed_distances: list[float] = []
    placed_by_joint: PerJoint = {}
    untouched_by_joint: dict[str, int] = dict.fromkeys(JOINT_NAMES, 0)
    a_errors: list[float] = []
    b_errors: list[float] = []
    model_values: list[float] = []
    placed_keys_joints: dict[tuple[int, str], set[str]] = {}
    edited_a = 0
    edited_b = 0
    considered = 0

    for key in common_keys:
        a_joints, b_joints, p_joints = a[key], b[key], pipeline[key]
        for joint in JOINT_NAMES:
            if joint not in a_joints or joint not in b_joints or joint not in p_joints:
                continue
            considered += 1
            da = _distance(a_joints[joint], p_joints[joint])
            db = _distance(b_joints[joint], p_joints[joint])
            if da > EDITED_TOLERANCE_PX:
                edited_a += 1
            if db > EDITED_TOLERANCE_PX:
                edited_b += 1
            if da <= PLACED_TOLERANCE_PX and db <= PLACED_TOLERANCE_PX:
                untouched_by_joint[joint] += 1
                continue
            d = _distance(a_joints[joint], b_joints[joint])
            placed_distances.append(d)
            placed_by_joint.setdefault(joint, []).append(d)
            a_errors.append(da)
            b_errors.append(db)
            model_values.append((da + db) / 2.0)
            placed_keys_joints.setdefault(key, set()).add(joint)

    placed = len(placed_distances)
    untouched = considered - placed
    unclassified = len(all_distances) - considered
    extent_keys = common_keys or paired_keys
    hand_extent_px = (
        float(statistics.median([hand_extent(a[key]) for key in extent_keys]))
        if extent_keys
        else 0.0
    )

    repeatability: dict[str, float | int] = {
        "n": placed,
        "median_px": float("nan"),
        "ci_lo_px": float("nan"),
        "ci_hi_px": float("nan"),
        "p90_px": float("nan"),
        "max_px": float("nan"),
        "percent_of_extent": float("nan"),
        "over_20px": 0,
    }
    if placed_distances:
        median_px, ci_lo, ci_hi = bootstrap_median_ci(
            placed_distances, seed=seed, draws=draws
        )
        repeatability.update(
            {
                "median_px": median_px,
                "ci_lo_px": ci_lo,
                "ci_hi_px": ci_hi,
                "p90_px": _p90(placed_distances),
                "max_px": float(max(placed_distances)),
                "percent_of_extent": (
                    100.0 * median_px / hand_extent_px if hand_extent_px > 0.0 else float("nan")
                ),
                "over_20px": sum(1 for value in placed_distances if value > 20.0),
            }
        )

    all_paired: dict[str, float | int] = {
        "n": len(all_distances),
        "median_px": float("nan"),
        "p90_px": float("nan"),
        "max_px": float("nan"),
    }
    if all_distances:
        all_paired.update(
            {
                "median_px": float(statistics.median(all_distances)),
                "p90_px": _p90(all_distances),
                "max_px": float(max(all_distances)),
            }
        )

    per_joint: dict[str, dict[str, float | int]] = {}
    for joint in JOINT_NAMES:
        values = placed_by_joint.get(joint, [])
        per_joint[joint] = {
            "n": len(values),
            "untouched": untouched_by_joint[joint],
            "median_px": float(statistics.median(values)) if values else float("nan"),
            "p90_px": _p90(values) if values else float("nan"),
            "max_px": float(max(values)) if values else float("nan"),
        }

    filtered_a: Hands = {}
    filtered_b: Hands = {}
    for key, joints in placed_keys_joints.items():
        filtered_a[key] = {joint: a[key][joint] for joint in joints}
        filtered_b[key] = {joint: b[key][joint] for joint in joints}

    icc = _empty_icc()
    bland: dict[str, float | tuple[float, float] | int | None] = {
        "bias_x": float("nan"),
        "bias_y": float("nan"),
        "loa_x": (float("nan"), float("nan")),
        "loa_y": (float("nan"), float("nan")),
        "n": 0,
    }
    if placed >= 2:
        icc = icc2k(filtered_a, filtered_b, seed=seed, draws=draws)
        bland = bland_altman(filtered_a, filtered_b)

    model: dict[str, float | int] = {
        "n": len(model_values),
        "median_px": float("nan"),
        "ci_lo_px": float("nan"),
        "ci_hi_px": float("nan"),
        "a_median_px": float("nan"),
        "b_median_px": float("nan"),
    }
    ratio: dict[str, float] = {
        "value": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
    }
    if model_values:
        median_px, ci_lo, ci_hi = bootstrap_median_ci(
            model_values, seed=seed, draws=draws
        )
        model.update(
            {
                "median_px": median_px,
                "ci_lo_px": ci_lo,
                "ci_hi_px": ci_hi,
                "a_median_px": float(statistics.median(a_errors)),
                "b_median_px": float(statistics.median(b_errors)),
            }
        )
        value, ratio_lo, ratio_hi = bootstrap_ratio_ci(
            model_values, placed_distances, seed=seed, draws=draws
        )
        ratio.update({"value": value, "ci_lo": ratio_lo, "ci_hi": ratio_hi})

    threshold_percent = 2.0
    threshold_icc = 0.95
    percent = float(repeatability["percent_of_extent"])
    icc_value = float(icc["icc_2_k"])
    checks = {
        "repeatability": bool(np.isfinite(percent) and percent <= threshold_percent),
        "icc": bool(np.isfinite(icc_value) and icc_value >= threshold_icc),
    }

    return {
        "hands": len(paired_keys),
        "frames": sorted({key[0] for key in paired_keys}),
        "totals": {
            "paired": len(all_distances),
            "considered": considered,
            "placed": placed,
            "untouched": untouched,
            "unclassified": unclassified,
        },
        "edited": {"a": edited_a, "b": edited_b},
        "hand_extent_px": hand_extent_px,
        "repeatability": repeatability,
        "all_paired": all_paired,
        "icc": icc,
        "bland_altman": bland,
        "model": model,
        "ratio": ratio,
        "per_joint": per_joint,
        "thresholds": {
            "repeatability_percent": threshold_percent,
            "icc_2_k": threshold_icc,
        },
        "checks": checks,
        "rule": (
            "placed = either pass differs from the pipeline pre-annotation by more than "
            f"{PLACED_TOLERANCE_PX} px; untouched joints are identical in A and B by construction "
            "and are counted in totals, never in the repeatability headline"
        ),
    }
