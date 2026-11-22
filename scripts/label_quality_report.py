"""Print the calibrated label-quality report for two blind passes.

This is the readable face of ``crinis.eval.label_quality``: per-joint agreement, a seeded bootstrap
interval for every headline number, ICC(2,k), Bland-Altman, and the acted-on-joint rule that keeps
untouched model joints out of the repeatability headline.

    python dist/label_quality_report.py --a 15 --b 16 --pipeline <exported-annotations.xml>

PASS A and PASS B may each be a CVAT task id or an exported XML. Exits 0 even when a check fails:
FAIL is the measurement, not an error. A loader failure (missing file, unknown task) is not caught.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from crinis.eval.label_quality import load, report
from crinis.hand.keypoint_spec import JOINT_NAMES


def _fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _signed(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}f}"


def _ci(lo: float, hi: float, digits: int = 2) -> str:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "n/a"
    return f"[95% CI {lo:.{digits}f}, {hi:.{digits}f}]"


def format_report(data: dict[str, Any]) -> str:
    """Render the dict from :func:`crinis.eval.label_quality.report` as readable lines."""
    totals = data["totals"]
    repeat = data["repeatability"]
    all_paired = data["all_paired"]
    icc = data["icc"]
    bland = data["bland_altman"]
    model = data["model"]
    ratio = data["ratio"]
    per_joint = data["per_joint"]
    checks = data["checks"]
    thresholds = data["thresholds"]
    extent = float(data["hand_extent_px"])

    paired = int(totals["paired"])
    considered = int(totals["considered"])
    placed = int(totals["placed"])
    untouched = int(totals["untouched"])
    unclassified = int(totals["unclassified"])
    percent_placed = 100.0 * placed / paired if paired else 0.0

    lines = [
        "label quality: PASS A vs PASS B",
        f"hands compared: {int(data['hands'])} | frames: {data['frames']}",
        "",
        f"joints placed: {placed} of {paired} ({percent_placed:.0f}%) | "
        f"untouched (left on the model): {untouched}",
    ]
    if considered != paired or unclassified:
        lines.append(
            f"  considered (present in the pre-annotation): {considered} | "
            f"unclassified (missing from the pre-annotation): {unclassified}"
        )

    median_px = float(repeat["median_px"])
    lines.append(
        f"repeatability (placed joints): median {_fmt(median_px)} px "
        f"{_ci(float(repeat['ci_lo_px']), float(repeat['ci_hi_px']))} = "
        f"{_fmt(float(repeat['percent_of_extent']))}% of {_fmt(extent, 1)} px hand extent | "
        f"p90 {_fmt(float(repeat['p90_px']))} px | max {_fmt(float(repeat['max_px']))} px"
    )
    lines.append(
        f"repeatability (all {int(all_paired['n'])} paired joints, untouched included): "
        f"median {_fmt(float(all_paired['median_px']))} px "
        "- deflated by construction, context only"
    )

    lines.append(
        f"ICC(2,k) {_fmt(float(icc['icc_2_k']))} "
        f"{_ci(float(icc['ci_lo']), float(icc['ci_hi']))} | "
        f"ICC(2,1) {_fmt(float(icc['icc_2_1']))}"
    )

    loa_x = bland["loa_x"]
    loa_y = bland["loa_y"]
    lines.append(
        f"Bland-Altman (A - B, placed joints): bias x {_signed(float(bland['bias_x']))} px, "
        f"y {_signed(float(bland['bias_y']))} px | "
        f"LoA x [{_fmt(float(loa_x[0]))}, {_fmt(float(loa_x[1]))}] px, "
        f"y [{_fmt(float(loa_y[0]))}, {_fmt(float(loa_y[1]))}] px | n {int(bland['n'])}"
    )

    lines.append(
        f"model-vs-human: median {_fmt(float(model['median_px']))} px "
        f"{_ci(float(model['ci_lo_px']), float(model['ci_hi_px']))} "
        f"(A {_fmt(float(model['a_median_px']))} px, "
        f"B {_fmt(float(model['b_median_px']))} px) | n {int(model['n'])}"
    )

    ratio_value = float(ratio["value"])
    if math.isfinite(ratio_value):
        ratio_text = (
            f"{ratio_value:.2f}x "
            f"{_ci(float(ratio['ci_lo']), float(ratio['ci_hi']))}"
        )
    else:
        ratio_text = "n/a (no measurable repeatability)"
    lines.append(f"model error / human repeatability: {ratio_text}")

    lines.extend(["", "per-joint (placed joints only; untouched ones stay with the model):"])
    lines.append(
        f"  {'joint':<12} {'placed':>7} {'untouched':>10} {'median':>9} {'p90':>9} {'max':>9}"
    )
    for joint in JOINT_NAMES:
        row = per_joint.get(joint, {})
        count = int(row.get("n", 0))
        if count:
            lines.append(
                f"  {joint:<12} {count:>7} {int(row.get('untouched', 0)):>10} "
                f"{_fmt(float(row['median_px'])):>9} {_fmt(float(row['p90_px'])):>9} "
                f"{_fmt(float(row['max_px'])):>9}"
            )
        else:
            lines.append(
                f"  {joint:<12} {0:>7} {int(row.get('untouched', 0)):>10} "
                f"{'-':>9} {'-':>9} {'-':>9}"
            )

    verdict_repeat = "PASS" if checks["repeatability"] else "FAIL"
    verdict_icc = "PASS" if checks["icc"] else "FAIL"
    lines.extend(
        [
            "",
            f"repeatability median {_fmt(median_px)} px = "
            f"{_fmt(float(repeat['percent_of_extent']))}% of hand extent "
            f"(limit {float(thresholds['repeatability_percent']):.2f}%): {verdict_repeat}",
            f"ICC(2,k) {_fmt(float(icc['icc_2_k']))} "
            f"(threshold {float(thresholds['icc_2_k']):.2f}): {verdict_icc}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeatability, ICC(2,k), Bland-Altman and model error for two blind passes."
    )
    parser.add_argument("--a", required=True, help="PASS A: task id or exported xml")
    parser.add_argument("--b", required=True, help="PASS B: task id or exported xml")
    parser.add_argument("--pipeline", required=True, help="the pre-annotation XML both passes started from")
    args = parser.parse_args()

    hands_a = load(args.a)
    hands_b = load(args.b)
    pipeline = load(args.pipeline)
    print(format_report(report(hands_a, hands_b, pipeline)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
