"""Measure the labeler's own repeatability from the two blind passes, and compare it to the model.

Reads each pass either straight from CVAT (pass a task id) or from an exported XML, normalises both
to ``{(frame, label): {joint_name: (x, y)}}``, and reports:

  * within-rater spread - the distance between the same joint in PASS A and PASS B. This is the
    ceiling on dataset accuracy: no label set beats it, however carefully it is clicked;
  * how far the human moved the pre-annotation in each pass (pipeline -> A, pipeline -> B), which is
    the size of the correction actually being applied;
  * the ratio model-error / repeatability - below ~1 the model is already inside human noise, so
    further labelling cannot sharpen these joints, only replace them.

The loaders and the statistics live in ``crinis.eval.label_quality``; this script is the original
terminal view of them. ``dist/label_quality_report.py`` prints the full calibrated report.

    python dist/cvat_repeatability.py --a <task-id|xml> --b <task-id|xml> [--pipeline <xml>]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from crinis.eval.label_quality import (
    EDITED_TOLERANCE_PX,
    hand_extent,
    load,
    pairwise,
    report,
)


def summarise(name: str, distances: list[float], fractions: list[float]) -> str:
    if not distances:
        return f"{name}: no common joints"
    ordered = sorted(distances)
    p90 = ordered[int(0.9 * (len(ordered) - 1))]
    return (
        f"{name}: median {statistics.median(distances):.2f} px "
        f"({100 * statistics.median(fractions):.2f}% hand extent) | p90 {p90:.2f} px | "
        f"max {ordered[-1]:.2f} px | joints {len(distances)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="PASS A: task id or exported xml")
    parser.add_argument("--b", required=True, help="PASS B: task id or exported xml")
    parser.add_argument("--pipeline", help="the pre-annotation XML both passes started from")
    parser.add_argument("--manifest", help="the double-label manifest.json, for hard/representative split")
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    spans = {key: hand_extent(joints) for key, joints in {**a, **b}.items()}

    d_ab, per_joint, keys, f_ab = pairwise(a, b, spans)
    print(summarise("within-rater (A vs B)", d_ab, f_ab))
    if per_joint:
        worst = sorted(per_joint.items(), key=lambda kv: -max(kv[1]))[:5]
        print(
            "  worst joints by max offset:",
            ", ".join(f"{joint} {max(values):.1f} px" for joint, values in worst),
        )

    if args.manifest and Path(args.manifest).exists():
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        hard = set(manifest.get("hard_frames", []))
        easy = set(manifest.get("representative_frames", []))
        for name, subset in (("hard frames", hard), ("representative frames", easy)):
            sub_a = {k: v for k, v in a.items() if k[0] in subset}
            sub_b = {k: v for k, v in b.items() if k[0] in subset}
            d, _, _, f = pairwise(sub_a, sub_b, spans)
            print("  " + summarise(name, d, f))

    if args.pipeline:
        pipeline = load(args.pipeline)
        pa = {k: v for k, v in a.items() if k in pipeline}
        pb = {k: v for k, v in b.items() if k in pipeline}
        d_pa, _, _, f_pa = pairwise(pa, pipeline, spans)
        d_pb, _, _, f_pb = pairwise(pb, pipeline, spans)
        print(summarise("pipeline vs PASS A", d_pa, f_pa))
        print(summarise("pipeline vs PASS B", d_pb, f_pb))

        result = report(a, b, pipeline)
        # The guard asks "was this pass labelled at all", which is a pass-vs-pipeline question and
        # not restricted to joints present in both passes; the historical 0.01 px threshold is kept.
        edited_a = sum(1 for d in d_pa if d > EDITED_TOLERANCE_PX)
        edited_b = sum(1 for d in d_pb if d > EDITED_TOLERANCE_PX)
        print(
            f"  edited joints: A {edited_a}/{len(d_pa)} | B {edited_b}/{len(d_pb)}"
        )
        if not edited_a or not edited_b:
            which = "A" if not edited_a else "B"
            other = "B" if which == "A" else "A"
            print(
                f"\nSTOP: PASS {which} has no edited joints. The A-vs-B line above is therefore PASS "
                f"{other} measured against the unedited model - that is model error, NOT repeatability."
            )
            print(f"  Label and SAVE PASS {which} (Ctrl+S) before this number means anything.")
            print(f"\nhands compared: {len(keys)} | frames: {sorted({k[0] for k in keys})}")
            return 2

        # Untouched joints keep the model's value in both passes, so they are identical by
        # construction and contribute 0 to |A-B|. The headline below is the acted-on subset; the
        # full-set number is printed above and is deflated.
        repeat = result["repeatability"]
        placed = int(repeat["n"])
        median_px = float(repeat["median_px"])
        print(
            f"\nrepeatability on the {placed} joints actually placed "
            f"({100 * placed / max(1, len(d_ab)):.0f}% of all): "
            f"median {median_px:.2f} px "
            f"[{float(repeat['ci_lo_px']):.2f}, {float(repeat['ci_hi_px']):.2f} 95% CI] "
            f"({float(repeat['percent_of_extent']):.2f}% hand extent) | "
            f"p90 {float(repeat['p90_px']):.2f} px | max {float(repeat['max_px']):.2f} px"
        )
        over_20 = int(repeat["over_20px"])
        print(
            f"  over 20 px apart in the two passes: {over_20}/{placed} "
            f"({100 * over_20 / max(1, placed):.0f}%)"
        )

        model = result["model"]
        print(
            f"  model-to-human on those joints: A {float(model['a_median_px']):.2f} px | "
            f"B {float(model['b_median_px']):.2f} px"
        )
        print(
            f"  model-to-human median {float(model['median_px']):.2f} px "
            f"[{float(model['ci_lo_px']):.2f}, {float(model['ci_hi_px']):.2f} 95% CI]"
        )

        ratio = result["ratio"]
        value = float(ratio["value"])
        if math.isfinite(value):
            print(
                f"\nmodel error / human repeatability = {value:.1f}x "
                f"[{float(ratio['ci_lo']):.1f}, {float(ratio['ci_hi']):.1f} 95% CI] "
                f"({float(model['median_px']):.2f} px vs {median_px:.2f} px)"
            )
        else:
            print(
                f"\nmodel error {float(model['median_px']):.2f} px; repeatability has no measurable "
                "spread (one or both passes were never edited)"
            )
        print(
            "  <1 means the model is already inside your clicking noise on these joints; "
            "labelling can only replace the values, not sharpen them."
        )
    print(f"\nhands compared: {len(keys)} | frames: {sorted({k[0] for k in keys})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
