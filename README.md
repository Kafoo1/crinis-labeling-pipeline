# crinis-labeling-pipeline

**Video in → human-checked hand labels out → a release gate that refuses to ship what it cannot measure.**

A production labeling pipeline extracted from the [Crinis](https://github.com/kafoo1) research stack:
egocentric hand video captured on a phone becomes a LeRobot v3 dataset with per-frame 21-joint hand
labels. This repository is the labeling half — CVAT automation, the label protocol, agreement
measurement, and the QA gate — with every number below traceable to a named artifact.

![labeling overlay demo](demo/labeling_overlay_demo.mp4)

*60-second demo: pre-annotation vs human-corrected 21-joint tracks on source footage
(`demo/labeling_overlay_demo.mp4`).*

## What it does

1. **Ingest & pre-annotate** — video frames are indexed (frames indexed == frames decoded, constant
   30 fps audit), a hand detector + pose model proposes 21-joint skeletons per frame.
2. **CVAT automation over REST** — `scripts/cvat_upload_video.py` creates video tasks directly from an
   `annotations.xml` label declaration; skeleton, segment and image task builders cover the other
   shapes. No hand-clicked task setup.
3. **Human correction with a written protocol** — `docs/label_protocol.md`: visibility taxonomy
   (visible / out-of-frame / occluded), missing-hand recall rules, no interpolation between frames,
   adjudication-not-averaging for disagreements.
4. **Agreement measurement** — `scripts/inter_rater.py` and `scripts/label_quality_report.py` compute
   paired joint differences with frame-level bootstrap CIs, Bland-Altman limits, and honestly label
   the statistic `CORRECTION AGREEMENT` vs `INTER-RATER AGREEMENT` depending on whether a seed
   pre-annotation was supplied.
5. **QA sampling** — `scripts/mine_hard_frames.py` selects the frames most likely to expose label
   errors; `scripts/reliability_curve.py` measures detector/label reliability against a reference.
6. **Release gate** — every required column returns `pass / fail / invalid / unverified` against a
   named reference; `unverified` never counts as a pass. A dataset that cannot prove a column ships
   it `invalid` with the instrument that would fix it (`docs/unsupported.md`).

## Measured results (iPhone 14, 0.5x ultrawide, 1920x1080, 30 fps)

| what | number | against |
|---|---|---|
| 2D landmark error | median **20.74 px** (p90 63.87) | human second-annotator correction, 378 joints / 18 hands |
| correction agreement | median **9.60 px** over 263/399 joints moved >0.5 px | two CVAT passes, tasks 15/16 |
| hand detection recall | 18 of 19 labelled hands matched; the two misses named | same reference |

These are agreement numbers against human references, not claims of absolute accuracy — the
protocol and reports say exactly which is which, and why ICC alone is deliberately not a pass
criterion.

## Layout

```
src/crinis/labeling/   CVAT XML dialect, REST client, export/ingest with provenance
src/crinis/eval/       evidence reader, label-quality stats, hand gate, ground-truth references
src/crinis/hand/       21-joint keypoint spec
src/crinis/io/         artifact storage, run layout
scripts/               CVAT task builders, double-label passes, agreement reports, QA sampling
docs/                  label_protocol.md, unsupported.md
demo/                  labeled-overlay demo video
```

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate on Linux
pip install numpy pyarrow av opencv-python

# create a CVAT video task from an annotations.xml package
CVAT_URL=https://your-cvat CVAT_TOKEN=... python scripts/cvat_upload_video.py <package-dir>

# measure agreement between two CVAT exports
python scripts/inter_rater.py --a pass_a.xml --b pass_b.xml [--seed-labels pre.xml]
```

## Scope note

This is the labeling/QA subset of a larger research pipeline (inference, 3-D lift, dataset export).
It is published so the labeling methodology and tooling can be inspected and reused on their own.
Code Apache-2.0; no footage, model weights or MANO-derived material is redistributed here.