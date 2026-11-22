# Hand label protocol

A label is a measurement. Two passes of the same ten frames by the same person disagreed by a median
**9.6 px** (4.2 % of hand extent) with **32 % of placed joints more than 20 px apart**, and the
disagreement was *not* concentrated at the frame edge. That spread is what the rules below exist to
remove: most of it is ambiguity this document has so far failed to resolve, not carelessness.

Measured on: tasks 15/16, 10 frames, 19 hands, 399 joints, against the `clip_0001` pre-annotation.
Numbers and method: `dist/label_quality_report.py --a 15 --b 16 --pipeline <pre-annotation xml>`.

The statistic this document's opening paragraph quotes is **`CORRECTION AGREEMENT`**, not an inter-rater
study. The two passes were *anchored corrections against the same pre-annotation*: the second pass started
from the first, so the number bounds **how closely one person reproduces another's edits**, including the
edits both made because the seed invited them. It is not independent inter-rater repeatability (that needs
two blinded reads of the same frames) and it is not an accuracy benchmark against any reference. Tools print
`CORRECTION AGREEMENT` when a seed is supplied and `INTER-RATER AGREEMENT` only when it is not
(`dist/inter_rater.py --seed-labels <pre-annotation>`).

## 1. Resolution

Label at **1920x1080** (the native source), not the 1280x720 pilot clips. Placement noise is roughly
constant in what the eye can resolve, so the same care buys ~1.5x smaller error as a fraction of the
hand. `dist/cvat_native_res.py` builds the native-resolution pair; the 720p tasks stay as pilots and
their numbers are not comparable to native-resolution results.

## 2. Visibility — mark it, never guess it

Every joint carries a `visibility` attribute. Set it honestly; it is what separates "the two passes
disagree" from "there was nothing to see".

| value | meaning | where to put the joint |
|---|---|---|
`2` | joint is visible in the frame | on the joint |
`1` | joint is outside the image | on the frame edge, on the line of the bone that leaves the frame |
`0` | joint is occluded (by an object, another hand, or the body) | at the last position where that finger segment is visible |

A joint that is `0` or `1` is **excluded from the accuracy statistics** and counted separately. Do not
invent a position to make a skeleton look complete.

## 3. Missing hands

- If a hand is visible and has no skeleton: **draw it** (`Draw a skeleton`), all 21 joints, and say so
  in the job — a missing hand is a recall failure of the pre-annotation, and recording it is worth
  more than a clean-looking task. Measured example: video frame **69** has no right hand in
  `clip_0001`, confirmed by eye.
- If no hand is visible in the frame: leave it absent. Never draw a skeleton on an object.

## 4. More than two hands

If a frame shows more than two skeletons, they are duplicates — the pre-annotation never emits more
than one per side (verified across all passes: maximum 2). **Keep the pair closest to the previous
frame's hand positions and delete the rest.** Do not merge them.

## 5. Frames are independent

Do not smooth, copy from a neighbouring frame, or interpolate. Each frame is a separate measurement,
and copying a value between frames manufactures agreement that the statistics would report as
precision.

## 6. Save

`Ctrl+S` per job. CVAT keeps edits in the browser until you do — a job that was never saved reads back
byte-identical to the pre-annotation (that is how an entire pass was lost once). Confirm the Save
button is no longer highlighted before closing the tab.

## 7. Adjudication, not averaging

Where two passes disagree by more than 20 px on a `visibility=2` joint, a **third, independent** person
places the joint once and that value wins. Averaging two guesses produces a position neither rater
chose and hides the ambiguity that produced it. If the adjudicator also cannot decide, the joint is
mis-specified: mark it `0` and log it, rather than forcing a coordinate.

## 8. What this protocol cannot fix

Absolute accuracy is still unmeasured — there is no ground truth on this machine (HOT3D-Clips ships
hand ground truth but its download needs a CDN manifest from the project; a printed ChArUco board or a
marker rig would serve). Until one exists, every number here is *agreement*, and agreement is an upper
bound on precision that says nothing about bias: two passes can be wrong together.

## 9. Pass criteria for a label set

- human repeatability median **<= 2.0 % of hand extent** on `visibility=2` joints;
- Bland-Altman limits of agreement within **+/- 5 % of hand extent**, and a **systematic bias of |bias| <= 1 %**
  (a bias is a rule being applied differently between passes, and it does not average out);
- every joint flagged `0` or `1` accounted for, never silently placed.

Below those, the label set is not measuring what it claims and no model comparison built on it means
anything.

**ICC is deliberately not a criterion.** It measures whether joints can be *ranked*, and on this data
it scores **0.99 [0.99, 1.00]** while the same joint lands **9.6 px [8.0, 12.3]** apart in two passes
and the limits of agreement span **+/- 35 px**. Those two figures are the spread of **those two seeded
correction passes** — the same pre-annotation edited twice — and they are **not a universal instrument
floor**. In particular, "below the floor" does not mean "accurate": two passes can reproduce one seed,
and each other, while both being wrong about the joint. Between-joint variance (a wrist and a fingertip
are 200 px apart) swamps the disagreement ICC is supposed to detect, so ICC passes any tolerance and
cannot fail a label set that PCK@0.1 would also wave through. Localisation needs the error itself, in px
and as a fraction of hand extent, with its interval — not a correlation.
