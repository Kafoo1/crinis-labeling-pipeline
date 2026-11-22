# What this dataset does not support

Every column below ships **invalid**, not zero, not a sentinel, not a plausible number. Each names the
instrument that would make it measurable. A dataset ships only if `crinis release check` passes on the
same run that produced it, and only for the profile named in `release/iphone14-ultrawide-1080p.yaml`.

## Supported profile

Base iPhone 14, 0.5x ultrawide, 1920x1080, one fixed capture configuration. Parameters measured on a
different configuration are not reused: `crinis camera profile --run` returns `profile_unknown` for any
run that did not record its config, and `profile_unknown` is never a pass.

## Columns with a claim

| column | state | why it is claimable | reference that proves it |
|---|---|---|---|
`observation.hand.landmarks2d` | measured | 21-joint skeletons against the frozen human reference: median **21.97 px**, p90 66.16, coverage@0.05 hand extent 0.586, coverage@0.10 0.697 (378 joints / 18 hands on the 1920x1080 grid; the reference is 1280x720, scaled x1.5). The 6.15 px figure that used to sit here was the frozen labels against an older pre-annotation, i.e. a correction magnitude, not model error | CVAT label set (anchored corrections); correction agreement 9.60 px over the 263 moved joints is a *statistic about those two passes*, not an instrument floor |
`observation.hand.bbox` | measured | derived from the same joints | same labels |
`observation.hand.observed` | measured | recall measured against the labels, including the frame with no right hand (frame 69) | labels |
`observation.hand.detection_confidence` | measured | the detector's own score, reported as a score and never as a probability of correctness | detector artifact |
frame index, timestamps | measured | frames indexed equals frames decoded, constant fps, no anisotropic resize | `crinis audit frames` |
`observation.segment.phase` | measured | human boundary tags on task 26 against the fixed seven-value vocabulary | task 26 tags |
`observation.contact.any` | measured | human intervals for the active hand; the non-active hand is refused rather than conflated | task 26 tags |
`landmarks3d`, `wrist_pose`, `mano.betas`, `mano.pose` | **estimated** | the lift's output, hand-scoped, no absolute claim | — (see the blocked item below) |

## Columns that ship invalid

| column | reason | instrument that would measure it |
|---|---|---|
metric depth | no reference in frame; the depth model's convention is unvalidated on this footage | board plane at independently measured distances |
`depth_at_hand` | inherits the above | same |
metres-per-unit, camera scale | monocular RGB has no scale without a known dimension in frame; the rig's 20 mm squares cover the rig's own configuration only | rig board solve, closing-loop drift |
object pose, object tracks | no object ground truth; a single view cannot be clicked to millimetre/degree accuracy | HOT3D object GT through `crinis.eval.object_pose` |
`contact_distance_m` | 5,705 of 5,915 values are the `1e6` sentinel; contact *geometry* is unmeasured | per-finger distances against a measured contact event |
`gripper_openness` | the definition was never fixed (normalised aperture vs thumb-index distance vs gripper mapping are different quantities) | define it, then compute from human-labelled joints |
interaction force | ordinary RGB cannot measure force | a force sensor |
IMU | no sensor stream was ever recorded; zero-filled data is not evidence | a capture stack that records IMU |
action progress | normalised clip time is not task progress; milestones were never defined | human-checked milestone boundaries |

## Two metrics this dataset does **not** use

- **PCK@0.1** — 24.4 px at this hand size, which is 2.5x the correction-agreement statistic (9.60 px). It
  scores a 9.6 px system and a 15 px system the same, so it cannot distinguish them.
- **ICC(2,k)** — 0.99 on the same data while the same joint lands 9.6 px apart, because between-joint
  variance (wrist to fingertip is 200+ px) swamps localisation error. It ranks joints; it does not
  measure where they are.

Headline metrics are median and p90 error in px, coverage at 0.05 and 0.10 hand extent, each with a seeded
bootstrap interval.

## Blocker on the hand-truth item

The right-hand defect (radial, per-frame k 1.000-1.298, plus a 9-28 px pose residual) cannot be attributed
to the lifter or to this footage without third-party ground truth. That comparison is **blocked on this
machine**: constructing the lifter's 630M-parameter backbone needs ~2.35 GB contiguous and the host cannot
allocate it (`_ArrayMemoryError` with 0.9 GB free); the alternate GPU runtime is unreachable. Freedom in
the record: no claim is made about which side is at fault until that runs.

## Black gloves defeat every hand detector on this machine (measured 2026-10-31)

On `clip_0002` (black nitrile gloves, iPhone 14 ultrawide) no installed detector localises the gloved
hand, so **every hand column on gloved footage is `invalid`**, whatever the lifter does downstream:

| lane | frames with a hand (of 150) | where it puts the hand |
|---|---|---|
| MediaPipe Tasks landmarker | 5 | on the ungloved hand only |
| WiLoR detector (`detector.pt`) + crop keypoint model | 45 | ~250 px right of the glove, on the waffle's corner |
| fused (both, per-slot union) | 50 | one of the two above |
| YOLO-World zero-shot, prompts `gloved hand` / `bare hand` / `hand` | 0 | nowhere |

The fusion is a genuine gain (+5 frames over the better parent, and 10x over MediaPipe alone), but it
cannot invent a detection the detector does not make. Consequence: on a gloved clip the honest task is a
**frames-only** package - `crinis label export --hands-from off` - not a pre-annotation a human must drag
onto the hand first. `crinis release check` already refuses these runs on `observation.hand.landmarks2d`.

### Translation, and what was rejected

Three mechanisms were built and measured against the only human labels in this repo (`clip_0001`, task 17):

| mechanism | right depth median | bare-hand left median | verdict |
|---|---|---|---|
| model `tz = 2f/(b·s)` | 0.311 m | **6.03 px** (p90 25.11, cov@0.05 0.714) | the reference |
| image anchor by extent ratio | 0.760 m | - | worse: conflates foreshortening with depth; 1.33 m steps |
| image anchor by reprojection fit | 0.321 m | - | refuses 37 of 46 frames on gloves; mixing sources made the track jumpier |
| temporal gate @ 3 m/s | - | 8.66 px (p90 135.17, cov 0.587) | rejected: clamps real motion |
| temporal gate @ 10 m/s | - | 6.12 px (p90 30.57, cov 0.661) | rejected as a default |

So `hands.translation_consistency` defaults to `none` and `hands.translation_source` to `model`: the raw
lift is the most accurate thing measured. The jumping labels that started this were **CVAT interpolating
between sparse keyframes in a video task**, not the lift - which is why the fix is images tasks and track
closing (`_write_xml`), not smoother geometry.

## The crop lane's 2-D geometry is wrong, and that is the open defect (measured 2026-10-31)

`hands.model_2d: wilor*` (detector box -> WiLoR crop -> linear crop-to-frame map) produces hands that are
about **half the size they should be**, so its joints miss the labelled hand by 53-54 px even though the
hand is only ~345 px across:

| lane | frames with a hand (clip_0001) | median joint error vs labels |
|---|---|---|
| mediapipe (whole frame) | 28 slots | 20.90 px (63 shared joints) |
| wilor crop lane | 159 slots | 53.09 px (378 shared joints) |
| fused, mediapipe preferred on shared slots | 187 slots | 54.20 px |

Evidence for the cause, all from `runs/clip1-fused`:

- the crop lane's boxes are **599 px** across at the median (`p90` 822) for a hand whose labelled span is
  ~345 px - a detector box about 1.7x the hand;
- its predicted/labelled extent ratio is **0.54** (sd 0.16), i.e. a scale error that varies per frame;
- mirroring in the box does not explain it (mirrored medians are worse: 95 px and 107 px).

A linear `[-1, 1] -> box` map cannot recover WiLoR's geometry: the demo maps its output through
`cam_crop_to_full` with the capture focal (`wilor/models/wilor.py`, `demo.py`), and the 3-D lane here
already does that correctly - which is why the *projected* 3-D hand agrees to 6.03 px on the left while the
2-D crop lane does not. Fixing this means extending the `KeypointModel` seam to carry the box and the
focal, then projecting the model's own joints; until that lands, the fused lane's extra coverage is a
coarse hint, not a label.

**The fused lane therefore helps coverage (5 -> 50 frames on gloved footage) and hurts 2-D accuracy where
it adds frames. It is not a default and not a certification.**
