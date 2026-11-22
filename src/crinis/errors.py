from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    category: str
    summary: str
    remediation: str


ERROR_SPECS: dict[str, ErrorSpec] = {
    # ── Config errors (CRN-0xxx) ────────────────────────────────────────────
    "CRN-0001": ErrorSpec(
        code="CRN-0001",
        category="config",
        summary="One or more config keys are missing, unknown, or have the wrong type.",
        remediation=(
            "Run `crinis config show` to see the merged config and fix the offending key(s)."
        ),
    ),
    "CRN-0002": ErrorSpec(
        code="CRN-0002",
        category="config",
        summary=(
            "video.size contains a dimension that is not a positive even integer "
            "(H.264 yuv420p requires even dimensions)."
        ),
        remediation="Set video.size to a pair of positive even integers, e.g. [640, 480].",
    ),
    "CRN-0003": ErrorSpec(
        code="CRN-0003",
        category="config",
        summary=(
            "A run ID is not a valid identifier, or an artifact reference string "
            "is malformed (expected '<stage>:<name>' format)."
        ),
        remediation=(
            "Run IDs must match ^[0-9A-Za-z._-]{1,64}$. "
            "Artifact references must be in '<stage>:<artifact-name>' form."
        ),
    ),
    "CRN-0004": ErrorSpec(
        code="CRN-0004",
        category="config",
        summary=(
            "A requested stage does not exist, or is planned but not implemented yet."
        ),
        remediation=(
            "Run `crinis run --help` for the implemented stages, or drop the stage from "
            "`--stages`. The plan (PLANS/egocentric-hand-pipeline.md) lists which slice "
            "implements each stage."
        ),
    ),
    # ── Input errors (CRN-1xxx) ─────────────────────────────────────────────
    "CRN-1001": ErrorSpec(
        code="CRN-1001",
        category="input",
        summary="The input video file is missing or cannot be opened for reading.",
        remediation=(
            "Check that the path is correct and the file is readable; "
            "re-copy from the device if needed."
        ),
    ),
    "CRN-1002": ErrorSpec(
        code="CRN-1002",
        category="input",
        summary="ffprobe is absent from PATH and no explicit path was configured.",
        remediation=(
            "Install ffmpeg (which includes ffprobe) and ensure it is on PATH, "
            "or set `ffmpeg.ffprobe_path` in config."
        ),
    ),
    "CRN-1003": ErrorSpec(
        code="CRN-1003",
        category="input",
        summary="The input file contains no video stream or decodes to zero frames.",
        remediation=(
            "Verify the file is a valid video with `ffprobe -v error -show_streams <file>`."
        ),
    ),
    "CRN-1004": ErrorSpec(
        code="CRN-1004",
        category="input",
        summary=(
            "IMU file is absent; observation.imu will be zero-filled and gravity "
            "alignment disabled."
        ),
        remediation=(
            "Export CoreMotion data from the iPhone app and place the CSV/JSON "
            "at the configured imu.path."
        ),
    ),
    "CRN-1005": ErrorSpec(
        code="CRN-1005",
        category="input",
        summary=(
            "The IMU file format is not recognised "
            "(not Sensor Logger, SensorLog, or Phyphox)."
        ),
        remediation=(
            "Check the file header and convert to a supported format; "
            "see `crinis doctor` for the expected column names."
        ),
    ),
    "CRN-1006": ErrorSpec(
        code="CRN-1006",
        category="input",
        summary=(
            "The extracted frame count differs from round(duration * fps) "
            "by more than one frame."
        ),
        remediation=(
            "Re-run `crinis ingest` with `--force`; if it persists, "
            "check the source file for variable frame rate."
        ),
    ),
    "CRN-1007": ErrorSpec(
        code="CRN-1007",
        category="input",
        summary="The input video has no audio stream; observation.audio.rms will be set to 0.",
        remediation=(
            "This is a warning only; no action required unless audio data is needed."
        ),
    ),
    "CRN-1008": ErrorSpec(
        code="CRN-1008",
        category="input",
        summary=(
            "The installed ffmpeg build lacks a filter or encoder the configured "
            "pipeline requires."
        ),
        remediation=(
            "Run `crinis doctor --ffmpeg` to list what the build provides, then either "
            "install a full ffmpeg build (libplacebo/zscale/x264) or set "
            "`video.tonemap: none` for SDR sources."
        ),
    ),
    "CRN-1009": ErrorSpec(
        code="CRN-1009",
        category="input",
        summary="ffmpeg or ffprobe exited non-zero.",
        remediation=(
            "The error carries the command and the tail of stderr; the usual causes are "
            "an unsupported pixel format, a bad filter string for this ffmpeg version, "
            "or an unreadable source file."
        ),
    ),
    "CRN-1010": ErrorSpec(
        code="CRN-1010",
        category="input",
        summary=(
            "A corrected-hand label document is unreadable, declares a foreign schema, or contains a "
            "malformed hand record (bad shape, duplicate frame/side, or a shape that contradicts a "
            "declared absence)."
        ),
        remediation=(
            "Re-import the labels with `python scripts/hand_accuracy.py import-labels`; the document "
            "must declare schema 'crinis.user-labels/2' and one hand record per side per frame."
        ),
    ),
    "CRN-1011": ErrorSpec(
        code="CRN-1011",
        category="input",
        summary=(
            "A corrected-hand label document addresses frames outside the run's canonical frame axis, "
            "or its declared frame count disagrees with the run."
        ),
        remediation=(
            "Labels and the run must cover the same canonical frame sequence; re-import the labels "
            "against this run's video, or pass the run whose frame count matches the document."
        ),
    ),
    "CRN-1012": ErrorSpec(
        code="CRN-1012",
        category="input",
        summary="A corrected-hand label document's fps differs from the run's canonical fps.",
        remediation=(
            "Re-import or re-annotate the labels at the run's `video.fps`; frame indices from a "
            "different rate do not describe the same frames."
        ),
    ),
    "CRN-1013": ErrorSpec(
        code="CRN-1013",
        category="input",
        summary=(
            "A corrected-hand label document's footage SHA-256 differs from the run's expected "
            "footage identity, or the run cannot establish that identity."
        ),
        remediation=(
            "Set `hands.corrected.video_sha256` to the digest the labels were made against, or point "
            "`input.video` at the source file; a label set is never applied to different bytes."
        ),
    ),
    "CRN-1014": ErrorSpec(
        code="CRN-1014",
        category="input",
        summary=(
            "A corrected-hand label document's joint count or joint order differs from the canonical "
            "21-joint hand order."
        ),
        remediation=(
            "Re-import the labels with the canonical wrist-first 21-joint order; a reordered joint "
            "array would silently map every fingertip onto another joint."
        ),
    ),
    "CRN-1015": ErrorSpec(
        code="CRN-1015",
        category="input",
        summary=(
            "A user-label document carries both a human attestation and a human-edited hand shape "
            "for the same frame and side."
        ),
        remediation=(
            "A slot is either edited or attested-unchanged, never both. Remove the attestation for "
            "a slot the human moved, or remove the edited shape for a slot the human accepted "
            "unchanged, then re-run `python scripts/hand_accuracy.py import-labels`."
        ),
    ),
    "CRN-1016": ErrorSpec(
        code="CRN-1016",
        category="input",
        summary=(
            "An attestation names a frame/side with no matching pre-label shape, or the document "
            "does not materialize the accepted pre-label as an attested hand record."
        ),
        remediation=(
            "Attest only slots the pre-label actually predicted; check the pre-label manifest's "
            "keyframes_with_shape. Re-run `python scripts/hand_accuracy.py import-labels "
            "--attestations <file> --prelabels <annotations.xml>` so the adopted coordinates are "
            "written into the document."
        ),
    ),
    "CRN-1017": ErrorSpec(
        code="CRN-1017",
        category="input",
        summary=(
            "An attestation is missing its operator, reviewed_at_utc timestamp, or a finite "
            "positive tolerance_px."
        ),
        remediation=(
            "Record who reviewed the pre-label, when (UTC), and the acceptance radius in pixels; "
            "an attestation without an operator, timestamp and tolerance is not a human sign-off."
        ),
    ),
    "CRN-1018": ErrorSpec(
        code="CRN-1018",
        category="input",
        summary=(
            "An attestation's prelabel_sha256 does not match the pre-label bytes it names, or the "
            "attested hand record and the attestation disagree on that digest."
        ),
        remediation=(
            "Re-attest against the current pre-label annotations.xml; the digest binds the "
            "sign-off to the exact bytes the operator reviewed. A changed pre-label invalidates "
            "the old attestation."
        ),
    ),
    "CRN-1019": ErrorSpec(
        code="CRN-1019",
        category="input",
        summary=(
            "A shape marked review_state=edited is identical to its pre-label (coordinates and "
            "per-joint visibility within the rounding tolerance), so it is not a human correction."
        ),
        remediation=(
            "An unchanged shape must be attested, not marked edited: use review_state=unedited_correct "
            "so it is recorded as provenance code 3 (censored), never as a zero-error human correction "
            "that would inflate accuracy. Mark edited only when a human actually changed the shape."
        ),
    ),
    "CRN-1020": ErrorSpec(
        code="CRN-1020",
        category="input",
        summary=(
            "A measured camera calibration is missing required measurement metadata: the schema, "
            "device, lens, solved grid, board inner-corner count or measured square_mm."
        ),
        remediation=(
            "Supply every field explicitly (scripts/calibrate_camera.py --device --lens --square-mm "
            "--cols --rows); a calibration without its physical scale and capture identity is an "
            "assumption, not a measurement, and is never emitted or loaded."
        ),
    ),
    "CRN-1021": ErrorSpec(
        code="CRN-1021",
        category="input",
        summary="A chessboard solve's RMS reprojection error is >= 0.5 px.",
        remediation=(
            "Re-film the board: sharper focus, more even lighting, no motion blur, and a flat "
            "board. A solve this loose is not a measured intrinsic; crinis refuses to emit it "
            "rather than publishing an unvalidated K."
        ),
    ),
    "CRN-1022": ErrorSpec(
        code="CRN-1022",
        category="input",
        summary="Fewer than 15 distinct chessboard views were detected for a measured solve.",
        remediation=(
            "Record a longer board sweep (20-40 s) tilting through both axes and varying distance; "
            "a focal fitted to fewer poses is an assumption wearing a measurement's clothes."
        ),
    ),
    "CRN-1023": ErrorSpec(
        code="CRN-1023",
        category="input",
        summary=(
            "The detected board corners never reach one or more outer-20% bands of the frame, so "
            "the lens distortion is unconstrained where it matters most."
        ),
        remediation=(
            "Sweep the board into every frame edge and corner during the recording; a centre-only "
            "solve cannot measure the distortion that pushes wrists out of frame."
        ),
    ),
    "CRN-1024": ErrorSpec(
        code="CRN-1024",
        category="input",
        summary=(
            "A clip's capture device or lens is NOT_ESTABLISHED (no mdta tag in the container), so "
            "no measured calibration may be bound to it."
        ),
        remediation=(
            "Probe the camera ORIGINAL (scripts/probe_devices.py); re-encodes strip Apple mdta tags. "
            "crinis refuses the clip rather than silently applying another device's K. Record the "
            "operator's device claim beside the tag, never in place of it."
        ),
    ),
    "CRN-1025": ErrorSpec(
        code="CRN-1025",
        category="input",
        summary="No measured crinis.camera-calibration/1 record matches the clip's (device, lens).",
        remediation=(
            "Film a chessboard on THAT device and lens at the production resolution and run "
            "scripts/calibrate_camera.py; a record measured on another device/lens is not a "
            "measurement of this one."
        ),
    ),
    "CRN-1026": ErrorSpec(
        code="CRN-1026",
        category="input",
        summary=(
            "A calibration solved at one pixel grid is being applied to a clip on a different grid "
            "without an explicit rescale (e.g. 960x540 K on 1920x1080 footage is a silent x2 focal "
            "error)."
        ),
        remediation=(
            "Re-solve the board at the clip's decoded grid, or pass the explicit rescale option; "
            "the binding then records both grids and the scale factors. Never rescale silently."
        ),
    ),
    "CRN-1027": ErrorSpec(
        code="CRN-1027",
        category="input",
        summary=(
            "The operator's measured wrist-to-middle_mcp hand-anchor length L_mm is missing or "
            "invalid, so the per-track metric scale cannot be solved."
        ),
        remediation=(
            "Measure the operator's wrist-to-middle_mcp length in millimetres and pass it "
            "(scripts/estimate_hand_scale.py --length-mm). Crinis never assumes a hand size: "
            "Z = fx * L_mm / l_px is linear in L_mm, so a guessed length is a guessed depth."
        ),
    ),
    "CRN-1028": ErrorSpec(
        code="CRN-1028",
        category="input",
        summary=(
            "A track offers only one eligible anchor frame, so its metric scale estimate is "
            "refused rather than emitted."
        ),
        remediation=(
            "A single foreshortened bone gives a wrong Z with no spread to judge it by; the "
            "scale solve needs >=2 eligible wrist/middle_mcp frames and prefers the "
            "largest-extent (near-fronto-parallel) ones. Record more of the track, or accept "
            "the refusal (status=1, z_m=0) in the hand_scale artifact."
        ),
    ),
    "CRN-1029": ErrorSpec(
        code="CRN-1029",
        category="input",
        summary=(
            "A device_spec_derived intrinsics request cannot be derived: the clip's mdta "
            "35mm-equivalent focal tag is absent, non-numeric, or non-positive."
        ),
        remediation=(
            "Probe the camera ORIGINAL's moov mdta (scripts/probe_devices.py); re-encodes strip "
            "the tag. A spec-derived focal is a documented estimate, never a default: crinis "
            "refuses the clip rather than falling back to max(w,h) or borrowing another "
            "device's value. Film a checkerboard for a measured calibration."
        ),
    ),
    # ── Artifact errors (CRN-2xxx) ──────────────────────────────────────────
    "CRN-2001": ErrorSpec(
        code="CRN-2001",
        category="artifact",
        summary="A required upstream artifact is missing.",
        remediation=(
            "Run the stage that produces the artifact first; "
            "check `crinis doctor` for the required stage order."
        ),
    ),
    "CRN-2002": ErrorSpec(
        code="CRN-2002",
        category="artifact",
        summary=(
            "An artifact's content digest does not match its recorded meta; "
            "the cache is invalid."
        ),
        remediation=(
            "Delete the affected artifact directory and re-run the stage to regenerate it."
        ),
    ),
    "CRN-2003": ErrorSpec(
        code="CRN-2003",
        category="artifact",
        summary=(
            "An artifact's schema_version is older than the current code expects."
        ),
        remediation=(
            "Delete the artifact directory and re-run the stage; "
            "schema version mismatches are never auto-migrated."
        ),
    ),
    "CRN-2004": ErrorSpec(
        code="CRN-2004",
        category="artifact",
        summary=(
            "A stage declared an output artifact in `produces` but did not write it."
        ),
        remediation=(
            "Re-run the stage with `--force`; if the error recurs, file a bug "
            "against the stage implementation."
        ),
    ),
    "CRN-2005": ErrorSpec(
        code="CRN-2005",
        category="artifact",
        summary=(
            "A stage raised an unexpected exception that is not a CrinisError."
        ),
        remediation=(
            "Check the stage log and the exception traceback printed above; "
            "re-run with `--debug` for more detail."
        ),
    ),
    "CRN-2006": ErrorSpec(
        code="CRN-2006",
        category="artifact",
        summary=(
            "A metric geometry pass (hand scale, corrected-world backprojection) requires a "
            "MEASURED camera calibration, but the supplied one is assumed, unverified, or absent."
        ),
        remediation=(
            "Run the measured calibration chain and bind its provenance sidecar "
            "(source='measured', calib_sha256 matching the file); set camera.calibration_json "
            "and camera.calibration_provenance. An assumed focal (max-dimension fallback) "
            "silently scales every metric claim and is refused here, never upgraded."
        ),
    ),
    # ── Model errors (CRN-3xxx) ─────────────────────────────────────────────
    "CRN-3001": ErrorSpec(
        code="CRN-3001",
        category="model",
        summary="A required model checkpoint file is absent.",
        remediation=(
            "Run `crinis fetch-weights --only <name>` to download and verify "
            "the missing checkpoint."
        ),
    ),
    "CRN-3002": ErrorSpec(
        code="CRN-3002",
        category="model",
        summary="The MANO .pkl model file is absent.",
        remediation=(
            "Download MANO from https://mano.is.tue.mpg.de and place the .pkl file "
            "at the path shown in `crinis doctor`."
        ),
    ),
    "CRN-3003": ErrorSpec(
        code="CRN-3003",
        category="model",
        summary="CUDA was requested but is not available in this environment.",
        remediation=(
            "Ensure a CUDA-capable GPU is present and the correct PyTorch CUDA build "
            "is installed."
        ),
    ),
    "CRN-3004": ErrorSpec(
        code="CRN-3004",
        category="model",
        summary="A CUDA out-of-memory error occurred during model inference.",
        remediation=(
            "Reduce the window or batch size via `runtime.vram_tier` or "
            "`--set runtime.window=N`; the stage will retry with a halved window "
            "automatically up to `oom_retry.attempts` times."
        ),
    ),
    "CRN-3005": ErrorSpec(
        code="CRN-3005",
        category="model",
        summary="The HAWoR subprocess exited with a non-zero status code.",
        remediation=(
            "Check the subprocess tail shown in this error message; ensure the HAWoR "
            "environment is built correctly with `crinis env selfcheck --env hawor`."
        ),
    ),
    "CRN-3006": ErrorSpec(
        code="CRN-3006",
        category="model",
        summary="A model output does not match the shape, dtype, or channel layout we contract for.",
        remediation=(
            "The upstream pipeline version differs from the one this adapter was written against. "
            "Check the model checkout (see `crinis doctor`) and the array named in this error."
        ),
    ),
    "CRN-3007": ErrorSpec(
        code="CRN-3007",
        category="model",
        summary=(
            "Object pose tracking produced no usable pose for the whole episode: every frame failed "
            "`objects.pose.min_inlier_ratio` or `objects.pose.max_depth_residual_m`."
        ),
        remediation=(
            "Check the instance mesh, the instance mask and the masked depth, and whether "
            "`objects.pose.min_inlier_ratio` is reachable for this capture "
            "(see PLANS/object-centric-upgrade.md §3.5)."
        ),
    ),
    "CRN-3008": ErrorSpec(
        code="CRN-3008",
        category="model",
        summary=(
            "Mesh reconstruction produced a degenerate mesh (fewer than `objects.mesh.min_vertices` "
            "vertices, no faces, or zero volume)."
        ),
        remediation=(
            "Fuse more views, use a longer clip, or lower `objects.mesh.voxel_m` "
            "(see PLANS/object-centric-upgrade.md §3.5)."
        ),
    ),
    "CRN-3009": ErrorSpec(
        code="CRN-3009",
        category="model",
        summary=(
            "The `hands.model_3d: wilor_depth` provider has no focal to interpret WiLoR's "
            "camera-space translation with: `camera.focal_px` is unset and `camera.model` is not a "
            "known preset. WiLoR does not estimate a focal length (its `focal_length` output is the "
            "constant 5000 it was trained with), so it cannot supply one."
        ),
        remediation=(
            "Set `camera.focal_px` (the phone's `AVCameraCalibrationData.intrinsicMatrix` fx is "
            "authoritative) or point `camera.model` at a preset in `config/presets.py`. "
            "Alternatively run `hands.model_3d: hawor`, whose SLAM estimates its own focal."
        ),
    ),
    "CRN-3010": ErrorSpec(
        code="CRN-3010",
        category="model",
        summary=(
            "A hand stage's resident memory exceeded its stated ceiling: the accounted dense frames "
            "(decoded block plus decoder cache) passed the per-batch budget, or the process's "
            "high-water RSS passed the stage ceiling."
        ),
        remediation=(
            "A sequential hand pass must decode one batch, release each frame and hold no "
            "clip-sized cache. Check the reader cache and any buffer that grows with the clip "
            "before raising `MAX_HAND_RESIDENT_BYTES` / `MAX_HAND_PROCESS_RSS_BYTES` in "
            "`crinis.runtime.memory`."
        ),
    ),
    "CRN-3011": ErrorSpec(
        code="CRN-3011",
        category="model",
        summary=(
            "A frame's LiDAR prompt is entirely invalid, so the `depth.metric.provider: promptda` "
            "lane refuses THAT FRAME: its depth stays NaN and is counted, never zero-filled "
            "(zeros would read as a surface at 0 m) and never a whole-episode abort."
        ),
        remediation=(
            "Check the frame in `depth_lidar.npz`: its `valid_mask` is all False because the "
            "confidence gate (>= 2 plus erosion) rejected every pixel, the sensor returned "
            "nothing usable (out of range, reflective or too close), or the frame_index map has "
            "no depth row for it. The refusal is counted in the run's `prompt_empty_frames` and "
            "the promptda receipt; re-shoot the frame's coverage if the fraction is high."
        ),
    ),
    # ── Geometry errors (CRN-4xxx) ──────────────────────────────────────────
    "CRN-4001": ErrorSpec(
        code="CRN-4001",
        category="geometry",
        summary=(
            "A non-finite value (NaN or Inf) was detected in a model output array."
        ),
        remediation=(
            "Check the array name and first bad index in this error; re-run with "
            "`--debug` to dump the offending frame to debug/<stage>/."
        ),
    ),
    "CRN-4002": ErrorSpec(
        code="CRN-4002",
        category="geometry",
        summary=(
            "A quaternion with zero or near-zero norm was passed to a normalisation function."
        ),
        remediation=(
            "Inspect the upstream model output for degenerate rotations; "
            "filter out frames where `pred_valid=0`."
        ),
    ),
    "CRN-4003": ErrorSpec(
        code="CRN-4003",
        category="geometry",
        summary=(
            "The Sim(3) / Umeyama alignment is undefined because the point set "
            "is collinear or too small."
        ),
        remediation=(
            "Increase the number of keyframes or verify that the camera trajectory "
            "spans a non-degenerate motion."
        ),
    ),
    "CRN-4004": ErrorSpec(
        code="CRN-4004",
        category="geometry",
        summary=(
            "The depth fusion scale solve produced too many negative or outlier values "
            "(> 20 % of pixels)."
        ),
        remediation=(
            "Check the Marigold and Metric3D outputs individually with `crinis inspect`; "
            "consider reducing `depth.fusion.ransac_percentile` or increasing keyframe stride."
        ),
    ),
    "CRN-4005": ErrorSpec(
        code="CRN-4005",
        category="geometry",
        summary="No hand detections were found in the entire episode.",
        remediation=(
            "Verify that the input clip contains visible hands; lower "
            "`hands2d.confidence_threshold` or re-run `hands2d` with `--debug` "
            "to see the detector output."
        ),
    ),
    "CRN-4006": ErrorSpec(
        code="CRN-4006",
        category="geometry",
        summary=(
            "The episode contains only one frame, making delta actions and "
            "next.done undefined."
        ),
        remediation=(
            "Ensure the input clip has at least two frames; "
            "check `crinis ingest` output for frame count."
        ),
    ),
    "CRN-4007": ErrorSpec(
        code="CRN-4007",
        category="geometry",
        summary=(
            "More than objects.top_k objects were detected in a frame; "
            "excess detections were dropped."
        ),
        remediation=(
            "This is an informational counter, not a failure; increase `objects.top_k` "
            "if all objects must be retained."
        ),
    ),
    "CRN-4008": ErrorSpec(
        code="CRN-4008",
        category="geometry",
        summary=(
            "state.object_source=track_id was set but the specified track ID "
            "never appears in the episode."
        ),
        remediation=(
            "Run `crinis inspect --stage objects` to list track IDs present, "
            "then update `state.track_id` to a valid ID."
        ),
    ),
    "CRN-4013": ErrorSpec(
        code="CRN-4013",
        category="geometry",
        summary=(
            "The image-anchored translation has no usable evidence: a degenerate pixel extent, "
            "non-finite joints or focal, or a depth outside the band a hand can be at."
        ),
        remediation=(
            "Inspect the failing hand's 2D keypoints and 3D joints for the named frame; a 21-joint "
            "hand under 8 px across or a depth outside [0.02, 15] m means the detection or the model "
            "output is degenerate. Re-run with `hands.translation_source=model`, or fix the 2D lane "
            "so it supplies usable joints."
        ),
    ),
    # ── Writer errors (CRN-5xxx) ─────────────────────────────────────────────
    "CRN-5001": ErrorSpec(
        code="CRN-5001",
        category="writer",
        summary=(
            "The feature dictionary shape or dtype does not match the declared spec "
            "in features.py."
        ),
        remediation=(
            "Check the assembler output against the feature spec with "
            "`crinis validate --table <parquet>`."
        ),
    ),
    "CRN-5002": ErrorSpec(
        code="CRN-5002",
        category="writer",
        summary=(
            "The dataset root already holds a dataset and directories are immutable; an export "
            "will not replace it silently."
        ),
        remediation=(
            "Export to a new path (for example `datasets/<name>/v2`), or pass `--overwrite` when "
            "you are deliberately re-exporting the same development dataset."
        ),
    ),
    "CRN-5003": ErrorSpec(
        code="CRN-5003",
        category="writer",
        summary=(
            "A disk write failed mid-episode; the atomic write guarantees the previous "
            "state is intact."
        ),
        remediation=(
            "Check available disk space with `crinis doctor` and re-run from the last "
            "successful stage."
        ),
    ),
    "CRN-5004": ErrorSpec(
        code="CRN-5004",
        category="writer",
        summary=(
            "The release profile was asked to include a component whose licence forbids it."
        ),
        remediation=(
            "Either drop the component (MANO-derived columns are stripped by "
            "`dataset.profile: release`; `camera.pose_source: hawor_slam` avoids the VGGT "
            "checkpoint) or pass `--allow-nc` after confirming your licence permits the "
            "non-commercial weights."
        ),
    ),
    "CRN-5010": ErrorSpec(
        code="CRN-5010",
        category="writer",
        summary="A NaN was found in a FrameTable column, violating the no-NaN invariant.",
        remediation=(
            "The error message names the column and the contributing stage; re-run that "
            "stage with `--debug` to locate the source frame."
        ),
    ),
    # ── External API errors (CRN-6xxx) ──────────────────────────────────────
    "CRN-6001": ErrorSpec(
        code="CRN-6001",
        category="external_api",
        summary=(
            "The Gemini API returned a 429 or 5xx error after the configured number of "
            "retries."
        ),
        remediation=(
            "Wait and retry, or set `language.description.provider=template` to use the "
            "deterministic fallback without any API calls."
        ),
    ),
    "CRN-6002": ErrorSpec(
        code="CRN-6002",
        category="external_api",
        summary=(
            "The GEMINI_API_KEY environment variable is not set; falling back to the "
            "template composer."
        ),
        remediation=(
            "Set the GEMINI_API_KEY environment variable, or set "
            "`language.description.provider=template` to suppress this warning."
        ),
    ),
    "CRN-6003": ErrorSpec(
        code="CRN-6003",
        category="external_api",
        summary=(
            "The Gemini response is not valid JSON or does not match the expected schema "
            "after one repair attempt."
        ),
        remediation=(
            "Falling back to the deterministic template composer; no action required."
        ),
    ),
}


class CrinisError(Exception):
    """Base error for all crinis pipeline failures."""

    PREFIX: ClassVar[str] = "CRN-"

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str | None = None,
        frame_index: int | None = None,
        remediation: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if code not in ERROR_SPECS:
            raise ValueError(
                f"Unknown error code {code!r}; must be one of {sorted(ERROR_SPECS)}"
            )
        if not code.startswith(self.PREFIX):
            raise ValueError(
                f"Error code {code!r} does not start with prefix {self.PREFIX!r} "
                f"for {type(self).__name__}"
            )
        super().__init__(message)
        self._code = code
        self._message = message
        self._stage = stage
        self._frame_index = frame_index
        self._remediation = (
            remediation if remediation is not None else ERROR_SPECS[code].remediation
        )
        self._context: dict[str, object] = dict(context) if context else {}

    @property
    def code(self) -> str:
        return self._code

    @property
    def message(self) -> str:
        return self._message

    @property
    def stage(self) -> str | None:
        return self._stage

    @property
    def frame_index(self) -> int | None:
        return self._frame_index

    @property
    def remediation(self) -> str:
        return self._remediation

    @property
    def context(self) -> dict[str, object]:
        return self._context

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "code": self._code,
            "message": self._message,
            "category": ERROR_SPECS[self._code].category,
            "remediation": self._remediation,
        }
        if self._stage is not None:
            d["stage"] = self._stage
        if self._frame_index is not None:
            d["frame_index"] = self._frame_index
        if self._context:
            d["context"] = self._context
        return d

    def __str__(self) -> str:
        parts = [self._code]
        if self._stage:
            parts.append(f"[{self._stage}]")
        parts.append(self._message)
        return " ".join(parts) + f" | remediation: {self._remediation}"


class ConfigError(CrinisError):
    PREFIX: ClassVar[str] = "CRN-0"


class InputError(CrinisError):
    PREFIX: ClassVar[str] = "CRN-1"


class ArtifactError(CrinisError):
    PREFIX: ClassVar[str] = "CRN-2"


class ModelError(CrinisError):
    PREFIX: ClassVar[str] = "CRN-3"


class GeometryError(CrinisError):
    PREFIX: ClassVar[str] = "CRN-4"


class WriterError(CrinisError):
    PREFIX: ClassVar[str] = "CRN-5"


class ExternalApiError(CrinisError):
    PREFIX: ClassVar[str] = "CRN-6"
