"""Rebuild the ten double-label frames at native 1920x1080 from the source video.

The 720p clip was cut from ``~/Downloads/ultrawide-video (1).mp4`` (1920x1080, 3339 frames), and
``clip_0001`` is source frames 0..149, so clip frame ``f`` is source frame ``f``. Coordinates scale
by exactly 1.5 (1920/1280 and 1080/720), which is the same factor the clip cutter applied.

Two blind tasks over the same frames, exactly as ``dist/cvat_double_label.py`` built at 720p:

  - PASS A: frames in ascending order, label it first;
  - PASS B: the same frames shuffled, label it second without reopening A.

The frame list, PASS B order and pipeline skeleton XML come from the 720p double-label manifest, so
the native run is the same experiment at native resolution rather than a different sample. Skeleton
labels and the per-joint ``visibility`` attribute spec come from ``crinis.labeling.cvat``, the same
writer the 1080p XML and the CVAT task payload use.

    python dist/cvat_native_res.py <out-dir> [--source <video>] [--manifest <manifest.json>]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import av
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvat_double_label import (  # noqa: E402 - dist is a script directory, not a package
    URL,
    _client,
    call,
    create_task,
    load_hands,
    skeleton_svg,
)
from crinis.hand.keypoint_spec import NUM_JOINTS  # noqa: E402
from crinis.labeling.cvat import (  # noqa: E402
    SkeletonShape,
    skeleton_label_payload,
    write_skeleton_xml,
)

SCALE = 1.5
WIDTH, HEIGHT = 1920, 1080
SHUFFLE_SEED = 20261028
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path.home() / "Downloads" / "ultrawide-video (1).mp4"
DEFAULT_MANIFEST = REPO_ROOT / ".omp/qa/mecka/double_label/manifest.json"


def _manifest_frames(manifest_path: Path) -> tuple[list[int], list[int], Path]:
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} does not exist; the native rebuild needs the 720p manifest for the "
            "frame list, PASS B order and pipeline skeleton XML"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = sorted(int(frame) for frame in manifest.get("frames", []))
    if not frames:
        raise SystemExit(f"{manifest_path} has no 'frames' list")
    raw_order_b = manifest.get("order_b")
    if raw_order_b:
        order_b = [int(frame) for frame in raw_order_b]
    else:
        order_b = list(frames)
        random.Random(SHUFFLE_SEED).shuffle(order_b)
    if sorted(order_b) != sorted(frames):
        raise SystemExit(
            f"{manifest_path} order_b {order_b} is not a permutation of frames {frames}"
        )
    raw_source = manifest.get("source_skeleton_xml")
    if not raw_source:
        raise SystemExit(f"{manifest_path} has no 'source_skeleton_xml'")
    source_xml = Path(str(raw_source))
    if not source_xml.is_absolute():
        source_xml = REPO_ROOT / source_xml
    return frames, order_b, source_xml


def _write_frames(
    source: Path,
    images: Path,
    wanted: set[int],
) -> None:
    if not source.exists():
        raise SystemExit(
            f"source video {source} does not exist; the native 1080p tasks cannot be rebuilt"
        )
    images.mkdir(parents=True, exist_ok=True)
    written: set[int] = set()
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        if (width, height) != (WIDTH, HEIGHT):
            raise SystemExit(
                f"source video is {width}x{height}, expected {WIDTH}x{HEIGHT}; refusing to scale "
                "coordinates by a factor that does not match the pixels"
            )
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                image = frame.to_ndarray(format="bgr24")
                if (image.shape[1], image.shape[0]) != (WIDTH, HEIGHT):
                    raise SystemExit(
                        f"decoded frame {index} is {image.shape[1]}x{image.shape[0]}, expected "
                        f"{WIDTH}x{HEIGHT}"
                    )
                target = images / f"frame_{index:04d}.png"
                if not cv2.imwrite(str(target), image):
                    raise SystemExit(f"could not write {target}")
                written.add(index)
            if index >= max(wanted):
                break
    missing = sorted(wanted - written)
    if missing:
        raise SystemExit(f"source video ended before frames {missing}")


def main(out_dir: str, source: str | None = None, manifest: str | None = None) -> int:
    out = Path(out_dir)
    source_path = Path(source) if source else DEFAULT_SOURCE
    manifest_path = Path(manifest) if manifest else DEFAULT_MANIFEST
    if out.resolve() == manifest_path.parent.resolve():
        raise SystemExit(
            f"{out} is the 720p double-label directory; pick another --out so its manifest and "
            "images are not overwritten"
        )
    frames, order_b, skeleton_xml = _manifest_frames(manifest_path)
    if not skeleton_xml.exists():
        raise SystemExit(f"pipeline skeleton XML {skeleton_xml} does not exist")

    hands = load_hands(skeleton_xml)
    missing_frames = [frame for frame in frames if frame not in hands]
    if missing_frames:
        raise SystemExit(
            f"pipeline skeleton XML {skeleton_xml} has no hands on frames {missing_frames}"
        )
    native_hands: dict[int, list[SkeletonShape]] = {}
    for frame in frames:
        shapes: list[SkeletonShape] = []
        for label, points in hands[frame].items():
            if len(points) != NUM_JOINTS:
                raise SystemExit(
                    f"frame {frame} {label} has {len(points)} joints, expected {NUM_JOINTS}"
                )
            shapes.append(
                SkeletonShape(
                    label=label,
                    joints=tuple(
                        (float(x) * SCALE, float(y) * SCALE) for x, y in points
                    ),
                )
            )
        native_hands[frame] = shapes

    images = out / "images"
    _write_frames(source_path, images, set(frames))
    print(
        f"{len(frames)} frames at {WIDTH}x{HEIGHT} from {source_path} "
        f"(scale {SCALE:g}, source frame == clip frame)",
        flush=True,
    )

    xml_a = out / "annotations_pass_a.xml"
    xml_b = out / "annotations_pass_b.xml"
    write_skeleton_xml(
        xml_a,
        task_name="repeatability PASS A (native 1080p)",
        order=frames,
        hands=native_hands,
        width=WIDTH,
        height=HEIGHT,
    )
    write_skeleton_xml(
        xml_b,
        task_name="repeatability PASS B (native 1080p)",
        order=order_b,
        hands=native_hands,
        width=WIDTH,
        height=HEIGHT,
        prefix="b",
    )

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "double-label-repeatability-native",
                "source_video": str(source_path),
                "source_skeleton_xml": str(skeleton_xml),
                "scale": SCALE,
                "width": WIDTH,
                "height": HEIGHT,
                "frames": frames,
                "order_a": frames,
                "order_b": order_b,
                "note": (
                    "PASS A must be labelled before PASS B is opened; blindness is what makes "
                    "the number a ceiling"
                ),
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    first_shapes = native_hands[frames[0]]
    template = next(
        (shape.joints for shape in first_shapes if shape.label == "hand_left"),
        first_shapes[0].joints,
    )
    token = _client().token
    label_payload = skeleton_label_payload(skeleton_svg(template))
    a_task, a_job = create_task(
        token,
        "repeatability PASS A (native 1080p, label first)",
        label_payload,
        frames,
        native_hands,
        images,
        xml_a,
    )
    b_task, b_job = create_task(
        token,
        "repeatability PASS B (native 1080p, label after A, blind)",
        label_payload,
        order_b,
        native_hands,
        images,
        xml_b,
        prefix="b",
    )
    print(f"PASS A: {URL}/tasks/{a_task}/jobs/{a_job}", flush=True)
    print(f"PASS B: {URL}/tasks/{b_task}/jobs/{b_job}", flush=True)
    print(f"frames: {frames}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild the double-label frames at native 1920x1080."
    )
    parser.add_argument("out_dir", help="directory for images, XML and manifest")
    parser.add_argument("--source", default=None, help="source video (default: the Downloads movie)")
    parser.add_argument("--manifest", default=None, help="720p double-label manifest.json")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, args.source, args.manifest))
