"""Build two blind CVAT tasks over the same ten frames, to measure the labeler's own repeatability.

The question "how much more accurate does hand labelling get us" has no answer until we know the
*ceiling*: how close the same person lands to the same joint twice, without seeing the first attempt.
That ceiling is what a label set can never beat, so it bounds every accuracy claim about the dataset.

Two tasks, identical frames, identical pre-annotations:
  - task PASS A: frames in ascending order, label it first;
  - task PASS B: the same frames shuffled, label it second *without reopening A*.

The frame set is not cherry-picked for ease: five are "hard" (a joint leaves the frame, or the
skeleton jumps between adjacent frames) and five are representative (both hands present, no jump, no
clip-to-border joint), because repeatability on the easy frames alone would flatter the result.

    python dist/cvat_double_label.py <skeleton-xml> <clip.mp4> <out-dir>
"""

from __future__ import annotations

import os
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

import av
import cv2

from crinis.labeling.client import CvatClient, multipart as _client_multipart

URL = os.environ.get("CVAT_URL", "http://localhost:8080")


def _client() -> CvatClient:
    """The authenticated client. Built on demand so ``--help`` works without a token."""
    return CvatClient.from_environment()
IMAGE_W, IMAGE_H = 1280, 720
PER_GROUP = 5
JUMP_FACTOR = 5.0

from crinis.hand.keypoint_spec import FINGER_RANGES, JOINT_NAMES, NUM_JOINTS, WRIST_JOINT_INDEX


def call(method: str, path: str, *, token=None, data=None, content_type="application/json"):
    """A thin wrapper over the shared client; ``token`` defaults to the environment's token."""
    if token is None:
        token = _client().token
    request = urllib.request.Request(
        f"{URL}/api/{path.lstrip('/')}", data=data, method=method,
        headers={"Content-Type": content_type, **({"Authorization": f"Token {token}"} if token else {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}")
    return json.loads(body) if body else None


def multipart(fields, files):
    """Backwards-compatible name for the shared multipart builder (no credential in the body)."""
    return _client_multipart(fields, files)


def skeleton_svg(template):
    xs = [p[0] for p in template]
    ys = [p[1] for p in template]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    scale = 80.0 / span
    ox = 50.0 - (max(xs) - min(xs)) * scale / 2 - min(xs) * scale
    oy = 50.0 - (max(ys) - min(ys)) * scale / 2 - min(ys) * scale
    coords = [(x * scale + ox, y * scale + oy) for x, y in template]
    bones = []
    for _, (start, end) in sorted(FINGER_RANGES.items()):
        chain = list(range(start, end + 1))
        bones.append((WRIST_JOINT_INDEX, start))
        bones.extend(zip(chain, chain[1:]))
    parts = []
    for a, b in bones:
        (x1, y1), (x2, y2) = coords[a], coords[b]
        parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="black" '
                     f'data-type="edge" data-node-from="{a + 1}" stroke-width="0.5" data-node-to="{b + 1}"></line>')
    for index, (cx, cy) in enumerate(coords):
        parts.append(f'<circle r="1.5" stroke="black" fill="#b3b3b3" cx="{cx:.3f}" cy="{cy:.3f}" '
                     'stroke-width="0.1" data-type="element node" '
                     f'data-element-id="{index + 1}" data-node-id="{index + 1}" '
                     f'data-label-name="{JOINT_NAMES[index]}"></circle>')
    return "".join(parts)


def load_hands(path: Path):
    """frame -> {label: [[x, y] * 21]} from the exported skeleton XML."""
    root = ET.parse(path).getroot()
    frames: dict[int, dict[str, list[list[float]]]] = {}
    for track in root.iter("track"):
        label = track.get("label")
        if label not in ("hand_left", "hand_right"):
            continue
        for sk in track.findall("skeleton"):
            points = [[float(v) for v in el.get("points").split(",")] for el in sk.findall("points")]
            if len(points) == NUM_JOINTS:
                frames.setdefault(int(sk.get("frame")), {})[label] = points
    return frames


def classify(frames):
    """Split frames into hard (a joint leaves the frame, or the skeleton jumps) and representative."""
    ordered = sorted(frames)
    steps = {}
    for a, b in zip(ordered, ordered[1:]):
        if b - a > 3:
            continue
        for label in set(frames[a]) & set(frames[b]):
            pa = frames[a][label]
            pb = frames[b][label]
            mean = sum(((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                       for (xa, ya), (xb, yb) in zip(pa, pb)) / len(pa)
            steps.setdefault(label, []).append((a, b, mean))
    medians = {label: sorted(v[2] for v in values)[len(values) // 2] for label, values in steps.items() if values}
    jumped = set()
    for label, values in steps.items():
        for a, b, mean in values:
            if medians.get(label) and mean > JUMP_FACTOR * medians[label]:
                jumped |= {a, b}

    hard, easy = [], []
    for frame in ordered:
        off = any(x < 0 or x > IMAGE_W - 1 or y < 0 or y > IMAGE_H - 1
                  for pts in frames[frame].values() for x, y in pts)
        both = len(frames[frame]) == 2
        if off or frame in jumped:
            hard.append(frame)
        elif both:
            easy.append(frame)
    return hard, easy


def write_xml(path: Path, order: list[int], frames, task_name: str, prefix: str = "") -> None:
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(len(order))
    labels = ET.SubElement(task, "labels")
    for label_name in ("hand_left", "hand_right"):
        node = ET.SubElement(labels, "label")
        ET.SubElement(node, "name").text = label_name
        ET.SubElement(node, "type").text = "skeleton"
        for joint in JOINT_NAMES:
            child = ET.SubElement(node, "sublabel")
            ET.SubElement(child, "name").text = joint
            ET.SubElement(child, "type").text = "points"
    for position, frame in enumerate(order):
        image = ET.SubElement(annotations, "image", id=str(position),
                              name=f"{prefix}frame_{frame:04d}.png", width=str(IMAGE_W), height=str(IMAGE_H))
        for label_name, points in sorted(frames[frame].items()):
            skeleton = ET.SubElement(image, "skeleton", label=label_name, keyframe="1", z_order="0")
            for joint, (x, y) in zip(JOINT_NAMES, points):
                ET.SubElement(skeleton, "points", label=joint, keyframe="1", occluded="0", z_order="0",
                              outside="0", points=f"{x:.3f},{y:.3f}")
    ET.indent(ET.ElementTree(annotations), space="  ")
    ET.ElementTree(annotations).write(path, encoding="utf-8", xml_declaration=True)


def create_task(token, name: str, label_payload, order, frames, images_dir, xml_path, prefix=""):
    created = call("POST", "tasks", token=token, data=json.dumps({"name": name, "labels": label_payload}).encode())
    task_id = created["id"]
    # CVAT orders an images task lexicographically by filename, not by upload order, so a shuffled
    # PASS B has to carry its position in the name or the shuffle is silently discarded.
    files = [(f"client_files[{index}]", f"{prefix}{index:02d}_frame_{frame:04d}.png",
              (images_dir / f"frame_{frame:04d}.png").read_bytes(), "image/png")
             for index, frame in enumerate(order)]
    body, kind = multipart({"image_quality": "100"}, files)
    queued = call("POST", f"tasks/{task_id}/data", token=token, data=body, content_type=kind)
    if isinstance(queued, dict) and queued.get("rq_id"):
        for _ in range(200):
            state = call("GET", f"requests/{urllib.parse.quote(str(queued['rq_id']), safe='')}", token=token)
            status = state.get("status") if isinstance(state, dict) else None
            if status in ("finished", "failed"):
                print(f"  {name}: upload {status}", flush=True)
                break
            time.sleep(3)
    body, kind = multipart({}, [("annotation_file", "annotations.xml", xml_path.read_bytes(), "application/xml")])
    queued = call("POST", f"tasks/{task_id}/annotations?format={urllib.parse.quote('CVAT 1.1')}",
                  token=token, data=body, content_type=kind)
    # The annotation import is asynchronous too, and returning before it finishes makes a freshly built
    # task read as empty - measured: task 32 reported 0 skeletons for ~12 s after the shapes were sent.
    if isinstance(queued, dict) and queued.get("rq_id"):
        for _ in range(200):
            state = call("GET", f"requests/{urllib.parse.quote(str(queued['rq_id']), safe='')}", token=token)
            status = state.get("status") if isinstance(state, dict) else None
            if status in ("finished", "failed"):
                print(f"  {name}: annotation import {status}", flush=True)
                break
            time.sleep(3)
    job = call("GET", f"jobs?task_id={task_id}", token=token)
    job_id = job["results"][0]["id"]
    return task_id, job_id


def main(skeleton_xml: str, clip: str, out_dir: str) -> int:
    frames = load_hands(Path(skeleton_xml))
    hard, easy = classify(frames)
    print(f"{len(frames)} frames with hands | hard {len(hard)} | representative {len(easy)}", flush=True)

    chosen_hard = hard[:: max(1, len(hard) // PER_GROUP)][:PER_GROUP]
    chosen_easy = easy[:: max(1, len(easy) // PER_GROUP)][:PER_GROUP]
    chosen = sorted(chosen_hard + chosen_easy)
    if len(chosen) < PER_GROUP * 2:
        raise SystemExit(f"only {len(chosen)} frames available, need {PER_GROUP * 2}")

    out = Path(out_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    wanted = set(chosen)
    with av.open(str(clip)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                cv2.imwrite(str(images / f"frame_{index:04d}.png"), frame.to_ndarray(format="bgr24"))
            if index >= max(wanted):
                break

    order_b = list(chosen)
    random.Random(20261028).shuffle(order_b)
    xml_a, xml_b = out / "annotations_pass_a.xml", out / "annotations_pass_b.xml"
    write_xml(xml_a, chosen, frames, "repeatability PASS A")
    write_xml(xml_b, order_b, frames, "repeatability PASS B", prefix="b")

    (out / "manifest.json").write_text(json.dumps({
        "kind": "double-label-repeatability",
        "frames": chosen,
        "hard_frames": chosen_hard,
        "representative_frames": chosen_easy,
        "order_a": chosen,
        "order_b": order_b,
        "source_skeleton_xml": str(skeleton_xml),
        "note": "PASS A must be labelled before PASS B is opened; blindness is what makes the number a ceiling",
    }, indent=1), encoding="utf-8")

    template = frames[chosen[0]]["hand_left"] if "hand_left" in frames[chosen[0]] else frames[chosen[0]]["hand_right"]
    token = _client().token
    svg = skeleton_svg(template)
    label_payload = [
        {"name": label_name, "type": "skeleton", "svg": svg, "attributes": [],
         "sublabels": [{"name": joint, "type": "points", "attributes": []} for joint in JOINT_NAMES]}
        for label_name in ("hand_left", "hand_right")
    ]
    a_task, a_job = create_task(token, "repeatability PASS A (label first)", label_payload, chosen, frames, images, xml_a)
    b_task, b_job = create_task(token, "repeatability PASS B (label after A, blind)", label_payload,
                                order_b, frames, images, xml_b, prefix="b")
    print(f"PASS A: {URL}/tasks/{a_task}/jobs/{a_job}", flush=True)
    print(f"PASS B: {URL}/tasks/{b_task}/jobs/{b_job}", flush=True)
    print(f"frames: {chosen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
