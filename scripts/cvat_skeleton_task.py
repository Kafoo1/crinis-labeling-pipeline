"""Build the CVAT task as 21-node skeletons instead of 21 anonymous dots.

A CVAT `points` shape is one undifferentiated point set: the labeler sees 21 dots and has to know the
index ordering by heart, which is exactly where a hand correction goes wrong. A CVAT `skeleton` label
is one object per hand whose child elements are named (``wrist``, ``index_pip``, ...) and whose nodes
are joined by bones read from the label's SVG template, so a joint moved to the wrong finger shows up
as a bent bone rather than as a dot that looks fine.

The node names are ``crinis.hand.keypoint_spec.JOINT_NAMES`` - the same 21 names the pipeline writes,
so a corrected skeleton maps back index-for-index with no lookup table.

    python dist/cvat_skeleton_task.py <video-xml-dir> <images-dir> <out-dir> [frames]
"""

from __future__ import annotations

import os
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

from crinis.labeling.client import CvatClient

URL = os.environ.get("CVAT_URL", "http://localhost:8080")


def _client() -> CvatClient:
    """The authenticated client. Built on demand so ``--help`` works without a token."""
    return CvatClient.from_environment()
MAX_FRAMES = 40
IMAGE_W, IMAGE_H = 1280, 720

from crinis.hand.keypoint_spec import FINGER_RANGES, JOINT_NAMES, NUM_JOINTS, WRIST_JOINT_INDEX


def call(method: str, path: str, *, token=None, data=None, content_type="application/json"):
    request = urllib.request.Request(
        f"{URL}/api/{path.lstrip('/')}", data=data, method=method,
        headers={"Content-Type": content_type, **({"Authorization": f"Token {token}"} if token else {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}")
    return json.loads(body) if body else None


def multipart(fields, files):
    boundary = "----crinisskeletonboundary"
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for key, filename, payload, kind in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
            f"Content-Type: {kind}\r\n\r\n".encode() + payload + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def skeleton_svg(template: list[list[float]]) -> str:
    """Render the 21-node bone template the label editor would have produced by hand.

    Node coordinates are a real projected hand squeezed into the editor's ``0 0 100 100`` box, so the
    template already looks like the hand being corrected. Node ids are the 1-based ordinal of the
    sublabel, which is what the canvas resolves ``data-node-id`` against.
    """
    xs = [p[0] for p in template]
    ys = [p[1] for p in template]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    scale = 80.0 / span
    ox = 50.0 - (max(xs) - min(xs)) * scale / 2 - min(xs) * scale
    oy = 50.0 - (max(ys) - min(ys)) * scale / 2 - min(ys) * scale
    coords = [(x * scale + ox, y * scale + oy) for x, y in template]

    bones: list[tuple[int, int]] = []
    for _, (start, end) in sorted(FINGER_RANGES.items()):
        chain = list(range(start, end + 1))
        bones.append((WRIST_JOINT_INDEX, start))
        bones.extend(zip(chain, chain[1:]))

    parts = []
    for a, b in bones:
        (x1, y1), (x2, y2) = coords[a], coords[b]
        parts.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="black" '
            f'data-type="edge" data-node-from="{a + 1}" stroke-width="0.5" data-node-to="{b + 1}"></line>'
        )
    for index, (cx, cy) in enumerate(coords):
        parts.append(
            f'<circle r="1.5" stroke="black" fill="#b3b3b3" cx="{cx:.3f}" cy="{cy:.3f}" '
            'stroke-width="0.1" data-type="element node" '
            f'data-element-id="{index + 1}" data-node-id="{index + 1}" '
            f'data-label-name="{JOINT_NAMES[index]}"></circle>'
        )
    return "".join(parts)


def main(source_dir: str, images_dir: str, out_dir: str, limit: int = MAX_FRAMES) -> int:
    source, images, out = Path(source_dir), Path(images_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    video_root = ET.parse(source / "annotations.xml").getroot()
    per_frame: dict[int, dict[str, list[list[float]]]] = {}
    for track in video_root.iter("track"):
        label = track.get("label")
        if label not in ("hand_left", "hand_right"):
            continue
        for element in track.findall("points"):
            points = [[float(v) for v in pair.split(",")] for pair in (element.get("points") or "").split(";")]
            per_frame.setdefault(int(element.get("frame")), {})[label] = points

    chosen = sorted(per_frame)
    step = max(1, len(chosen) // limit)
    chosen = chosen[::step][:limit]
    print(f"{len(per_frame)} frames carry hands; building {len(chosen)} skeleton frames", flush=True)

    present = sorted(images.glob("*.png"))
    wanted = {f"frame_{frame:04d}.png" for frame in chosen}
    present = [path for path in present if path.name in wanted]
    if not present:
        raise SystemExit(f"no frames for {sorted(wanted)[:3]} in {images}")
    print(f"{len(present)} images available", flush=True)

    # Template from the first frame's first hand - a real hand shape, not an invented one.
    template = next(iter(per_frame[int(present[0].stem.split('_')[1])].values()))
    if len(template) != NUM_JOINTS:
        raise SystemExit(f"template has {len(template)} joints, expected {NUM_JOINTS}")

    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "clip1 hands (skeleton)"
    ET.SubElement(task, "size").text = str(len(present))
    labels = ET.SubElement(task, "labels")
    for label_name in ("hand_left", "hand_right"):
        node = ET.SubElement(labels, "label")
        ET.SubElement(node, "name").text = label_name
        ET.SubElement(node, "type").text = "skeleton"
        attribute = ET.SubElement(node, "attribute")
        ET.SubElement(attribute, "name").text = "provenance"
        ET.SubElement(attribute, "input_type").text = "text"
        ET.SubElement(attribute, "mutable").text = "True"
        for joint in JOINT_NAMES:
            child = ET.SubElement(node, "sublabel")
            ET.SubElement(child, "name").text = joint
            ET.SubElement(child, "type").text = "points"

    drawn = 0
    for position, path in enumerate(present):
        frame = int(path.stem.split("_")[1])
        image = ET.SubElement(annotations, "image", id=str(position), name=path.name,
                              width=str(IMAGE_W), height=str(IMAGE_H))
        for label_name, points in sorted(per_frame[frame].items()):
            skeleton = ET.SubElement(image, "skeleton", label=label_name, keyframe="1", z_order="0")
            provenance = ET.SubElement(skeleton, "attribute", name="provenance")
            provenance.text = "pipeline"
            for joint, (x, y) in zip(JOINT_NAMES, points):
                ET.SubElement(skeleton, "points", label=joint, keyframe="1", occluded="0",
                              z_order="0", outside="0", points=f"{x:.3f},{y:.3f}")
            drawn += 1

    ET.indent(ET.ElementTree(annotations), space="  ")
    xml_path = out / "annotations_skeletons.xml"
    ET.ElementTree(annotations).write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"wrote {xml_path} ({drawn} skeletons)", flush=True)

    token = _client().token
    svg = skeleton_svg(template)
    label_payload = [
        {
            "name": label_name, "type": "skeleton", "svg": svg,
            "attributes": [{"name": "provenance", "input_type": "text", "mutable": True,
                            "values": [""], "default_value": ""}],
            "sublabels": [{"name": joint, "type": "points", "attributes": []} for joint in JOINT_NAMES],
        }
        for label_name in ("hand_left", "hand_right")
    ]
    created = call("POST", "tasks", token=token, data=json.dumps(
        {"name": "clip1 hands - label me (skeleton)", "labels": label_payload}).encode())
    task_id = created["id"]
    print(f"task {task_id} created with 2 skeleton labels", flush=True)

    files = [(f"client_files[{index}]", path.name, path.read_bytes(), "image/png")
             for index, path in enumerate(present)]
    body, kind = multipart({"image_quality": "100"}, files)
    queued = call("POST", f"tasks/{task_id}/data", token=token, data=body, content_type=kind)
    if isinstance(queued, dict) and queued.get("rq_id"):
        for _ in range(120):
            state = call("GET", f"requests/{urllib.parse.quote(str(queued['rq_id']), safe='')}", token=token)
            if isinstance(state, dict) and state.get("status") in ("finished", "failed"):
                print("upload", state.get("status"), flush=True)
                break
            time.sleep(3)

    body, kind = multipart({}, [("annotation_file", "annotations_skeletons.xml",
                                 xml_path.read_bytes(), "application/xml")])
    call("POST", f"tasks/{task_id}/annotations?format={urllib.parse.quote('CVAT 1.1')}",
         token=token, data=body, content_type=kind)
    job = call("GET", f"jobs?task_id={task_id}", token=token)
    job_id = job["results"][0]["id"]
    print(f"TASK {task_id} JOB {job_id}", flush=True)
    print(f"{URL}/tasks/{task_id}/jobs/{job_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3],
                          int(sys.argv[4]) if len(sys.argv) > 4 else MAX_FRAMES))
