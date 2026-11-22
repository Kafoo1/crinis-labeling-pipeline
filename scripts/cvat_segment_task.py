"""Upload a `crinis label export --segments` package as a CVAT **video** task.

The package declares only the two tag labels (`phase`, `contact`), so the task is created from its
`annotations.xml` rather than a hand-written payload that could drift from the file. Tags are imported
last and the import request is awaited, because the API queues it: reading the annotations immediately
shows the empty task, not the labels that were just imported.

    python dist/cvat_segment_task.py <task-dir> <clip.mp4> ["task name"]
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


def call(method: str, path: str, *, token: str | None = None, data: bytes | None = None,
         content_type: str = "application/json"):
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


def labels_from_xml(path: Path) -> list[dict]:
    """The label payload CVAT needs, read from the document the annotations will be imported into."""
    root = ET.parse(path).getroot()
    labels: list[dict] = []
    for label in root.iter("label"):
        attributes = []
        # CVAT's document nests them as <label><attributes><attribute>…: a direct-child search
        # finds none, and a task created without them silently drops every imported tag value.
        for attribute in label.findall("attributes/attribute"):
            kind = attribute.findtext("input_type") or "text"
            values = (attribute.findtext("values") or "").split()
            attributes.append(
                {
                    "name": attribute.findtext("name"),
                    "input_type": kind,
                    "mutable": True,
                    "values": values or [""],
                    "default_value": attribute.findtext("default_value")
                    or (values[0] if values else ""),
                }
            )
        labels.append(
            {
                "name": label.findtext("name"),
                "type": label.findtext("type"),
                "attributes": attributes,
            }
        )
    return labels


def multipart(fields: dict[str, str], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----crinissegmenttask"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for key, filename, payload, kind in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
            f"Content-Type: {kind}\r\n\r\n".encode() + payload + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def wait_for(token: str, queued: object, *, what: str, attempts: int = 200) -> None:
    """Wait for a queued RQ request, so the next read sees its effect."""
    if not (isinstance(queued, dict) and queued.get("rq_id")):
        return
    rq_id = urllib.parse.quote(str(queued["rq_id"]), safe="")
    for _ in range(attempts):
        state = call("GET", f"requests/{rq_id}", token=token)
        status = state.get("status") if isinstance(state, dict) else None
        if status == "finished":
            print(f"{what} finished", flush=True)
            return
        if status == "failed":
            raise SystemExit(f"{what} failed: {state}")
        time.sleep(2)
    raise SystemExit(f"{what} did not finish after {attempts} polls")


def main(task_dir: str, clip: str, name: str = "clip1 phase+contact") -> int:
    directory, clip_path = Path(task_dir), Path(clip)
    xml_path = directory / "annotations.xml"
    if not xml_path.is_file():
        raise SystemExit(f"{xml_path} does not exist; run `crinis label export --segments` first")
    token = _client().token

    labels = labels_from_xml(xml_path)
    print("labels:", [label["name"] for label in labels], flush=True)
    task = call("POST", "tasks", token=token, data=json.dumps({"name": name, "labels": labels}).encode())
    task_id = task["id"]
    print("task", task_id, flush=True)

    body, kind = multipart(
        {"image_quality": "100"},
        [("client_files[0]", clip_path.name, clip_path.read_bytes(), "video/mp4")],
    )
    wait_for(
        token,
        call("POST", f"tasks/{task_id}/data", token=token, data=body, content_type=kind),
        what="video upload",
    )

    listing = call("GET", f"tasks/{task_id}/data/meta", token=token)
    print("meta:", {key: listing.get(key) for key in ("size", "chunk_size", "image_quality")}, flush=True)

    body, kind = multipart({}, [("annotation_file", "annotations.xml", xml_path.read_bytes(), "application/xml")])
    wait_for(
        token,
        call("POST", f"tasks/{task_id}/annotations?format={urllib.parse.quote('CVAT 1.1')}",
             token=token, data=body, content_type=kind),
        what="annotation import",
    )

    stored = call("GET", f"tasks/{task_id}/annotations", token=token)
    job = call("GET", f"jobs?task_id={task_id}", token=token)
    job_id = job["results"][0]["id"]
    print(
        "imported:", len(stored.get("tags") or []), "tag(s),",
        len(stored.get("shapes") or []), "shape(s),",
        len(stored.get("tracks") or []), "track(s)", flush=True,
    )
    print(f"TASK {task_id} JOB {job_id}", flush=True)
    print(f"{URL}/tasks/{task_id}/jobs/{job_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "clip1 phase+contact")
    )
