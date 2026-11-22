"""Upload a `label export` package as a CVAT **video** task and push its annotations.

The package's `annotations.xml` already declares the labels, so the task is created from that
declaration: a hand-written label payload that drifts from the file is the usual reason an import
fails with "label not found".
"""

from __future__ import annotations

import os
import json
import sys
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
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}")
    return json.loads(body) if body else None


def labels_from_xml(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    labels = []
    for label in root.iter("label"):
        attributes = []
        for attribute in label.findall("attribute"):
            kind = attribute.findtext("input_type") or "text"
            values = (attribute.findtext("values") or "").split()
            attributes.append(
                {
                    "name": attribute.findtext("name"),
                    "input_type": kind,
                    "mutable": True,
                    "values": values or [""],
                    "default_value": attribute.findtext("default") or (values[0] if values else ""),
                }
            )
        labels.append({"name": label.findtext("name"), "type": label.findtext("type"),
                       "attributes": attributes})
    return labels


def multipart(fields: dict[str, str], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----crinistaskboundary"
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


def main(task_dir: str, video: str, name: str) -> int:
    directory, clip = Path(task_dir), Path(video)
    token = _client().token
    labels = labels_from_xml(directory / "annotations.xml")
    print("labels:", [label["name"] for label in labels], flush=True)

    task = call("POST", "tasks", token=token,
                data=json.dumps({"name": name, "labels": labels}).encode())
    task_id = task["id"]
    print("task", task_id, flush=True)

    body, kind = multipart({"image_quality": "100"},
                           [("client_files[0]", clip.name, clip.read_bytes(), "video/mp4")])
    queued = call("POST", f"tasks/{task_id}/data", token=token, data=body, content_type=kind)
    if isinstance(queued, dict) and queued.get("rq_id"):
        import time

        for _ in range(100):
            state = call("GET", f"requests/{urllib.parse.quote(str(queued['rq_id']), safe='')}", token=token)
            if isinstance(state, dict) and state.get("status") == "finished":
                print("video upload finished", flush=True)
                break
            if isinstance(state, dict) and state.get("status") == "failed":
                raise SystemExit(f"upload failed: {state}")
            time.sleep(3)

    xml = (directory / "annotations.xml").read_bytes()
    body, kind = multipart({}, [("annotation_file", "annotations.xml", xml, "application/xml")])
    call("POST", f"tasks/{task_id}/annotations?format={urllib.parse.quote('CVAT 1.1')}",
         token=token, data=body, content_type=kind)
    print(f"open {URL}/tasks/{task_id}/jobs", flush=True)
    job = call("GET", f"jobs?task_id={task_id}", token=token)
    if job.get("results"):
        print("job", job["results"][0]["id"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "clip1 hands"))
