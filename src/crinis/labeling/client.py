"""The one place CVAT credentials and base URL come from.

Scripts used to carry ``URL = "http://localhost:8080"`` and ``USER, PASSWORD = "admin", "..."`` at module
scope, and three of them imported those globals from each other. That is a credential in the payload, and
it also made every tool fail in the same opaque way when the password was not the one it remembered.

Now:

* the base URL is ``CVAT_URL`` (default ``http://localhost:8080``);
* the token is ``CVAT_TOKEN`` (or the same value from the OS credential store, when a caller supplies it);
* a missing token is a named failure - exit 60 for the CLIs - not a fallback admin password;
* requests are bounded by ``timeout_s``;
* an error body is redacted before it is printed, so a token echoed by a proxy cannot land in a log.

``assert_no_embedded_secrets`` is the scanner the release payload runs over itself: a file that still
assigns a password literal is refused by name, rather than shipped.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://localhost:8080"
DEFAULT_TIMEOUT_S = 30.0

#: Names that make a string assignment a credential. The scan is line-based and deliberately narrow:
#: it looks at the names left of ``=`` and only flags a quoted literal on the right, so
#: ``token = CLIENT.token`` is not an offender while ``PASSWORD = "..."`` is.
_SECRET_NAMES = ("password", "passwd", "secret", "token", "api_key", "apikey")
_ASSIGNMENT_LINE = re.compile(
    r"(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_]*(?:[ \t]*,[ \t]*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"[ \t]*=[ \t]*([\"'])"
)
_BEARER = re.compile(r"(Token|Bearer)\s+[A-Za-z0-9._\-]+")


class CvatAuthError(RuntimeError):
    """No usable credential (exit 60 at the CLI boundary)."""


class CvatRequestError(RuntimeError):
    """A request that did not succeed, with a redacted body."""


def redact(text: str) -> str:
    """Remove anything that looks like a credential from text that is about to be printed."""
    return _BEARER.sub(r"\1 <redacted>", str(text))


@dataclass(frozen=True)
class CvatClient:
    base_url: str
    token: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_environment(cls, *, environ: dict[str, str] | None = None) -> CvatClient:
        """Read ``CVAT_URL``/``CVAT_TOKEN``; raise ``CvatAuthError`` when the token is absent."""
        environ = dict(os.environ if environ is None else environ)
        token = (environ.get("CVAT_TOKEN") or "").strip()
        if not token:
            raise CvatAuthError(
                "CVAT_TOKEN is not set. The annotation tools authenticate with a token from the CVAT "
                "UI (Account settings -> Personal access tokens); export it rather than embedding a "
                "password in a script. No default credential exists"
            )
        base = (environ.get("CVAT_URL") or DEFAULT_URL).rstrip("/")
        try:
            timeout = float(environ.get("CVAT_TIMEOUT") or DEFAULT_TIMEOUT_S)
        except ValueError:
            timeout = DEFAULT_TIMEOUT_S
        return cls(base_url=base, token=token, timeout_s=timeout)

    # ── transport ────────────────────────────────────────────────────────────

    def call(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str = "application/json",
    ) -> Any:  # noqa: ANN401 - a CVAT response is JSON of varying shape
        """One bounded request. Errors carry the status and a redacted body, never the headers."""
        request = urllib.request.Request(
            f"{self.base_url}/api/{path.lstrip('/')}",
            data=data,
            method=method,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Token {self.token}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = redact(exc.read().decode("utf-8", "replace")[:400])
            raise CvatRequestError(f"{method} {path} -> {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise CvatRequestError(f"{method} {path} -> {redact(str(exc.reason))}") from exc
        return json.loads(body) if body else None

    def raw(self, path: str) -> bytes:
        request = urllib.request.Request(
            f"{self.base_url}/api/{path.lstrip('/')}",
            headers={"Authorization": f"Token {self.token}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    # ── tasks ────────────────────────────────────────────────────────────────

    def find_task(self, name: str) -> dict[str, Any] | None:
        """The first task whose name matches exactly, or ``None``.

        Review-task creation is idempotent through this lookup: re-running a router must not create a
        duplicate of a task a human may already be working in.
        """
        listing = self.call("GET", f"tasks?name={urllib.parse.quote(name)}&page_size=50")
        for task in listing.get("results", []) if isinstance(listing, dict) else []:
            if task.get("name") == name:
                return task
        return None

    def create_task(
        self, *, name: str, labels: list[dict[str, Any]], image_quality: int = 100
    ) -> dict[str, Any]:
        return self.call(
            "POST",
            "tasks",
            data=json.dumps({"name": name, "labels": labels, "image_quality": image_quality}).encode(),
        )

    def ensure_task(
        self, *, name: str, labels: list[dict[str, Any]], image_quality: int = 100
    ) -> tuple[dict[str, Any], bool]:
        """``(task, created)`` - an existing task with this exact name is reused, never reset."""
        existing = self.find_task(name)
        if existing is not None:
            return existing, False
        return self.create_task(name=name, labels=labels, image_quality=image_quality), True

    def upload_data(self, task_id: int, archive: bytes, *, content_type: str) -> dict[str, Any]:
        return self.call(
            "POST",
            f"tasks/{task_id}/data",
            data=archive,
            content_type=content_type,
        )

    def wait_request(self, request_id: str, *, attempts: int = 600) -> dict[str, Any]:
        """Poll a CVAT queued request until it finishes, with a bounded number of polls."""
        quoted = urllib.parse.quote(str(request_id), safe="")
        for _ in range(attempts):
            state = self.call("GET", f"requests/{quoted}")
            if isinstance(state, dict) and state.get("status") in ("finished", "failed"):
                return state
        return state if isinstance(state, dict) else {"status": "unknown"}

    def import_annotations(self, task_id: int, xml: bytes, *, fmt: str = "CVAT 1.1") -> dict[str, Any]:
        return self.call(
            "POST",
            f"tasks/{task_id}/annotations?format={urllib.parse.quote(fmt)}",
            data=xml,
            content_type="application/xml",
        )

    def job_url(self, task_id: int, job_id: int | None = None) -> str:
        suffix = f"/jobs/{job_id}" if job_id is not None else ""
        return f"{self.base_url}/tasks/{task_id}{suffix}"


def multipart(fields: dict[str, object], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    """A deterministic multipart body; no credential is placed in it by this helper."""
    boundary = "----crinisreview"
    parts = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    for key, filename, payload, kind in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; '
            f'filename="{filename}"\r\nContent-Type: {kind}\r\n\r\n'.encode()
            + payload
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def assert_no_embedded_secrets(paths: list[Path] | tuple[Path, ...]) -> list[str]:
    """Files that assign a credential literal. The release payload runs this over itself.

    A file is an offender when a line assigns a quoted literal to a name that means a credential
    (``password``, ``token``, ``secret``, ``api_key``, or any name containing ``password``). Reading a
    value from the environment or from another object is not an assignment of a secret.
    """
    offenders: list[str] = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in _ASSIGNMENT_LINE.finditer(text):
            names = [name.strip().lower() for name in match.group(1).split(",")]
            if any(any(marker in name for marker in _SECRET_NAMES) for name in names):
                offenders.append(str(path))
                break
    return offenders
