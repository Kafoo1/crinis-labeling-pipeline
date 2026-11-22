from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from crinis.errors import ConfigError

_RUN_ID_RE = re.compile(r"^[0-9A-Za-z._-]{1,64}$")


@dataclass(frozen=True)
class RunLayout:
    """Directory layout for a single pipeline run."""

    root: Path  # runs/<run_id>/

    @classmethod
    def create(cls, runs_root: Path, run_id: str) -> RunLayout:
        """Create the run directory tree and return a RunLayout.

        Raises ConfigError("CRN-0003") when run_id fails the pattern.
        """
        if not _RUN_ID_RE.match(run_id):
            raise ConfigError(
                "CRN-0003",
                f"Invalid run_id {run_id!r}: must match ^[0-9A-Za-z._-]{{1,64}}$",
            )
        root = runs_root / run_id
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        (root / "debug").mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    # ── Directory helpers ─────────────────────────────────────────────────────

    def logs_dir(self) -> Path:
        return self.root / "logs"

    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def debug_dir(self) -> Path:
        return self.root / "debug"

    def artifact_dir(self, stage: str) -> Path:
        return self.root / "artifacts" / stage

    # ── File helpers ──────────────────────────────────────────────────────────

    def artifact(self, stage: str, name: str) -> Path:
        """Return the path for a named artifact produced by *stage*, creating its directory.

        *name* may contain a directory component, e.g. ``frames/frames_index.parquet``.
        The path is always ``artifacts/<stage>/<name>``.

        The parent directory is created here, and that is deliberate: a producer that receives a path
        is entitled to write to it. Measured on Colab, before this line existed, ``ingest`` handed
        ``artifacts/ingest/frames/frames.mp4`` straight to ffmpeg, which cannot create directories -
        so the stage died with ``Error opening output ... No such file or directory`` inside a
        subprocess, and ``hands2d`` and ``depth`` were skipped behind it. One missing ``mkdir``
        presented as a cryptic ffmpeg failure three stages from the cause.
        """
        path = self.root / "artifacts" / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def metadata(self, stage: str, name: str) -> Path:
        """Return the ``.meta.json`` sidecar for *artifact(stage, name)*, with its directory created."""
        path = self.root / "artifacts" / stage / (name + ".meta.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, stage: str) -> Path:
        return self.root / "logs" / f"{stage}.jsonl"

    def anomalies(self) -> Path:
        return self.root / "logs" / "anomalies.jsonl"

    def run_json(self) -> Path:
        return self.root / "run.json"
