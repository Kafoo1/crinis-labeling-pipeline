from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Context variable for run_id ─────────────────────────────────────────────

_run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)


def set_run_id(run_id: str) -> None:
    """Set the run_id bound into every subsequent log record."""
    _run_id_var.set(run_id)


def get_run_id() -> str | None:
    return _run_id_var.get()


# ── Redaction ────────────────────────────────────────────────────────────────

_REDACT_PATTERN = re.compile(r"(token|key|secret)", re.IGNORECASE)


def redact(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *mapping* with sensitive keys replaced by '***'."""
    result: dict[str, Any] = {}
    for k, v in mapping.items():
        if _REDACT_PATTERN.search(k):
            result[k] = "***"
        else:
            result[k] = v
    return result


# ── JSONL handler ────────────────────────────────────────────────────────────

class _JsonlHandler(logging.Handler):
    """Writes one JSON object per line to a file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            obj: dict[str, Any] = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "run_id": getattr(record, "run_id", None),
                "stage": getattr(record, "stage", None),
                "event": getattr(record, "event", record.getMessage()),
            }
            if getattr(record, "code", None):
                obj["code"] = record.code  # type: ignore[attr-defined]
            if getattr(record, "frame", None) is not None:
                obj["frame"] = record.frame  # type: ignore[attr-defined]
            extra = getattr(record, "fields", {})
            obj.update(redact(extra))
            self._file.write(json.dumps(obj) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._file.close()
        super().close()


# ── Console formatter ────────────────────────────────────────────────────────

class _ConsoleFormatter(logging.Formatter):
    """ISO8601 LEVEL [stage] event key=value..."""

    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z"
        )
        level = record.levelname
        stage = getattr(record, "stage", None) or "?"
        event = getattr(record, "event", record.getMessage())
        fields = getattr(record, "fields", {})
        parts = [f"{ts} {level:8s} [{stage}] {event}"]
        for k, v in redact(fields).items():
            parts.append(f"{k}={v}")
        return " ".join(parts)


# ── configure() ──────────────────────────────────────────────────────────────

_configured = False


def configure(
    level: str = "INFO",
    *,
    log_dir: Path | None = None,
    run_id: str | None = None,
    jsonl: bool = True,
    console: bool = True,
    debug_arrays: bool = False,
) -> None:
    """Configure the crinis logging stack (idempotent; never duplicates handlers)."""
    global _configured  # standard sentinel pattern

    root = logging.getLogger("crinis")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    if run_id is not None:
        set_run_id(run_id)

    # Guard against duplicate handlers on repeated calls
    existing_types = {type(h) for h in root.handlers}

    if console and logging.StreamHandler not in existing_types:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(_ConsoleFormatter())
        root.addHandler(sh)

    if jsonl and log_dir is not None and _JsonlHandler not in existing_types:
        log_path = log_dir / "crinis.jsonl"
        root.addHandler(_JsonlHandler(log_path))

    # Expose debug_arrays flag on the root logger for stages to read
    root.debug_arrays = debug_arrays  # type: ignore[attr-defined]
    _configured = True


# ── StageLogger ──────────────────────────────────────────────────────────────

class StageLogger:
    """Structured logger bound to a stage and the current run_id contextvar."""

    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._logger = logging.getLogger(f"crinis.{stage}")

    def _emit(self, level: int, event: str, **fields: object) -> None:
        run_id = get_run_id()
        extra = {
            "run_id": run_id,
            "stage": self._stage,
            "event": event,
            "fields": fields,
        }
        self._logger.log(level, event, extra=extra)

    def debug(self, msg: str, **fields: object) -> None:
        self._emit(logging.DEBUG, msg, **fields)

    def info(self, msg: str, **fields: object) -> None:
        self._emit(logging.INFO, msg, **fields)

    def warning(self, msg: str, **fields: object) -> None:
        self._emit(logging.WARNING, msg, **fields)

    def error(self, msg: str, **fields: object) -> None:
        self._emit(logging.ERROR, msg, **fields)

    def event(self, event: str, **fields: object) -> None:
        self._emit(logging.INFO, event, **fields)

    def progress(self, frame: int, total: int, **fields: object) -> None:
        self._emit(logging.INFO, "progress", frame=frame, total=total, **fields)


def get_logger(stage: str) -> StageLogger:
    """Return a StageLogger bound to *stage*."""
    return StageLogger(stage)


# ── AnomalyCollector ─────────────────────────────────────────────────────────

class AnomalyCollector:
    """Collect per-frame anomalies keyed by CRN error code."""

    _MAX_SAMPLES = 5

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def add(self, code: str, frame_index: int | None = None, **fields: object) -> None:
        sample: dict[str, Any] = {"frame_index": frame_index, **fields}
        if code not in self._data:
            self._data[code] = {
                "count": 0,
                "first": sample,
                "last": sample,
                "samples": [],
            }
        entry = self._data[code]
        entry["count"] = entry["count"] + 1
        entry["last"] = sample
        if len(entry["samples"]) < self._MAX_SAMPLES:
            entry["samples"].append(sample)

    def summary(self) -> dict[str, dict[str, Any]]:
        return {code: dict(v) for code, v in self._data.items()}

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for code, entry in self._data.items():
                obj = {"code": code, **entry}
                fh.write(json.dumps(obj) + "\n")

    def is_empty(self) -> bool:
        return len(self._data) == 0
