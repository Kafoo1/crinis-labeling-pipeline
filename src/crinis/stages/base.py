from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from crinis.errors import ConfigError
from crinis.io.layout import RunLayout
from crinis.logging_setup import AnomalyCollector, StageLogger

# ── ref parsing ───────────────────────────────────────────────────────────────

_REF_RE = re.compile(r"^([0-9A-Za-z._-]+):([0-9A-Za-z._/-]+)$")


def parse_ref(ref: str) -> tuple[str, str]:
    """Parse ``'<stage>:<artifact-name>'`` into ``(stage, name)``.

    Raises ConfigError("CRN-0003") on a malformed ref.
    """
    m = _REF_RE.match(ref)
    if not m:
        raise ConfigError(
            "CRN-0003",
            f"Malformed artifact reference {ref!r}: expected '<stage>:<artifact-name>'",
        )
    return m.group(1), m.group(2)


# ── StageContext ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageContext:
    """All context a stage receives when ``run()`` is called."""

    layout: RunLayout
    config: Mapping[str, object]
    run_id: str
    force: bool
    limit_frames: int | None
    device: str
    dtype: str
    debug: bool
    env: str
    logger: StageLogger
    anomalies: AnomalyCollector
    #: Cooperative cancellation: a callable that raises :class:`crinis.research.io.ResearchCancelled`
    #: when the job's deadline has passed or an operator asked to stop. ``None`` (the default) leaves
    #: every stage loop unchanged, so a legacy run cannot be affected by this seam.
    cancel_check: Callable[[], None] | None = None

    def check_cancelled(self) -> None:
        """Raise if the job has been cancelled; a no-op when no check is installed."""
        if self.cancel_check is not None:
            self.cancel_check()

    def config_value(self, key: str) -> object:
        """Return the leaf value addressed by a dotted *key*, or ``None``.

        Unlike :meth:`config_subtree` this accepts any leaf type, because a stage may
        gate its cache on a scalar (``project.seed``) or a list (``video.size``) just
        as legitimately as on a block. A missing path returns ``None`` so the digest
        stays stable and explicit rather than silently empty.
        """
        value: object = self.config
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return None
            value = value[part]
        return value

    def config_subtree(self, key: str) -> Mapping[str, object]:
        """Return the nested mapping addressed by a dotted *key*, or ``{}``.

        ``"depth"`` and ``"hands.hawor"`` both work. A missing path, or a leaf that
        is not a mapping (e.g. a scalar or a list), yields ``{}`` -- so a stage that
        expects a block never has to branch on absence.
        """
        value = self.config_value(key)
        if isinstance(value, Mapping):
            return value
        return {}


# ── Stage protocol ────────────────────────────────────────────────────────────


class Stage:
    """Protocol: every concrete stage must satisfy this interface.

    Class-level attributes are the contract; ``run()`` does the work.
    Declared as a regular class (not ``typing.Protocol``) so that
    ``isinstance`` checks and ``ClassVar`` annotations work under
    ``from __future__ import annotations`` without Protocol machinery.
    """

    name: ClassVar[str]
    stage_version: ClassVar[int]
    env: ClassVar[str]  # "modern" | "hawor"
    requires: ClassVar[tuple[str, ...]]   # "<stage>:<artifact-name>"
    produces: ClassVar[tuple[str, ...]]   # "<stage>:<artifact-name>"

    def produced(self, ctx: StageContext) -> tuple[str, ...]:
        """The outputs *this run* will write.

        ``produces`` unless the configuration makes one optional - which is the case for the depth
        stage's visualisation video (``depth.store_video``): declaring it unconditionally would make
        every run that turns it off fail the runner's CRN-2004 output check.
        """
        del ctx
        return self.produces

    def requires_for(self, ctx: StageContext) -> tuple[str, ...]:
        """The inputs *this run* will read, which are not always ``requires``.

        The default keeps every existing stage byte-identical: the class declaration is the answer.
        A config-dependent lane overrides this when one artifact is a substitute for another - the
        EgoHand lane reads the detector manifest, for instance. Whatever this returns is the single
        source of truth for input hashing, cache validation *and* skip propagation, so the three can
        never disagree about what a run consumed.
        """
        del ctx
        return self.requires

    def verify_cached_outputs(
        self, ctx: StageContext, produced: Sequence[str]
    ) -> str | None:
        """Second opinion on a metadata cache hit: ``None`` means "take the cache".

        The metadata sidecar proves the config digest, stage version and input bytes still match. A
        stage whose bundle is made of *many* files (chunks plus a manifest) can be stale in a way the
        sidecar cannot see - a missing final artifact, a corrupt chunk, a topology table that no longer
        matches the model. Returning a *reason* string here makes the runner log
        ``stage.cache_invalid`` and fall through to the normal run path, which then enforces the
        stage's own completed-bundle rules. Returning ``None`` is the unchanged behaviour.
        """
        del ctx, produced
        return None

    config_keys: ClassVar[tuple[str, ...]]

    def run(self, ctx: StageContext) -> Mapping[str, int | float | str]:
        """Execute the stage; return counters."""
        raise NotImplementedError


# ── StageResult ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str        # "ok" | "cached" | "skipped" | "failed"
    code: str | None   # CrinisError.code when status=="failed", else None
    reason: str        # human-readable summary
    wall_s: float
    attempts: int
    counters: dict[str, int | float | str]
    outputs: dict[str, str]  # artifact name → absolute path string

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "status": self.status,
            "code": self.code,
            "reason": self.reason,
            "wall_s": self.wall_s,
            "attempts": self.attempts,
            "counters": dict(self.counters),
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> StageResult:
        return cls(
            stage=str(d["stage"]),
            status=str(d["status"]),
            code=d.get("code") if d.get("code") is not None else None,  # type: ignore[arg-type]
            reason=str(d.get("reason", "")),
            wall_s=float(d.get("wall_s", 0.0)),  # type: ignore[arg-type]
            attempts=int(d.get("attempts", 1)),  # type: ignore[arg-type]
            counters=dict(d.get("counters", {})),  # type: ignore[arg-type]
            outputs=dict(d.get("outputs", {})),  # type: ignore[arg-type]
        )
