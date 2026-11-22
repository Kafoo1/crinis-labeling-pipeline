"""Named aliases for the dynamic values that configuration and artifact metadata carry.

``ConfigValue`` is ``Any`` on purpose. A YAML file, a ``--set`` override and a JSON
sidecar can each hold arbitrary JSON (``dict``, ``list``, ``str``, ``int``, ``float``,
``bool``, ``None``), and it is the schema layer -- not the type checker -- that
validates those values structurally. Naming the alias means every dynamic position
states *why* it is dynamic, instead of a bare ``typing.Any`` that tells a reader
nothing and trips ``ruff``'s ANN401 at the same rate.
"""

from __future__ import annotations

from typing import Any, TypeAlias

#: Any JSON-representable value: a scalar, list, mapping or ``None``.
ConfigValue: TypeAlias = Any

#: A class object used as a constructor/coercion target (e.g. a config dataclass).
ConfigType: TypeAlias = type[Any]
