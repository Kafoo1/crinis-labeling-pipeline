from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

import numpy as np

from crinis.errors import ArtifactError
from crinis.types import ConfigValue

# ── Digest helpers ────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


# ── Atomic write helpers ──────────────────────────────────────────────────────


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically via a .tmp sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, obj: ConfigValue) -> None:
    """Write *obj* as canonical JSON (sorted_keys, indent=2) to *path* atomically."""
    text = json.dumps(obj, sort_keys=True, indent=2)
    atomic_write_text(path, text)


# ── NumPy helpers ─────────────────────────────────────────────────────────────


def save_npz_atomic(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    compress: bool = True,
) -> None:
    """Save *arrays* to an .npz file at *path* atomically.

    Raises ArtifactError("CRN-2002") for object-dtype arrays.
    """
    for name, arr in arrays.items():
        if arr.dtype == object:
            raise ArtifactError(
                "CRN-2002",
                f"Array {name!r} has object dtype; only numeric arrays are allowed.",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy.savez* always appends ".npz" to the path argument.
    # Strategy: build a temp stem that ends with "_crinis_tmp" (no extension),
    # so numpy writes "<stem>_crinis_tmp.npz"; then os.replace that to path.
    tmp_stem = path.parent / (path.name + "_crinis_tmp")
    tmp_npz = tmp_stem.parent / (tmp_stem.name + ".npz")
    # The numpy stubs mistype **kwargs for savez*, so the payload is cast explicitly.
    payload = cast("dict[str, ConfigValue]", dict(arrays))
    try:
        if compress:
            np.savez_compressed(tmp_stem, **payload)
        else:
            np.savez(tmp_stem, **payload)
        os.replace(tmp_npz, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_npz.unlink(missing_ok=True)
        raise


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load an .npz file, preserving dtype and shape exactly.

    Raises ``ArtifactError("CRN-2002")`` for a missing file, a file that is not an npz at all
    (a parquet, say), object-dtype arrays, and anything else numpy refuses to read with
    ``allow_pickle=False``.
    """
    try:
        with np.load(path, allow_pickle=False) as npz:
            result: dict[str, np.ndarray] = {}
            for key in npz.files:
                arr = npz[key]
                if arr.dtype == object:
                    raise ArtifactError(
                        "CRN-2002",
                        f"Array {key!r} has object dtype; only numeric arrays are allowed.",
                    )
                result[key] = arr
    except (OSError, ValueError) as exc:
        raise ArtifactError(
            "CRN-2002",
            f"cannot read {path} as an .npz artifact: {exc}",
            stage="io.storage",
        ) from exc
    return result


def npz_member_header(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype[Any]] | None:
    """The shape and dtype of one ``.npz`` member, without materialising it.

    ``None`` when the file has no such member. A compressed npz is a zip of ``.npy`` streams, so the
    header is readable through the archive without reading - or decompressing - the payload, which is
    what lets a stage size a multi-gigabyte depth volume before deciding how to read it.

    The header is ``_read_npy_header``'s ``(shape, fortran_order, dtype)`` minus the middle field:
    the promise here is ``(shape, dtype)``, and the caller that needs the layout gets it from
    :func:`iter_npz_rows`, which refuses a Fortran-ordered member rather than streaming it wrongly.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            if key + ".npy" not in archive.namelist():
                return None
            with archive.open(key + ".npy", "r") as member:
                shape, _fortran, dtype = _read_npy_header(member, path=path, key=key)
                return shape, dtype
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactError(
            "CRN-2002", f"cannot read {path} as an .npz artifact: {exc}", stage="io.storage"
        ) from exc


def npz_member_names(path: Path) -> tuple[str, ...]:
    """The member names of an ``.npz`` file, in the archive's own order.

    The read side of :func:`save_npz_atomic`'s key set, for a caller that must check the artifact
    declares what a stage requires without loading any member. Raises ``CRN-2002`` for a missing
    file or one that is not a zip.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            return tuple(
                name[: -len(".npy")] for name in archive.namelist() if name.endswith(".npy")
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactError(
            "CRN-2002", f"cannot read {path} as an .npz artifact: {exc}", stage="io.storage"
        ) from exc


def iter_npz_rows(path: Path, key: str, *, rows: int) -> Iterator[np.ndarray]:
    """Yield one ``.npz`` member's leading-axis rows, ``rows`` at a time, in order.

    ``np.load`` materialises a whole member; ``depth_fused.npz`` is 43.6 GiB for a 5 639-frame 1080p
    episode, and the camera stage only needs a per-frame statistic from it. This reads the npy header
    and then hands back contiguous blocks, so the resident part of the artifact is one block - the
    decompressor still streams the bytes it must skip over.
    """
    block_rows = max(1, int(rows))
    try:
        with zipfile.ZipFile(path) as archive:
            member_name = key + ".npy"
            if member_name not in archive.namelist():
                raise ArtifactError(
                    "CRN-2002",
                    f"{path} has no member {key!r}; it holds {sorted(archive.namelist())}",
                    stage="io.storage",
                )
            with archive.open(member_name, "r") as member:
                shape, fortran, dtype = _read_npy_header(member, path=path, key=key)
                if fortran:
                    raise ArtifactError(
                        "CRN-2002",
                        f"{path}:{key} is Fortran-ordered; rows can only be streamed from a C-ordered "
                        f"member",
                        stage="io.storage",
                    )
                if len(shape) == 0:
                    yield np.asarray(np.frombuffer(_read_exact(member, dtype.itemsize), dtype=dtype))
                    return
                row_bytes = dtype.itemsize
                for axis in shape[1:]:
                    row_bytes *= int(axis)
                for start in range(0, int(shape[0]), block_rows):
                    count = min(block_rows, int(shape[0]) - start)
                    payload = _read_exact(member, count * row_bytes)
                    yield np.frombuffer(payload, dtype=dtype).reshape((count, *shape[1:]))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactError(
            "CRN-2002", f"cannot read {path} as an .npz artifact: {exc}", stage="io.storage"
        ) from exc


def _read_npy_header(
    member: IO[bytes], *, path: Path, key: str
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    """``(shape, fortran_order, dtype)`` from one npy stream, refusing the pickle formats."""
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
    else:
        raise ArtifactError(
            "CRN-2002",
            f"{path}:{key} uses the npy format {version[0]}.{version[1]}, which this reader does not "
            f"stream; it is either pickled (never written by this pipeline) or newer than numpy here",
            stage="io.storage",
        )
    if dtype.hasobject:
        raise ArtifactError(
            "CRN-2002", f"{path}:{key} has object dtype; only numeric arrays are allowed.",
            stage="io.storage",
        )
    return tuple(int(axis) for axis in shape), bool(fortran), dtype


def _read_exact(member: IO[bytes], count: int) -> bytes:
    """Read exactly *count* bytes, or refuse a truncated member."""
    chunks: list[bytes] = []
    remaining = int(count)
    while remaining > 0:
        chunk = member.read(remaining)
        if not chunk:
            raise ArtifactError(
                "CRN-2002",
                f"an .npz member ended {remaining} byte(s) before its header said it would; the "
                f"artifact is truncated",
                stage="io.storage",
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


#: Rows of a frame-major member a :class:`NpzRowReader` holds at once. 16 frames of a 1080p
#: float32 volume is ~133 MiB, and the block is the only resident part of the member.
DEFAULT_ROW_BLOCK = 16


class NpzRowReader:
    """Bounded, frame-at-a-time access to one frame-major ``.npz`` member.

    A consumer that needs a per-frame slice of ``depth_fused.npz`` must not call
    :func:`load_artifact`/``np.load``: for a 5 639-frame 1080p episode that member is 43.6 GiB, and
    ``np.asarray(..., dtype=np.float64)`` doubles it. This object reads the npy header once (so
    ``shape``/``dtype`` are known without the payload), then hands out one block of
    :data:`DEFAULT_ROW_BLOCK` frames at a time through :func:`iter_npz_rows`.

    Access may move forward for free and backwards by rewinding: the underlying member is
    compressed, so there is no seeking into the middle of it. ``frame`` therefore documents its cost
    as ``O(index)`` decompression after a rewind, and every caller in this repo walks frames in
    ascending order (one sequential pass) or rewinds once between passes. A returned row is a view
    into the current block - it is valid until the next call that changes the block.

    The shape and dtype are exactly what ``np.load(path)[key]`` would report, so a stage can keep
    asserting the artifact's contract without materialising it.
    """

    def __init__(self, path: Path, key: str, *, block_rows: int = DEFAULT_ROW_BLOCK) -> None:
        self._path = Path(path)
        self._key = key
        self._block_rows = max(1, int(block_rows))
        header = npz_member_header(path, key)
        if header is None:
            raise ArtifactError(
                "CRN-2002",
                f"{path} has no member {key!r}; it holds {npz_member_names(path)}",
                stage="io.storage",
            )
        shape, dtype = header
        if len(shape) == 0:
            raise ArtifactError(
                "CRN-2002",
                f"{path}:{key} is a scalar ({shape}); a frame reader needs a leading frame axis",
                stage="io.storage",
            )
        self._shape = shape
        self._dtype = dtype
        self._rows: Iterator[np.ndarray] | None = None
        self._block: np.ndarray | None = None
        self._block_start = 0
        self._next_row = 0
        self._open()

    @property
    def shape(self) -> tuple[int, ...]:
        """The member's shape, as ``np.load`` would report it."""
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """The member's dtype, as ``np.load`` would report it."""
        return self._dtype

    def __len__(self) -> int:
        return int(self._shape[0])

    def __getitem__(self, index: object) -> np.ndarray:
        """``reader[frame]`` or ``reader[frame, y0:y1, x0:x1]``, like indexing the member itself."""
        if isinstance(index, tuple):
            if not index:
                raise IndexError("an empty index selects the whole member, which is not bounded")
            row = self.frame(int(index[0]))  # type: ignore[arg-type]
            rest = index[1:]
            return row[rest] if rest else row
        return self.frame(int(index))  # type: ignore[arg-type]

    def frame(self, index: int) -> np.ndarray:
        """Row *index* of the leading axis.

        A forward request advances the one open member; a backward request rewinds and re-streams,
        because a compressed member cannot seek. The returned array is a view into the current block
        and stays valid until the next call that advances or rewinds.
        """
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(
                f"frame {index} is outside {self._path}:{self._key}'s {len(self)} frames"
            )
        if self._block is not None and self._block_start <= position < self._next_row:
            return self._block[position - self._block_start]
        if self._rows is None or position < self._block_start:
            self.rewind()
        while self._next_row <= position:
            block = next(self._rows)  # type: ignore[arg-type]
            self._block = block
            self._block_start = self._next_row
            self._next_row += int(block.shape[0])
        assert self._block is not None
        return self._block[position - self._block_start]

    def rewind(self) -> None:
        """Drop the current block and reopen the member from its first frame."""
        self.close()
        self._open()

    def close(self) -> None:
        """Release the member's handle (the archive closes with the generator)."""
        rows = self._rows
        self._rows = None
        self._block = None
        self._block_start = 0
        self._next_row = 0
        if rows is not None:
            rows.close()

    def __enter__(self) -> NpzRowReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        """Every frame, in order, from the beginning (a fresh sequential pass)."""
        self.rewind()
        for position in range(len(self)):
            yield self.frame(position)

    def _open(self) -> None:
        self._rows = iter_npz_rows(self._path, self._key, rows=self._block_rows)
        self._block = None
        self._block_start = 0
        self._next_row = 0


# ── Canonical JSON + digest helpers ──────────────────────────────────────────


def canonical_json(obj: ConfigValue) -> str:
    """Return a canonical JSON string (sorted keys, compact, floats via repr)."""

    def _default(o: ConfigValue) -> ConfigValue:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")

    # We need floats formatted with repr.  The stdlib encoder doesn't support
    # a custom float formatter, so we pre-walk the structure.
    def _normalise(o: ConfigValue) -> ConfigValue:
        if isinstance(o, float):
            return repr(o)
        if isinstance(o, dict):
            return {k: _normalise(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_normalise(v) for v in o]
        return o

    # After normalisation, floats have become strings - we embed them as-is but
    # need a sentinel so the encoder won't add extra quotes.  Instead we use a
    # two-pass approach: encode with a placeholder class.
    class _FloatStr(float):
        """A float subclass whose repr is the original repr string."""

        _s: str

        def __new__(cls, s: str) -> _FloatStr:
            obj2 = super().__new__(cls, float(s))
            obj2._s = s
            return obj2

        def __repr__(self) -> str:
            return self._s

    def _normalise2(o: ConfigValue) -> ConfigValue:
        if isinstance(o, float):
            return _FloatStr(repr(o))
        if isinstance(o, dict):
            return {k: _normalise2(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_normalise2(v) for v in o]
        return o

    return json.dumps(_normalise2(obj), sort_keys=True, separators=(",", ":"), default=_default)


def config_digest(subtree: Mapping[str, object]) -> str:
    """Return the SHA-256 of the canonical JSON of *subtree*."""
    return sha256_bytes(canonical_json(dict(subtree)).encode())


def code_digest(repo_root: Path) -> str:
    """Return a reproducible digest of the current code state.

    Tries ``git rev-parse HEAD``; falls back to hashing src/crinis/**/*.py.
    Never raises.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    # Fallback: hash sorted paths + contents
    try:
        src_root = repo_root / "src" / "crinis"
        paths = sorted(
            str(p.relative_to(repo_root)).replace("\\", "/")
            for p in src_root.rglob("*.py")
        )
        h = hashlib.sha256()
        for rel in paths:
            h.update(rel.encode())
            h.update((repo_root / rel).read_bytes())
        return "nogit-" + h.hexdigest()
    except Exception:
        return "nogit-unknown"


# ── FileRef ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileRef:
    """A reference to a file relative to the run root."""

    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, ConfigValue]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, d: dict[str, ConfigValue]) -> FileRef:
        return cls(path=d["path"], sha256=d["sha256"], bytes=d["bytes"])

    @classmethod
    def from_path(cls, path: Path, run_root: Path) -> FileRef:
        """Build a FileRef by digesting *path* (relative to *run_root*).

        The digest streams in 64 KiB blocks: a produced artifact is not loaded into memory just to
        be described, and ``depth_fused.npz`` is tens of gigabytes.
        """
        return cls(
            path=str(path.relative_to(run_root)).replace("\\", "/"),
            sha256=sha256_file(path),
            bytes=int(path.stat().st_size),
        )


# ── ArraySummary ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArraySummary:
    name: str
    shape: tuple[int, ...]
    dtype: str
    finite_ok: bool
    nonfinite_count: int
    minimum: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, ConfigValue]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "finite_ok": self.finite_ok,
            "nonfinite_count": self.nonfinite_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, d: dict[str, ConfigValue]) -> ArraySummary:
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
            finite_ok=d["finite_ok"],
            nonfinite_count=d["nonfinite_count"],
            minimum=d.get("minimum"),
            maximum=d.get("maximum"),
        )


#: Elements per summary chunk. A memmapped 5 639 x 1080p depth volume must be summarised without
#: becoming resident; this keeps each chunk's temporary bool/float copy a few tens of MiB.
_SUMMARY_CHUNK_ELEMENTS = 1 << 22


def _row_chunks(arr: np.ndarray) -> Iterator[np.ndarray]:
    """Yield *arr* in leading-axis chunks of at most :data:`_SUMMARY_CHUNK_ELEMENTS` elements.

    A 0-d member is one chunk; an empty member yields nothing. Only views are created, so a memmap
    stays paged rather than being read whole.
    """
    if arr.ndim == 0:
        yield arr.reshape(1)
        return
    rows = int(arr.shape[0])
    if rows == 0:
        return
    row_elements = max(1, int(arr.size) // rows)
    step = max(1, _SUMMARY_CHUNK_ELEMENTS // row_elements)
    for start in range(0, rows, step):
        yield arr[start : start + step]


def summarize_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, ArraySummary]:
    """Return an ArraySummary for each array in *arrays*.

    The scan is chunked over the leading axis and its reductions are accumulated, so the numbers are
    the ones the historical whole-array implementation reported (``nonfinite_count`` counts
    ``~isfinite``, ``minimum``/``maximum`` are over the finite values only) while a multi-gigabyte
    memmap is never made resident. Integer and boolean arrays are not reduced, exactly as before.
    """
    return {
        name: _summarize_chunks(
            name, tuple(int(x) for x in arr.shape), arr.dtype, _row_chunks(arr)
        )
        for name, arr in arrays.items()
    }


def summarize_npz(path: Path, *, block_rows: int = DEFAULT_ROW_BLOCK) -> dict[str, ArraySummary]:
    """``summarize_arrays`` for an ``.npz`` on disk, without materialising any member.

    The runner logs a summary of every produced artifact. Doing that with
    ``summarize_arrays(load_npz(path))`` makes a bounded stage's output resident again: for a
    5 639-frame 1080p episode ``depth_fused.npz`` is 43.6 GiB, so the summary alone would undo the
    streaming. Each member is therefore read through :func:`iter_npz_rows` in ``block_rows``-frame
    blocks, and the header supplies the shape/dtype. A scalar (0-d) member is one value, an empty
    member has no values, and the numbers are identical to :func:`summarize_arrays` on
    ``np.load(path)`` for the same bytes.

    Raises ``ArtifactError("CRN-2002")`` for an unreadable, malformed or truncated artifact (the same
    refusal ``load_npz`` gives), so corruption is never summarised as "no arrays".
    """
    result: dict[str, ArraySummary] = {}
    for key in npz_member_names(path):
        header = npz_member_header(path, key)
        if header is None:  # pragma: no cover - the names came from the same archive
            continue
        shape, dtype = header
        result[key] = _summarize_chunks(
            key, shape, dtype, iter_npz_rows(path, key, rows=max(1, int(block_rows)))
        )
    return result


def _summarize_chunks(
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    chunks: Iterator[np.ndarray],
) -> ArraySummary:
    """Accumulate one member's summary over bounded chunks.

    Float members are reduced; every chunk is still pulled, including for integer/bool members, so a
    truncated payload is refused even when its values need no reduction.
    """
    is_float = np.issubdtype(dtype, np.floating)
    nonfinite_count = 0
    minimum: float | None = None
    maximum: float | None = None
    for chunk in chunks:
        if not is_float:
            continue
        values = np.asarray(chunk)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.size == 0:
            continue
        finite = np.isfinite(values)
        nonfinite_count += int(finite.size - np.count_nonzero(finite))
        finite_values = values if bool(finite.all()) else values[finite]
        if finite_values.size == 0:
            continue
        chunk_min = float(np.min(finite_values))
        chunk_max = float(np.max(finite_values))
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
    return ArraySummary(
        name=name,
        shape=shape,
        dtype=str(dtype),
        finite_ok=True if not is_float else nonfinite_count == 0,
        nonfinite_count=nonfinite_count,
        minimum=minimum,
        maximum=maximum,
    )


# ── ArtifactMeta ──────────────────────────────────────────────────────────────

_SCHEMA_VERSION = 1

_TORCH_VERSION_CACHE: str | None = None
_CUDA_VERSION_CACHE: str | None = None


def _get_torch_version() -> str:
    global _TORCH_VERSION_CACHE
    if _TORCH_VERSION_CACHE is None:
        try:
            import importlib
            torch = importlib.import_module("torch")
            _TORCH_VERSION_CACHE = str(torch.__version__)
        except Exception:
            _TORCH_VERSION_CACHE = "unavailable"
    return _TORCH_VERSION_CACHE


def _get_cuda_version() -> str:
    global _CUDA_VERSION_CACHE
    if _CUDA_VERSION_CACHE is None:
        try:
            import importlib
            torch = importlib.import_module("torch")
            cuda = torch.version.cuda
            _CUDA_VERSION_CACHE = str(cuda) if cuda else "unavailable"
        except Exception:
            _CUDA_VERSION_CACHE = "unavailable"
    return _CUDA_VERSION_CACHE


@dataclass(frozen=True)
class ArtifactMeta:
    """Metadata sidecar for a single stage artifact."""

    schema_version: int
    stage: str
    stage_version: int
    config_digest: str
    code_digest: str
    inputs: dict[str, FileRef]
    outputs: dict[str, FileRef]
    host: str
    env: str
    python: str
    torch: str
    cuda: str
    started_utc: str
    ended_utc: str
    wall_s: float
    arrays: dict[str, ArraySummary]
    counters: dict[str, int | float | str]

    def to_json(self) -> str:
        obj: dict[str, ConfigValue] = {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "stage_version": self.stage_version,
            "config_digest": self.config_digest,
            "code_digest": self.code_digest,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "host": self.host,
            "env": self.env,
            "python": self.python,
            "torch": self.torch,
            "cuda": self.cuda,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "wall_s": self.wall_s,
            "arrays": {k: v.to_dict() for k, v in self.arrays.items()},
            "counters": dict(self.counters),
        }
        return json.dumps(obj, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> ArtifactMeta:
        d = json.loads(text)
        return cls(
            schema_version=d["schema_version"],
            stage=d["stage"],
            stage_version=d["stage_version"],
            config_digest=d["config_digest"],
            code_digest=d["code_digest"],
            inputs={k: FileRef.from_dict(v) for k, v in d["inputs"].items()},
            outputs={k: FileRef.from_dict(v) for k, v in d["outputs"].items()},
            host=d["host"],
            env=d["env"],
            python=d["python"],
            torch=d["torch"],
            cuda=d["cuda"],
            started_utc=d["started_utc"],
            ended_utc=d["ended_utc"],
            wall_s=d["wall_s"],
            arrays={k: ArraySummary.from_dict(v) for k, v in d["arrays"].items()},
            counters=d["counters"],
        )

    @classmethod
    def build(
        cls,
        *,
        stage: str,
        stage_version: int,
        cfg_digest: str,
        code_dgst: str,
        env: str,
        inputs: dict[str, FileRef],
        outputs: dict[str, FileRef],
        started_utc: str,
        ended_utc: str,
        wall_s: float,
        arrays: dict[str, ArraySummary],
        counters: dict[str, int | float | str],
    ) -> ArtifactMeta:
        return cls(
            schema_version=_SCHEMA_VERSION,
            stage=stage,
            stage_version=stage_version,
            config_digest=cfg_digest,
            code_digest=code_dgst,
            inputs=inputs,
            outputs=outputs,
            host=socket.gethostname(),
            env=env,
            python=sys.version,
            torch=_get_torch_version(),
            cuda=_get_cuda_version(),
            started_utc=started_utc,
            ended_utc=ended_utc,
            wall_s=wall_s,
            arrays=arrays,
            counters=counters,
        )


# ── Cache validation ──────────────────────────────────────────────────────────


def cache_is_valid(
    meta_path: Path,
    *,
    schema_version: int,
    stage: str,
    stage_version: int,
    config_digest: str,
    inputs: Mapping[str, FileRef],
) -> tuple[bool, str]:
    """Check whether the cached artifact at *meta_path* is still valid.

    Returns ``(True, "")`` when valid, or ``(False, reason)`` otherwise.
    Missing / corrupt meta is never allowed to raise.
    """
    if not meta_path.exists():
        return False, f"meta file missing: {meta_path}"
    try:
        meta = ArtifactMeta.from_json(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"meta unreadable: {exc}"

    if meta.schema_version != schema_version:
        return (
            False,
            f"schema_version mismatch: cached={meta.schema_version} expected={schema_version}",
        )
    if meta.stage != stage:
        return False, f"stage mismatch: cached={meta.stage!r} expected={stage!r}"
    if meta.stage_version != stage_version:
        return (
            False,
            f"stage_version mismatch: cached={meta.stage_version} expected={stage_version}",
        )
    if meta.config_digest != config_digest:
        return (
            False,
            f"config_digest mismatch: cached={meta.config_digest!r} expected={config_digest!r}",
        )
    for name, ref in inputs.items():
        if name not in meta.inputs:
            return False, f"input {name!r} missing from cached meta"
        cached_ref = meta.inputs[name]
        if cached_ref.path != ref.path:
            return (
                False,
                f"input {name!r} path mismatch: cached={cached_ref.path!r} expected={ref.path!r}",
            )
        if cached_ref.sha256 != ref.sha256:
            return (
                False,
                f"input {name!r} digest mismatch: cached={cached_ref.sha256!r} expected={ref.sha256!r}",
            )
    return True, ""
