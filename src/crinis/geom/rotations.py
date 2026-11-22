"""Rotation representations: quaternions, rotation matrices, rot6d, axis-angle.

Conventions (fixed; do not drift):
- Quaternion: ``(w, x, y, z)`` - Hamilton, right-handed, unit norm.
- Rot6d: ``concat(R[0, :], R[1, :])`` - first two rows of R, row-major; the third row
  is recovered by Gram-Schmidt (same convention as pytorch3d
  ``matrix_to_rotation_6d`` / ``rotation_6d_to_matrix``).
- Axis-angle: ``θ * axis`` in radians; the zero vector encodes the identity.

All functions accept batched leading dimensions - e.g. ``(..., 4)`` for quaternions,
``(..., 3, 3)`` for matrices - and preserve them in the output.

Dtype policy: float32 input → float32 output; float64 input → float64 output.

Degenerate / non-finite input raises
``GeometryError("CRN-4002", ...)`` naming the offending array and, where applicable,
the first bad index.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from crinis.errors import GeometryError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NORM_EPS = 1e-8  # below this norm a quaternion is "degenerate"


def _promote(arr: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Ensure arr is at least float32; preserve float64."""
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    return arr.astype(np.float64)


def _check_finite(
    arr: np.ndarray[Any, np.dtype[Any]],
    name: str,
) -> None:
    """Raise CRN-4002 if *arr* contains non-finite values."""
    if not np.all(np.isfinite(arr)):
        bad = np.argmax(~np.isfinite(arr).ravel())
        idx = np.unravel_index(int(bad), arr.shape)
        raise GeometryError(
            "CRN-4002",
            f"Array {name!r} contains non-finite values; first bad index={idx}",
            stage="rotations",
        )


def _check_quat_norm(
    q: np.ndarray[Any, np.dtype[Any]],
    name: str,
) -> None:
    """Raise CRN-4002 if any quaternion in *q* (..., 4) has near-zero norm."""
    _check_finite(q, name)
    norms = np.linalg.norm(q, axis=-1)
    bad = norms < _NORM_EPS
    if np.any(bad):
        flat = int(np.argmax(bad.ravel()))
        idx = np.unravel_index(flat, bad.shape)
        raise GeometryError(
            "CRN-4002",
            f"Array {name!r} has a near-zero-norm quaternion at index {idx}; "
            f"norm={norms[idx]:.3e}",
            stage="rotations",
        )


# ---------------------------------------------------------------------------
# Quaternion ops  (w, x, y, z)
# ---------------------------------------------------------------------------


def quat_normalize(
    q: np.ndarray[Any, np.dtype[Any]],
    name: str = "q",
) -> np.ndarray[Any, np.dtype[Any]]:
    """Return unit quaternions; shape ``(..., 4)`` preserved.

    Args:
        q: ``(..., 4)`` float array ``[w, x, y, z]``.
        name: label used in error messages.

    Returns:
        ``(..., 4)`` unit quaternions with the same dtype and leading shape.

    Raises:
        GeometryError("CRN-4002"): if any quaternion has near-zero norm or is non-finite.
    """
    q = _promote(np.asarray(q))
    _check_quat_norm(q, name)
    unit: np.ndarray[Any, np.dtype[Any]] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return unit


def quat_conjugate(
    q: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Return the conjugate ``(w, -x, -y, -z)``; shape ``(..., 4)`` preserved.

    Args:
        q: ``(..., 4)`` float array ``[w, x, y, z]``.  Need not be unit norm.

    Returns:
        ``(..., 4)`` conjugate quaternions.
    """
    q = _promote(np.asarray(q))
    conj = q.copy()
    conj[..., 1:] = -q[..., 1:]
    return conj


def quat_multiply(
    q1: np.ndarray[Any, np.dtype[Any]],
    q2: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Hamilton product ``q1 ⊗ q2``; shapes must broadcast on leading dims.

    Args:
        q1: ``(..., 4)`` float array ``[w, x, y, z]``.
        q2: ``(..., 4)`` float array ``[w, x, y, z]``.

    Returns:
        ``(..., 4)`` product quaternion.
    """
    q1 = _promote(np.asarray(q1))
    q2 = _promote(np.asarray(q2))
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(
    q: np.ndarray[Any, np.dtype[Any]],
    v: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Rotate vectors ``v`` by unit quaternions ``q``.

    Uses the sandwich product ``q ⊗ [0,v] ⊗ q*``.  Leading dimensions must
    broadcast.

    Args:
        q: ``(..., 4)`` unit quaternion ``[w, x, y, z]``.
        v: ``(..., 3)`` vectors in the same frame.

    Returns:
        ``(..., 3)`` rotated vectors.

    Raises:
        GeometryError("CRN-4002"): if *q* is degenerate or non-finite.
    """
    q = quat_normalize(_promote(np.asarray(q)), name="q")
    v = _promote(np.asarray(v))
    _check_finite(v, "v")
    # Extend v to a pure quaternion [0, vx, vy, vz]
    zero = np.zeros((*v.shape[:-1], 1), dtype=v.dtype)
    v_quat = np.concatenate([zero, v], axis=-1)
    qv = quat_multiply(q, v_quat)
    qvq = quat_multiply(qv, quat_conjugate(q))
    return qvq[..., 1:]


def quat_to_matrix(
    q: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert unit quaternions ``(..., 4)`` → rotation matrices ``(..., 3, 3)``.

    Args:
        q: ``(..., 4)`` float array ``[w, x, y, z]``.

    Returns:
        ``(..., 3, 3)`` rotation matrices (det = +1, orthonormal).

    Raises:
        GeometryError("CRN-4002"): if *q* is degenerate or non-finite.
    """
    q = quat_normalize(_promote(np.asarray(q)), name="q")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Standard closed-form (Shepperd 1978)
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape((*q.shape[:-1], 3, 3))
    # NumPy 1.x promotes ``float32_scalar * python_int`` to float64 (NumPy 2's NEP 50
    # promotion keeps it float32), so the closed form above can silently widen the result.
    # The documented contract is dtype preservation; restore it explicitly.
    return R.astype(q.dtype, copy=False)


def matrix_to_quat(
    R: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert rotation matrices ``(..., 3, 3)`` → unit quaternions ``(..., 4)``.

    Uses the Shepperd method with numerical stability.  The sign of *w* is
    always non-negative (canonical half-space).

    Args:
        R: ``(..., 3, 3)`` float rotation matrices.

    Returns:
        ``(..., 4)`` unit quaternions ``[w, x, y, z]``.

    Raises:
        GeometryError("CRN-4002"): if *R* contains non-finite values.
    """
    R = _promote(np.asarray(R))
    _check_finite(R, "R")
    batch = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0] if R_flat.ndim == 3 else 1

    quats = np.empty((max(n, 1), 4), dtype=R.dtype)
    for i in range(R_flat.shape[0]):
        m = R_flat[i]
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = 0.5 / float(np.sqrt(trace + 1.0))
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quats[i] = [w, x, y, z]

    # Canonical half-space: w >= 0
    quats[quats[:, 0] < 0] *= -1
    # Normalise
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / norms
    result: np.ndarray[Any, np.dtype[Any]] = quats.reshape((*batch, 4)).astype(R.dtype)
    return result


# ---------------------------------------------------------------------------
# Axis-angle
# ---------------------------------------------------------------------------


def quat_to_axis_angle(
    q: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert unit quaternions ``(..., 4)`` → axis-angle ``(..., 3)`` in radians.

    The axis-angle vector is ``θ * axis``, computed with ``2*arctan2(|v|, w)`` so that rotations
    arbitrarily close to the identity keep their small angle instead of collapsing to zero.  At
    ``θ = 0`` the result is the zero vector.

    Args:
        q: ``(..., 4)`` float array ``[w, x, y, z]``.

    Returns:
        ``(..., 3)`` axis-angle in radians.

    Raises:
        GeometryError("CRN-4002"): if *q* is degenerate or non-finite.
    """
    q = quat_normalize(_promote(np.asarray(q)), name="q")
    w = q[..., 0]
    xyz = q[..., 1:]
    # A quaternion and its negation are the same rotation, so flipping both signs is exact and
    # gives the shortest arc (w >= 0).
    flip = (w < 0)[..., np.newaxis]
    w = np.where(w < 0, -w, w)
    xyz = np.where(flip, -xyz, xyz)

    # theta = 2*arctan2(|v|, w) is accurate at every angle.  The equivalent 2*arccos(w) is not:
    # near the identity cos(theta/2) rounds to exactly 1.0, so a rotation of 1e-8 rad becomes
    # indistinguishable from none - and a near-still wrist is exactly that case.
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    safe = norm > 0.0
    theta = 2.0 * np.arctan2(np.where(safe, norm, 0.0), w[..., np.newaxis])
    axis = np.where(safe, xyz / np.where(safe, norm, 1.0), np.zeros_like(xyz))
    result: np.ndarray[Any, np.dtype[Any]] = axis * theta
    return result


def axis_angle_to_quat(
    aa: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert axis-angle vectors ``(..., 3)`` → unit quaternions ``(..., 4)``.

    The magnitude of the input vector is the rotation angle in radians.
    The zero vector maps to the identity quaternion.

    Args:
        aa: ``(..., 3)`` axis-angle in radians.

    Returns:
        ``(..., 4)`` unit quaternions ``[w, x, y, z]``.

    Raises:
        GeometryError("CRN-4002"): if *aa* contains non-finite values.
    """
    aa = _promote(np.asarray(aa))
    _check_finite(aa, "aa")
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)  # (..., 1)
    w = np.cos(theta / 2.0)
    # sin(theta/2)/theta = 0.5 * sinc(theta/2pi).  Written this way it is exact at theta = 0 and
    # stays accurate for arbitrarily small angles, where dividing by theta (or switching to the
    # identity below a fixed threshold) drops a rotation that really is there.
    scale = 0.5 * np.sinc(theta / (2.0 * np.pi))
    xyz = aa * scale
    return np.concatenate([w, xyz], axis=-1)


def matrix_to_axis_angle(
    R: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert rotation matrices ``(..., 3, 3)`` → axis-angle ``(..., 3)``.

    Args:
        R: ``(..., 3, 3)`` float rotation matrices.

    Returns:
        ``(..., 3)`` axis-angle in radians.

    Raises:
        GeometryError("CRN-4002"): if *R* contains non-finite values.
    """
    return quat_to_axis_angle(matrix_to_quat(R))


def axis_angle_to_matrix(
    aa: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert axis-angle ``(..., 3)`` → rotation matrices ``(..., 3, 3)``.

    Args:
        aa: ``(..., 3)`` axis-angle in radians.

    Returns:
        ``(..., 3, 3)`` rotation matrices (det = +1, orthonormal).

    Raises:
        GeometryError("CRN-4002"): if *aa* contains non-finite values.
    """
    return quat_to_matrix(axis_angle_to_quat(aa))


# ---------------------------------------------------------------------------
# Rot6d  (first two rows of R, row-major)
# ---------------------------------------------------------------------------


def matrix_to_rot6d(
    R: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Extract the rot6d representation ``(..., 6)`` from rotation matrices.

    ``rot6d = concat(R[0, :], R[1, :])`` - first two rows, row-major.
    This is the pytorch3d ``matrix_to_rotation_6d`` convention.

    Args:
        R: ``(..., 3, 3)`` float rotation matrices.

    Returns:
        ``(..., 6)`` rot6d representation.
    """
    R = _promote(np.asarray(R))
    _check_finite(R, "R")
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def rot6d_to_matrix(
    r6: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Reconstruct rotation matrices from rot6d via Gram-Schmidt.

    Given ``r6 = [a1, a2]`` (each ``(..., 3)``), compute:
      ``b1 = a1 / ||a1||``,
      ``b2 = (a2 - (b1·a2)·b1) / ||(a2 - (b1·a2)·b1)||``,
      ``b3 = b1 x b2``.

    Args:
        r6: ``(..., 6)`` rot6d representation.

    Returns:
        ``(..., 3, 3)`` rotation matrices.

    Raises:
        GeometryError("CRN-4002"): if *r6* contains non-finite values or degenerate columns.
    """
    r6 = _promote(np.asarray(r6))
    _check_finite(r6, "r6")
    a1 = r6[..., :3]
    a2 = r6[..., 3:]

    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(n1 < _NORM_EPS):
        raise GeometryError(
            "CRN-4002",
            "rot6d first column has near-zero norm; cannot reconstruct rotation",
            stage="rotations",
        )
    b1 = a1 / n1

    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    a2_orth = a2 - dot * b1
    n2 = np.linalg.norm(a2_orth, axis=-1, keepdims=True)
    if np.any(n2 < _NORM_EPS):
        raise GeometryError(
            "CRN-4002",
            "rot6d second column is parallel to first (after Gram-Schmidt); "
            "cannot reconstruct rotation",
            stage="rotations",
        )
    b2 = a2_orth / n2

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def quat_to_rot6d(
    q: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert unit quaternions ``(..., 4)`` → rot6d ``(..., 6)``.

    Args:
        q: ``(..., 4)`` float array ``[w, x, y, z]``.

    Returns:
        ``(..., 6)`` rot6d representation.

    Raises:
        GeometryError("CRN-4002"): if *q* is degenerate or non-finite.
    """
    return matrix_to_rot6d(quat_to_matrix(q))


def rot6d_to_quat(
    r6: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert rot6d ``(..., 6)`` → unit quaternions ``(..., 4)``.

    Args:
        r6: ``(..., 6)`` rot6d representation.

    Returns:
        ``(..., 4)`` unit quaternions ``[w, x, y, z]``.

    Raises:
        GeometryError("CRN-4002"): if *r6* contains non-finite values or degenerate columns.
    """
    return matrix_to_quat(rot6d_to_matrix(r6))


# ---------------------------------------------------------------------------
# Slerp
# ---------------------------------------------------------------------------


def slerp(
    q0: np.ndarray[Any, np.dtype[Any]],
    q1: np.ndarray[Any, np.dtype[Any]],
    t: float | np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Spherical linear interpolation (shortest path) between two quaternions.

    ``slerp(q0, q1, 0) = q0`` and ``slerp(q0, q1, 1) = q1`` exactly.
    The path is the shortest arc on the unit 3-sphere.

    Args:
        q0: ``(..., 4)`` unit quaternion ``[w, x, y, z]``.
        q1: ``(..., 4)`` unit quaternion ``[w, x, y, z]``.
        t:  Scalar or ``(...)`` array of interpolation parameters in ``[0, 1]``.

    Returns:
        ``(..., 4)`` interpolated unit quaternions.

    Raises:
        GeometryError("CRN-4002"): if either quaternion is degenerate or non-finite.
    """
    q0 = quat_normalize(_promote(np.asarray(q0)), name="q0")
    q1 = quat_normalize(_promote(np.asarray(q1)), name="q1")
    t_arr = np.asarray(t, dtype=q0.dtype)

    # Ensure shortest path: flip q1 if dot < 0
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    # For nearly-parallel quaternions use linear interp to avoid div-by-zero
    THRESH = 1.0 - 1e-10
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_omega = np.sin(omega)

    t_b = t_arr[..., np.newaxis] if t_arr.ndim < q0.ndim else t_arr

    denom = np.where(sin_omega > 1e-15, sin_omega, 1.0)
    w0 = np.where(dot > THRESH, 1.0 - t_b, np.sin((1.0 - t_b) * omega) / denom)
    w1 = np.where(dot > THRESH, t_b, np.sin(t_b * omega) / denom)

    out = w0 * q0 + w1 * q1
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    result: np.ndarray[Any, np.dtype[Any]] = out / np.where(norms > 1e-15, norms, 1.0)
    return result


# ---------------------------------------------------------------------------
# Orthonormalise
# ---------------------------------------------------------------------------


def orthonormalize(
    R: np.ndarray[Any, np.dtype[Any]],
) -> tuple[np.ndarray[Any, np.dtype[Any]], float]:
    """Project a near-rotation matrix onto SO(3) via SVD.

    Args:
        R: ``(..., 3, 3)`` float matrix (may have small numerical drift).

    Returns:
        ``(R_clean, residual)`` where *R_clean* is the nearest rotation matrix
        and *residual* = ``float(np.abs(R_clean - R).max())``.

    Raises:
        GeometryError("CRN-4002"): if *R* contains non-finite values.
    """
    R = _promote(np.asarray(R))
    _check_finite(R, "R")
    R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    out = np.empty_like(R_flat)
    for i in range(R_flat.shape[0]):
        U, _s, Vt = np.linalg.svd(R_flat[i])
        d = np.linalg.det(U @ Vt)
        D = np.diag(np.array([1.0, 1.0, d], dtype=R.dtype))
        out[i] = U @ D @ Vt
    R_clean = out.reshape(R.shape)
    residual = float(np.abs(R_clean - R).max())
    return R_clean, residual
