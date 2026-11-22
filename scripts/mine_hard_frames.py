"""MOAT B2 — cross-model disagreement as a free hard-frame miner (§46.3).

ACE and EgoHand both predict 21-joint 2-D hands. Where they disagree, the frame is hard; ranking
frames by disagreement localises the labelling budget without a single human label. This script
consumes the two preserved prediction artifacts and emits a ranked ``hard_frames.csv`` plus a
``mining_receipt.json`` whose budget curve says what fraction of the total disagreement mass the
top-1% / 5% / 10% of frames capture — the number that justifies labelling only the top slice.

Two correctness traps are handled explicitly, because each already produced one discarded report:

1. **ACE ``joints_2d`` are NORMALIZED [0, 1] of the frame**, not pixels of its encode grid. They are
   rescaled by ``--width``/``--height`` (default 1920x1080) before any distance is taken.
2. **EgoHand zero-filled wrists.** A present wrist whose ``kpts2d_px[..., 0, :]`` is ``(0, 0)`` is a
   placeholder, not a measurement. The wrist gate is
   ``valid_2d[..., 0] & (kpts2d_px[..., 0, :] != 0).any(-1)``; gated wrists never enter a distance.

Only **co-claimed** slots are compared, matched **by slot index** (slot 0 ↔ slot 0, slot 1 ↔ slot 1).
Every excluded slot is counted and named in the receipt: ACE-only, EgoHand-only, invalid wrist,
zero-filled wrist, non-finite ACE wrist. The per-frame score combines the mean wrist distance, the
mean valid-joint distance, and the claim-set difference (slots exactly one model claims), weighted by
:data:`CLAIM_DIFF_WEIGHT_PX`.

**Coverage caveat, printed and recorded:** disagreement is not error. Two models can agree and both
be wrong; the receipt carries ``disagreement_is_not_error: true``. No prediction artifact is written.

Usage:
    python scripts/mine_hard_frames.py \
        --ace .omp/qa/prod/gpu-rerun/live-run/IMG_5949/ace/frames.pkl \
        --egohand .omp/qa/prod/gpu-rerun/live-run/IMG_5949/hands2d.npz
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

from crinis.eval.label_quality import DEFAULT_SEED, DRAWS, PERCENTILES
from crinis.io.storage import atomic_write_json, sha256_file

#: Output receipt schema.
MINING_SCHEMA = "crinis.hard-frame-mining/1"

#: Pixel-equivalent weight of a one-slot claim-set difference in the per-frame score. A hand one
#: model claims and the other does not is a large disagreement; this puts it on the pixel scale.
CLAIM_DIFF_WEIGHT_PX = 50.0

#: The budget-curve fractions of frames whose captured disagreement mass is reported.
BUDGET_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)

#: The wrist distance (px) used for the headline "fraction within" summary.
WITHIN_PX = 50.0

#: Clip identity inferred from an artifact path component, e.g. ``IMG_5949``.
_CLIP_RE = re.compile(r"IMG_\d+")


def rescale_ace(joints_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rescale ACE normalized ``[0, 1]`` joints to pixels of the ``width x height`` frame.

    The x axis scales by *width* and the y axis by *height*; this is the trap that produced a bad
    "0 px jitter" report when the encode grid was used instead of the frame.
    """
    scale = np.array([float(width), float(height)], dtype=np.float64)
    return np.asarray(joints_norm, dtype=np.float64) * scale


def wrist_gate(valid_2d: np.ndarray, kpts_px: np.ndarray) -> np.ndarray:
    """ ``(T, 2)`` mask of EgoHand wrists that are valid AND not the ``(0, 0)`` zero-fill placeholder."""
    wrist_valid = np.asarray(valid_2d[..., 0], dtype=bool)
    nonzero = (np.asarray(kpts_px[..., 0, :]) != 0).any(axis=-1)
    return wrist_valid & nonzero


def co_claim_mask(
    ace_claim: np.ndarray,
    ego_present: np.ndarray,
    ego_wrist_ok: np.ndarray,
    ace_wrist_finite: np.ndarray,
) -> np.ndarray:
    """Slots both models claim with a usable EgoHand wrist and a finite ACE wrist, matched by slot."""
    return (
        np.asarray(ace_claim, dtype=bool)
        & np.asarray(ego_present, dtype=bool)
        & np.asarray(ego_wrist_ok, dtype=bool)
        & np.asarray(ace_wrist_finite, dtype=bool)
    )


def frame_disagreement(
    ace_px: np.ndarray,
    ego_kpts: np.ndarray,
    ego_valid: np.ndarray,
    co_mask: np.ndarray,
    ace_claim: np.ndarray,
    ego_present: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-frame disagreement components over the co-claimed, slot-matched instances.

    Returns arrays keyed ``score``, ``wrist_dist_px``, ``mean_joint_dist_px``, ``claim_set_diff``,
    ``n_compared``. The wrist and joint components average over co-claimed slots (0 when none); the
    claim-set difference counts slots exactly one model claims. Joint distances use EgoHand-valid,
    non-zero joints so a placeholder never contaminates the mean.
    """
    n_frames = co_mask.shape[0]
    scores = np.zeros(n_frames, dtype=np.float64)
    wrist_comp = np.zeros(n_frames, dtype=np.float64)
    joint_comp = np.zeros(n_frames, dtype=np.float64)
    claim_diff = np.zeros(n_frames, dtype=np.int64)
    n_compared = np.zeros(n_frames, dtype=np.int64)
    joint_nonzero = (ego_kpts != 0).any(axis=-1)
    joint_usable = np.asarray(ego_valid, dtype=bool) & joint_nonzero
    for frame in range(n_frames):
        slots = np.flatnonzero(co_mask[frame])
        n_compared[frame] = slots.size
        claim_diff[frame] = int(np.count_nonzero(ace_claim[frame] ^ ego_present[frame]))
        if slots.size:
            wrist_d = np.linalg.norm(ace_px[frame, slots, 0, :] - ego_kpts[frame, slots, 0, :], axis=-1)
            wrist_comp[frame] = float(wrist_d.mean())
            joint_means = []
            for slot in slots:
                usable = joint_usable[frame, slot]
                if not bool(usable.any()):
                    continue
                d = np.linalg.norm(ace_px[frame, slot, usable] - ego_kpts[frame, slot, usable], axis=-1)
                joint_means.append(float(d.mean()))
            if joint_means:
                joint_comp[frame] = float(np.mean(joint_means))
        scores[frame] = wrist_comp[frame] + joint_comp[frame] + CLAIM_DIFF_WEIGHT_PX * claim_diff[frame]
    return {
        "score": scores,
        "wrist_dist_px": wrist_comp,
        "mean_joint_dist_px": joint_comp,
        "claim_set_diff": claim_diff,
        "n_compared": n_compared,
    }


def coverage_fraction(scores_desc: np.ndarray, k: int) -> float:
    """Fraction of total disagreement mass in the top ``k`` of a descending-sorted score array."""
    total = float(scores_desc.sum())
    if total <= 0.0 or k <= 0:
        return 0.0
    k = min(k, scores_desc.size)
    return float(scores_desc[:k].sum()) / total


def _top_k(fraction: float, n: int) -> int:
    return max(1, math.ceil(fraction * n)) if n > 0 else 0


def budget_curve(
    scores: np.ndarray,
    fractions: tuple[float, ...] = BUDGET_FRACTIONS,
    seed: int = DEFAULT_SEED,
    draws: int = DRAWS,
) -> dict[str, object]:
    """Top-fraction coverage of the disagreement mass, with a seeded bootstrap CI on each coverage.

    Each bootstrap draw resamples the frames with replacement, re-sorts, and recomputes the coverage
    of the top ``ceil(fraction * n)``; the percentile interval uses the shared bootstrap convention.
    Draws are processed in batches so the ``(batch, n)`` index stays bounded on a long clip.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    order = np.argsort(-scores, kind="stable")
    scores_desc = scores[order]
    total = float(scores.sum())
    rng = np.random.default_rng(seed)
    curve: dict[str, object] = {}
    batch = max(1, min(draws, 2_000_000 // max(n, 1)))
    for fraction in fractions:
        k = _top_k(fraction, n)
        point = coverage_fraction(scores_desc, k)
        coverages = np.empty(draws, dtype=np.float64)
        produced = 0
        while produced < draws:
            size = min(batch, draws - produced)
            index = rng.integers(0, n, size=(size, n)) if n > 0 else np.zeros((size, 0), dtype=np.int64)
            resampled = scores[index]
            resampled.sort(axis=1)
            resampled = resampled[:, ::-1]
            denom = resampled.sum(axis=1)
            numer = resampled[:, :k].sum(axis=1) if k > 0 else np.zeros(size)
            with np.errstate(divide="ignore", invalid="ignore"):
                cov = np.where(denom > 0.0, numer / denom, 0.0)
            coverages[produced : produced + size] = cov
            produced += size
        lo, hi = np.percentile(coverages, PERCENTILES)
        curve[f"top_{fraction * 100:g}pct"] = {
            "fraction_of_frames": fraction,
            "frames": k,
            "coverage": point,
            "ci95": [float(lo), float(hi)],
        }
    curve["total_disagreement_mass"] = total
    curve["bootstrap"] = {"seed": seed, "draws": draws, "percentiles": list(PERCENTILES)}
    return curve


def exclusion_counts(
    ace_claim: np.ndarray,
    ego_present: np.ndarray,
    ego_valid: np.ndarray,
    ego_kpts: np.ndarray,
    ace_wrist_finite: np.ndarray,
) -> dict[str, int]:
    """Named counts of every slot kept out of the co-claimed comparison, and why."""
    claim = np.asarray(ace_claim, dtype=bool)
    present = np.asarray(ego_present, dtype=bool)
    wrist_valid = np.asarray(ego_valid[..., 0], dtype=bool)
    nonzero = (np.asarray(ego_kpts[..., 0, :]) != 0).any(axis=-1)
    both = claim & present
    return {
        "ace_only_slots": int(np.count_nonzero(claim & ~present)),
        "egohand_only_slots": int(np.count_nonzero(~claim & present)),
        "co_claimed_invalid_wrist": int(np.count_nonzero(both & ~wrist_valid)),
        "co_claimed_zero_filled_wrist": int(np.count_nonzero(both & wrist_valid & ~nonzero)),
        "co_claimed_nonfinite_ace_wrist": int(
            np.count_nonzero(both & wrist_valid & nonzero & ~ace_wrist_finite)
        ),
    }


def _infer_clip(*paths: Path) -> str | None:
    for path in paths:
        for part in path.parts:
            match = _CLIP_RE.fullmatch(part)
            if match:
                return match.group(0)
    return None


def _join_surface(surface_path: Path, clip: str | None) -> dict[int, tuple[str, str]]:
    """``frame -> (pose_quality, presence_confidence)`` for the clip, lowest-slot row winning."""
    join: dict[int, tuple[str, str]] = {}
    if clip is None:
        return join
    best_slot: dict[int, int] = {}
    with surface_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["clip"] != clip:
                continue
            try:
                frame = int(row["frame"])
                slot = int(row["slot"])
            except ValueError:
                continue
            if frame not in best_slot or slot < best_slot[frame]:
                best_slot[frame] = slot
                join[frame] = (row["pose_quality"], row["presence_confidence"])
    return join


def write_hard_frames(
    path: Path,
    components: dict[str, np.ndarray],
    order: np.ndarray,
    surface_join: dict[int, tuple[str, str]],
) -> None:
    """Write the ranked hard-frame table, descending by score, with the joined tier when present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame",
        "score",
        "wrist_dist_px",
        "mean_joint_dist_px",
        "claim_set_diff",
        "n_compared",
        "pose_quality",
        "presence_confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in order:
            tier = surface_join.get(int(frame))
            writer.writerow(
                {
                    "frame": int(frame),
                    "score": f"{components['score'][frame]:.6f}",
                    "wrist_dist_px": f"{components['wrist_dist_px'][frame]:.6f}",
                    "mean_joint_dist_px": f"{components['mean_joint_dist_px'][frame]:.6f}",
                    "claim_set_diff": int(components["claim_set_diff"][frame]),
                    "n_compared": int(components["n_compared"][frame]),
                    "pose_quality": tier[0] if tier else "",
                    "presence_confidence": tier[1] if tier else "",
                }
            )


def mine(
    ace_path: Path,
    egohand_path: Path,
    width: int,
    height: int,
    surface_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object], dict[int, tuple[str, str]]]:
    """Load both artifacts, gate, compare co-claimed slots, and build the receipt body.

    Returns ``(components, descending_frame_order, receipt, surface_join)``. Pure of any output writing so the
    real-artifact replay test can assert on the computed numbers directly.
    """
    with ace_path.open("rb") as handle:
        ace = pickle.load(handle)
    ego = np.load(egohand_path, allow_pickle=False)
    ace_claim_full = np.asarray(ace["claim"], dtype=bool)
    ace_joints_full = np.asarray(ace["joints_2d"], dtype=np.float64)
    ego_present_full = np.asarray(ego["present2d"], dtype=bool)
    ego_valid_full = np.asarray(ego["valid_2d"], dtype=bool)
    ego_kpts_full = np.asarray(ego["kpts2d_px"], dtype=np.float64)
    ego.close()

    ace_frames = ace_claim_full.shape[0]
    ego_frames = ego_present_full.shape[0]
    n_frames = min(ace_frames, ego_frames)
    ace_claim = ace_claim_full[:n_frames]
    ace_px = rescale_ace(ace_joints_full[:n_frames], width, height)
    ego_present = ego_present_full[:n_frames]
    ego_valid = ego_valid_full[:n_frames]
    ego_kpts = ego_kpts_full[:n_frames]

    ego_wrist_ok = wrist_gate(ego_valid, ego_kpts)
    ace_wrist_finite = np.isfinite(ace_px[..., 0, :]).all(axis=-1)
    co_mask = co_claim_mask(ace_claim, ego_present, ego_wrist_ok, ace_wrist_finite)

    components = frame_disagreement(ace_px, ego_kpts, ego_valid, co_mask, ace_claim, ego_present)
    order = np.argsort(-components["score"], kind="stable")

    co_wrist_dist = np.linalg.norm(ace_px[..., 0, :][co_mask] - ego_kpts[..., 0, :][co_mask], axis=-1)
    n_co = int(co_mask.sum())
    exclusions = exclusion_counts(ace_claim, ego_present, ego_valid, ego_kpts, ace_wrist_finite)
    clip = _infer_clip(ace_path, egohand_path)
    surface_join = _join_surface(surface_path, clip) if surface_path is not None else {}

    wrist_summary: dict[str, object] = {"n": n_co}
    if n_co:
        p10, p50, p90 = (float(v) for v in np.percentile(co_wrist_dist, (10.0, 50.0, 90.0)))
        wrist_summary.update(
            {
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "mean": float(co_wrist_dist.mean()),
                "fraction_within_50px": float(np.count_nonzero(co_wrist_dist <= WITHIN_PX) / n_co),
            }
        )
    else:
        wrist_summary.update(
            {"p10": None, "p50": None, "p90": None, "mean": None, "fraction_within_50px": None}
        )

    receipt: dict[str, object] = {
        "schema": MINING_SCHEMA,
        "ace": str(ace_path),
        "ace_sha256": sha256_file(ace_path),
        "egohand": str(egohand_path),
        "egohand_sha256": sha256_file(egohand_path),
        "width": width,
        "height": height,
        "clip": clip,
        "ace_frames": ace_frames,
        "egohand_frames": ego_frames,
        "frames_compared": n_frames,
        "frames_dropped_axis_mismatch": max(ace_frames, ego_frames) - n_frames,
        "disagreement_is_not_error": True,
        "caveat": (
            "Disagreement localises hard frames without labels; it is not error. Two models can agree "
            "and both be wrong, so a low-disagreement frame is not certified correct. Label the top "
            "slice to measure actual error."
        ),
        "claim_diff_weight_px": CLAIM_DIFF_WEIGHT_PX,
        "co_claimed_wrists": n_co,
        "wrist_distance_px": wrist_summary,
        "exclusions": exclusions,
        "surface_joined": surface_path is not None and bool(surface_join),
        "budget_curve": budget_curve(components["score"]),
    }
    return components, order, receipt, surface_join


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Writes the ranked CSV and the receipt; returns 0 on success."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ace", required=True, type=Path, help="ACE frames.pkl (normalized joints_2d)")
    parser.add_argument("--egohand", required=True, type=Path, help="EgoHand hands2d.npz (pixel kpts2d_px)")
    parser.add_argument(
        "--surface", default=None, type=Path, help="optional quality surface CSV to join tiers"
    )
    parser.add_argument("--out", default=Path("hard_frames.csv"), type=Path, help="ranked hard-frame CSV")
    parser.add_argument(
        "--receipt", default=Path("mining_receipt.json"), type=Path, help="mining receipt JSON"
    )
    parser.add_argument("--width", default=1920, type=int, help="frame width for the ACE rescale")
    parser.add_argument("--height", default=1080, type=int, help="frame height for the ACE rescale")
    args = parser.parse_args(argv)
    if not args.ace.is_file():
        print(f"error: ACE artifact '{args.ace}' does not exist", file=sys.stderr)
        return 2
    if not args.egohand.is_file():
        print(f"error: EgoHand artifact '{args.egohand}' does not exist", file=sys.stderr)
        return 2

    components, order, receipt, surface_join = mine(
        args.ace, args.egohand, args.width, args.height, args.surface
    )
    write_hard_frames(args.out, components, order, surface_join)
    atomic_write_json(args.receipt, receipt)

    wrist = receipt["wrist_distance_px"]
    assert isinstance(wrist, dict)
    print(
        f"hard frames: {receipt['frames_compared']} frames, "
        f"{receipt['co_claimed_wrists']} co-claimed wrists"
    )
    print(f"wrist p50 {wrist['p50']} px, fraction within {WITHIN_PX:g}px {wrist['fraction_within_50px']}")
    curve = receipt["budget_curve"]
    assert isinstance(curve, dict)
    for fraction in BUDGET_FRACTIONS:
        entry = curve[f"top_{fraction * 100:g}pct"]
        assert isinstance(entry, dict)
        ci = entry["ci95"]
        print(
            f"top {fraction * 100:g}% ({entry['frames']} frames) captures "
            f"{entry['coverage']:.3f} of disagreement mass [{ci[0]:.3f}, {ci[1]:.3f}]"
        )
    print("disagreement_is_not_error: true — agreement does not certify correctness.")
    print(f"wrote {args.out} and {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
