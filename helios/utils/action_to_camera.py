"""Action -> camera trajectory utilities for Helios inference.

Adapted from ``lingbot-world/wan/utils/wasd_ijkl_to_c2ws.py`` so that we can
drive the Plucker FiLM head at inference time even when no recorded camera
trajectory is available. The math (move/rotate per pressed key) is a verbatim
port; the Helios-specific additions are:

  * ``helios_action_to_wasd_ijkl`` — translate Helios's ``(action_keys,
    action_mouse)`` labels (e.g. "W+A", "↑→") into the (4,) WASD and (4,)
    IJKL one-hot vectors used by the lingbot kinematics.

  * ``build_camera_trajectory_from_actions`` — given a per-chunk Helios
    action schedule plus chunk count / frames-per-chunk, return the
    ``camera_poses_per_chunk`` and ``camera_intrinsics_per_chunk`` lists
    expected by ``HeliosPipeline.__call__``. Each chunk is independent
    (poses start from identity in its own local frame) because the
    transformer normalises per-chunk via ``compute_relative_poses``.

  * ``DEFAULT_MOVE_SPEED`` / ``DEFAULT_ROTATE_SPEED_RAD`` — calibrated
    against ``data/seadance2_action_1p`` per-chunk motion stats so the
    synthesised trajectories sit near the training median rather than the
    aggressive lingbot defaults (which would be ~5× the training q75 for
    rotation).

  * ``DEFAULT_INTRINSICS_384x640`` — sane fx/fy/cx/cy at the Helios
    training grid; used when a caller doesn't supply intrinsics.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------
# Calibrated defaults (see docstring above for provenance)
# ---------------------------------------------------------------------

# Translation gets normalised away by ``compute_relative_poses(normalize_trans=True)``
# at chunk granularity, so the absolute value is largely cosmetic; we keep the
# lingbot default which produces a clean linear trajectory.
DEFAULT_MOVE_SPEED: float = 0.05

# Rotation is NOT normalised. Training set median per-chunk rotation is ~5°,
# q75 ≈ 13°, so ~0.4°/frame × 33 frames ≈ 13° matches the upper quartile of
# real motion. Lingbot's 2°/frame would give ~66°/chunk — far above q90 (~23°).
DEFAULT_ROTATE_SPEED_RAD: float = float(np.deg2rad(0.5))

# Median seadance2 training intrinsics rescaled to 384×640: fx≈484, fy≈516,
# cx=320, cy=192 (sample of 200 videos). We round to fx=fy=500, which keeps
# horizontal FOV ≈ 65° and vertical ≈ 41° — within the q25–q75 band.
DEFAULT_INTRINSICS_384x640: Tuple[float, float, float, float] = (
    500.0, 500.0, 320.0, 192.0,
)


# ---------------------------------------------------------------------
# Helios action labels → lingbot WASD/IJKL frames
# ---------------------------------------------------------------------

_HELIOS_KEYBOARD_TO_WASD = {
    "None": (),
    "W": ("w",),
    "S": ("s",),
    "A": ("a",),
    "D": ("d",),
    "W+A": ("w", "a"),
    "W+D": ("w", "d"),
    "S+A": ("s", "a"),
    "S+D": ("s", "d"),
}

# Mouse arrow → lingbot ijkl pitch/yaw key set.
#   ↑ = pitch up    (i)
#   ↓ = pitch down  (k)
#   ← = yaw left    (j)
#   → = yaw right   (l)
_HELIOS_MOUSE_TO_IJKL = {
    "·": (),
    "→": ("l",),
    "←": ("j",),
    "↑": ("i",),
    "↓": ("k",),
    "↑→": ("i", "l"),
    "↑←": ("i", "j"),
    "↓→": ("k", "l"),
    "↓←": ("k", "j"),
}


def helios_action_to_wasd_ijkl(keys: str, mouse: str) -> Tuple[List[str]]:
    """Return the list of lingbot key-chars implied by one (keys, mouse) pair."""
    wasd = _HELIOS_KEYBOARD_TO_WASD.get(keys, ())
    ijkl = _HELIOS_MOUSE_TO_IJKL.get(mouse, ())
    return list(wasd) + list(ijkl)


# ---------------------------------------------------------------------
# Lingbot-world kinematics (verbatim port from
# lingbot-world/wan/utils/wasd_ijkl_to_c2ws.py:138-210)
# ---------------------------------------------------------------------


def _get_rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)


def generate_c2ws_from_keys(
    arrow_actions: Sequence[Sequence[str]],
    move_speed: float = DEFAULT_MOVE_SPEED,
    rotate_speed_rad: float = DEFAULT_ROTATE_SPEED_RAD,
    init_c2w: Optional[np.ndarray] = None,
    init_pitch: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simulate one frame per element of ``arrow_actions`` and return c2ws.

    Args:
        arrow_actions: list[list[str]] of length T; each inner list contains
            any subset of {w,a,s,d,i,j,k,l} that are "pressed" at frame t.
        move_speed: per-frame translation step in world units (x,z plane).
        rotate_speed_rad: per-frame yaw / pitch step in radians.
        init_c2w: optional (4,4) starting pose; defaults to identity.
        init_pitch: optional starting pitch (rad) so chunks can chain
            without violating the ±85° pitch clamp.

    Returns:
        c2ws: (T+1, 4, 4) numpy array — the initial pose plus one per frame.
        last_c2w: (4, 4) the final pose (so callers can chain chunks).
        last_pitch: float, the final pitch (rad) for chaining.
    """
    pitch_limit = np.deg2rad(85)
    current_c2w = np.eye(4) if init_c2w is None else init_c2w.copy()
    current_pitch = init_pitch

    matrices = [current_c2w.copy()]
    for frame_keys in arrow_actions:
        R = current_c2w[:3, :3]
        T = current_c2w[:3, 3]

        pitch_delta = 0.0
        if "i" in frame_keys:
            pitch_delta += rotate_speed_rad
        if "k" in frame_keys:
            pitch_delta -= rotate_speed_rad
        new_pitch = current_pitch + pitch_delta
        if -pitch_limit <= new_pitch <= pitch_limit:
            current_pitch = new_pitch
        else:
            pitch_delta = 0.0
        R_pitch = _get_rotation_matrix("x", pitch_delta)

        yaw_delta = 0.0
        if "j" in frame_keys:
            yaw_delta -= rotate_speed_rad
        if "l" in frame_keys:
            yaw_delta += rotate_speed_rad
        R_yaw = _get_rotation_matrix("y", yaw_delta)

        R_new = R_yaw @ R @ R_pitch

        vec_right = R_new[:, 0]
        vec_forward = R_new[:, 2]
        forward_flat = np.array([vec_forward[0], 0.0, vec_forward[2]])
        right_flat = np.array([vec_right[0], 0.0, vec_right[2]])
        f_norm = np.linalg.norm(forward_flat)
        r_norm = np.linalg.norm(right_flat)
        if f_norm > 0:
            forward_flat = forward_flat / (f_norm + 1e-6)
        if r_norm > 0:
            right_flat = right_flat / (r_norm + 1e-6)

        move_vec = np.zeros(3)
        if "w" in frame_keys:
            move_vec += forward_flat * move_speed
        if "s" in frame_keys:
            move_vec -= forward_flat * move_speed
        if "d" in frame_keys:
            move_vec += right_flat * move_speed
        if "a" in frame_keys:
            move_vec -= right_flat * move_speed

        T_new = T + move_vec
        current_c2w = np.eye(4)
        current_c2w[:3, :3] = R_new
        current_c2w[:3, 3] = T_new
        matrices.append(current_c2w.copy())

    return np.stack(matrices, axis=0), current_c2w, current_pitch


# ---------------------------------------------------------------------
# Action-string DSL (verbatim port from lingbot, kept for parity)
# ---------------------------------------------------------------------

_ALLOWED_KEYS = frozenset("wasdijkl")


def parse_action_string(action_string: str):
    """Parse "w-3,iw-1,none-5" -> list[(frozenset(keys), n_frames)]."""
    if not action_string:
        raise ValueError("action_string is empty")
    s = action_string.replace("，", ",")
    s = "".join(s.split())
    parts = [p for p in s.split(",") if p]
    segments = []
    for part in parts:
        if "-" not in part:
            raise ValueError(
                f"Invalid segment {part!r}: expected <keys>-<duration>")
        keys_part, dur_str = part.rsplit("-", 1)
        if not dur_str.isdigit():
            raise ValueError(f"Invalid duration in segment {part!r}: {dur_str!r}")
        n = int(dur_str)
        if n <= 0:
            raise ValueError(f"Duration must be positive in segment {part!r}")
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys = frozenset()
        else:
            bad = [c for c in keys_lower if c not in _ALLOWED_KEYS]
            if bad:
                raise ValueError(
                    f"Invalid key character(s) {bad!r} in segment {part!r}; "
                    "allowed: w,a,s,d,i,j,k,l")
            keys = frozenset(keys_lower)
        segments.append((keys, n))
    return segments


def action_string_to_frame_keys(action_string: str) -> List[List[str]]:
    """Expand "w-3,none-2" → [['w'],['w'],['w'],[],[]]."""
    out: List[List[str]] = []
    for keys, n in parse_action_string(action_string):
        for _ in range(n):
            out.append(sorted(keys))
    return out


# ---------------------------------------------------------------------
# Top-level Helios adapter
# ---------------------------------------------------------------------


def _intrinsics_to_tensor(
    intrinsics: Optional[Sequence[float] | np.ndarray | torch.Tensor],
) -> torch.Tensor:
    if intrinsics is None:
        intrinsics = DEFAULT_INTRINSICS_384x640
    if isinstance(intrinsics, torch.Tensor):
        ks = intrinsics.detach().to(torch.float32).cpu().reshape(-1)
    else:
        ks = torch.tensor(list(intrinsics), dtype=torch.float32).reshape(-1)
    if ks.numel() != 4:
        raise ValueError(
            f"intrinsics must be a 4-vector [fx,fy,cx,cy]; got numel={ks.numel()}")
    return ks


def build_camera_trajectory_from_actions(
    per_chunk_actions: Sequence[Tuple[str, str]],
    *,
    frames_per_chunk: int = 33,
    intrinsics: Optional[Sequence[float] | np.ndarray | torch.Tensor] = None,
    move_speed: float = DEFAULT_MOVE_SPEED,
    rotate_speed_rad: float = DEFAULT_ROTATE_SPEED_RAD,
    chain_chunks: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build per-chunk c2w + intrinsics lists from a Helios action schedule.

    Each chunk gets ``frames_per_chunk`` poses (33 by default to match
    Helios's 33→9 VAE compression). The default ``chain_chunks=False``
    starts each chunk from identity, which mirrors the training pipeline
    where ``compute_relative_poses`` re-anchors every chunk to its own
    frame 0 — so the per-chunk Plucker embedding only depends on local
    motion, not on global drift across chunks.

    Args:
        per_chunk_actions: list[(keys, mouse)] of length ``num_chunks``;
            each tuple is one Helios label pair like ("W", "→").
        frames_per_chunk: pixel frames per chunk (Helios uses 33).
        intrinsics: optional (4,) [fx,fy,cx,cy] in 384×640 pixel grid units
            (apply ``get_Ks_transformed`` before passing); defaults to
            ``DEFAULT_INTRINSICS_384x640``.
        move_speed / rotate_speed_rad: per-frame velocity knobs; see module
            docstring for calibration notes.
        chain_chunks: when True, the next chunk continues from the previous
            chunk's final pose. Has no effect on the Plucker FiLM signal
            (because of per-chunk normalisation) but is useful if the
            caller wants to visualise the full trajectory.

    Returns:
        (camera_poses_per_chunk, camera_intrinsics_per_chunk) — both lists
        of length ``len(per_chunk_actions)``. Each entry shape:
            poses:  (frames_per_chunk, 4, 4) float32
            Ks:     (4,) float32
        Suitable to pass directly to ``HeliosPipeline.__call__``.
    """
    if frames_per_chunk <= 1:
        raise ValueError("frames_per_chunk must be >= 2")
    ks = _intrinsics_to_tensor(intrinsics)

    poses_per_chunk: List[torch.Tensor] = []
    ks_per_chunk: List[torch.Tensor] = []

    init_c2w = None
    init_pitch = 0.0
    for keys, mouse in per_chunk_actions:
        per_frame_keys = [
            helios_action_to_wasd_ijkl(keys, mouse)
            for _ in range(frames_per_chunk - 1)
        ]
        # generate_c2ws_from_keys returns T+1 poses for T action frames.
        c2ws, last_c2w, last_pitch = generate_c2ws_from_keys(
            per_frame_keys,
            move_speed=move_speed,
            rotate_speed_rad=rotate_speed_rad,
            init_c2w=init_c2w,
            init_pitch=init_pitch,
        )
        assert c2ws.shape == (frames_per_chunk, 4, 4), c2ws.shape
        poses_per_chunk.append(torch.from_numpy(c2ws).to(torch.float32))
        ks_per_chunk.append(ks.clone())
        if chain_chunks:
            init_c2w = last_c2w
            init_pitch = last_pitch
    return poses_per_chunk, ks_per_chunk


# ---------------------------------------------------------------------
# Sampled-from-training intrinsics (closes the FOV-distribution gap when
# Mode B/C uses the constant DEFAULT_INTRINSICS_384x640 instead of the
# real per-video Ks the model saw at training time).
# ---------------------------------------------------------------------


def sample_intrinsics_from_pt_dir(
    pt_dir: str,
    sample_id: Optional[str] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Pick one (4,) intrinsics vector from a directory of training-format .pt files.

    Args:
        pt_dir: directory of *.pt files with ``camera_intrinsics`` field
                (shape (M, 33, 4) / (M, 4) / (4,) all accepted; we extract
                the first chunk's first frame).
        sample_id: when provided, hashed to a deterministic file index so
                the same sample_id always picks the same intrinsics across
                runs (useful for reproducible Mode B sweeps).
        seed: secondary entropy when sample_id is None — uses ``random.Random(seed)``.

    Returns:
        (4,) float32 tensor [fx, fy, cx, cy] in 384x640 grid.
    """
    import glob as _glob
    import hashlib as _hashlib
    import random as _random
    files = sorted(_glob.glob(os.path.join(pt_dir, "*.pt")))
    if not files:
        raise ValueError(f"No .pt files in {pt_dir}")
    if sample_id is not None:
        h = int(_hashlib.md5(sample_id.encode()).hexdigest(), 16)
        f = files[h % len(files)]
    else:
        f = _random.Random(seed).choice(files)
    blob = torch.load(f, map_location="cpu", weights_only=False)
    ki = blob.get("camera_intrinsics", None)
    if ki is None:
        raise ValueError(f"{f} has no 'camera_intrinsics' field")
    if ki.dim() == 3:    # (M, T, 4)
        k = ki[0, 0]
    elif ki.dim() == 2:  # (M, 4) or (T, 4)
        k = ki[0]
    else:                # (4,)
        k = ki
    return k.to(torch.float32).clone()


import os  # for the helper above (was already imported transitively but be explicit)


# ---------------------------------------------------------------------
# Loading from npy/.pt sources
# ---------------------------------------------------------------------


def load_trajectory_from_npy(
    poses_npy: str,
    intrinsics_npy: Optional[str] = None,
    *,
    num_chunks: int,
    frames_per_chunk: int = 33,
    fallback_intrinsics: Optional[Sequence[float]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load a pre-recorded poses.npy (+ optional intrinsics.npy) and split
    it into ``num_chunks`` × ``frames_per_chunk`` chunks.

    If the file has fewer than ``num_chunks * frames_per_chunk`` poses we
    repeat-pad the tail; if it has more, we truncate. Intrinsics are
    expected to already be in the 384×640 pixel grid (use
    ``helios.utils.camera_utils.get_Ks_transformed`` to rescale before
    saving).
    """
    poses = np.load(poses_npy)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"poses.npy must have shape (N, 4, 4); got {poses.shape}")
    target = num_chunks * frames_per_chunk
    if poses.shape[0] < target:
        pad = target - poses.shape[0]
        poses = np.concatenate([poses, np.repeat(poses[-1:], pad, axis=0)], axis=0)
    poses = poses[:target]

    if intrinsics_npy is not None:
        ks_full = np.load(intrinsics_npy).astype(np.float32)
        if ks_full.ndim == 1 and ks_full.shape[0] == 4:
            ks_per_frame = np.tile(ks_full[None, :], (target, 1))
        elif ks_full.ndim == 2 and ks_full.shape[1] == 4:
            if ks_full.shape[0] < target:
                pad = target - ks_full.shape[0]
                ks_full = np.concatenate(
                    [ks_full, np.repeat(ks_full[-1:], pad, axis=0)], axis=0)
            ks_per_frame = ks_full[:target]
        else:
            raise ValueError(
                f"intrinsics.npy must have shape (4,) or (N, 4); got {ks_full.shape}")
    else:
        if fallback_intrinsics is None:
            fallback_intrinsics = DEFAULT_INTRINSICS_384x640
        ks_per_frame = np.tile(
            np.asarray(fallback_intrinsics, dtype=np.float32)[None, :],
            (target, 1),
        )

    poses_per_chunk: List[torch.Tensor] = []
    ks_per_chunk: List[torch.Tensor] = []
    for k in range(num_chunks):
        a = k * frames_per_chunk
        b = a + frames_per_chunk
        poses_per_chunk.append(torch.from_numpy(poses[a:b]).to(torch.float32))
        ks_per_chunk.append(torch.from_numpy(ks_per_frame[a:b]).to(torch.float32))
    return poses_per_chunk, ks_per_chunk


def load_trajectory_from_pt(
    pt_path: str,
    *,
    num_chunks: int,
    frames_per_chunk: int = 33,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load camera_poses + camera_intrinsics from a Helios training-format
    .pt (same schema as in ``data/helios/*_chunked/*.pt``).

    Expects:
        camera_poses:       Tensor (M, 33, 4, 4)  per-pixel-frame c2w
        camera_intrinsics:  Tensor (M, 33, 4) or (M, 4) or (4,)  in
                            training-grid units.

    Returns the leading ``num_chunks`` (repeat-padding when fewer).
    """
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    cp = blob.get("camera_poses", None)
    if cp is None:
        raise ValueError(f"{pt_path} does not contain 'camera_poses'")
    if cp.dim() != 4 or cp.shape[1] != frames_per_chunk:
        raise ValueError(
            f"camera_poses must be (M,{frames_per_chunk},4,4); got {tuple(cp.shape)}")
    ci = blob.get("camera_intrinsics", None)
    if ci is None:
        ci = torch.tensor(DEFAULT_INTRINSICS_384x640, dtype=torch.float32)

    def _ki(k: int) -> torch.Tensor:
        if ci.dim() == 1:
            return ci.clone()
        if ci.dim() == 2:
            return ci[min(k, ci.shape[0] - 1)].clone()
        if ci.dim() == 3:
            return ci[min(k, ci.shape[0] - 1)].clone()
        raise ValueError(f"camera_intrinsics rank must be 1/2/3; got {ci.dim()}")

    poses_per_chunk: List[torch.Tensor] = []
    ks_per_chunk: List[torch.Tensor] = []
    for k in range(num_chunks):
        idx = min(k, cp.shape[0] - 1)
        poses_per_chunk.append(cp[idx].to(torch.float32).clone())
        ks_per_chunk.append(_ki(k).to(torch.float32))
    return poses_per_chunk, ks_per_chunk
