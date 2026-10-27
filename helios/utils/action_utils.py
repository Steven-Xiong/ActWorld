"""
Action encoding utilities for the interactive world model.

Converts discrete keyboard/mouse labels into:
  - action_id:  scalar int in [0, 80]  (9 keyboard x 9 mouse)
  - action_vec: float tensor [forward, right, yaw, pitch]

This module also provides action-memory primitives used by the
Dual-Memory Helios extension (branch idea1):
  - ActionTaxonomy / action_taxonomy()
  - PHASE_VOCAB / IMPORTANCE_PRIOR
  - phase_transition_detect()
  - chunk_importance()
  - event_aware_split()

All action-memory helpers are pure functions and are only invoked when
the corresponding YAML flags (``action_memory.eafr_enabled`` etc.) are
True. When the flags are False, existing code paths remain untouched.
"""

import math
import zlib
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ── label → 2-D direction vectors ──────────────────────────────────

KEYBOARD_TO_VEC = {
    "None": (0.0, 0.0),
    "W":    (1.0, 0.0),
    "S":    (-1.0, 0.0),
    "D":    (0.0, 1.0),
    "A":    (0.0, -1.0),
    "W+D":  (1.0, 1.0),
    "W+A":  (1.0, -1.0),
    "S+D":  (-1.0, 1.0),
    "S+A":  (-1.0, -1.0),
}

MOUSE_TO_VEC = {
    "·":   (0.0, 0.0),
    "→":   (1.0, 0.0),
    "←":   (-1.0, 0.0),
    "↑":   (0.0, 1.0),
    "↓":   (0.0, -1.0),
    "↑→":  (1.0, 1.0),
    "↑←":  (-1.0, 1.0),
    "↓→":  (1.0, -1.0),
    "↓←":  (-1.0, -1.0),
}

KEYBOARD_LABELS = ["None", "W", "S", "D", "A", "W+D", "W+A", "S+D", "S+A"]
MOUSE_LABELS = ["·", "→", "←", "↑", "↓", "↑→", "↑←", "↓→", "↓←"]

_KEYBOARD_TO_IDX = {k: i for i, k in enumerate(KEYBOARD_LABELS)}
_MOUSE_TO_IDX = {m: i for i, m in enumerate(MOUSE_LABELS)}

NUM_ACTION_CLASSES = len(KEYBOARD_LABELS) * len(MOUSE_LABELS)  # 81


def _normalize_pair(x: float, y: float) -> Tuple[float, float]:
    """Normalize diagonal directions to unit length."""
    n = math.sqrt(x * x + y * y)
    if n > 1.0 + 1e-6:
        return x / n, y / n
    return x, y


def action_labels_to_id(keys: str, mouse: str) -> int:
    """Convert action labels to a single integer id in [0, 80]."""
    k_idx = _KEYBOARD_TO_IDX.get(keys, 0)
    m_idx = _MOUSE_TO_IDX.get(mouse, 0)
    return k_idx * len(MOUSE_LABELS) + m_idx


def action_labels_to_vec(keys: str, mouse: str) -> Tuple[float, float, float, float]:
    """Convert action labels to a 4-D direction vector [forward, right, yaw, pitch].

    Diagonal directions are normalized to unit length so that composite
    actions do not have larger magnitude than single-axis actions.
    """
    fwd, right = KEYBOARD_TO_VEC.get(keys, (0.0, 0.0))
    fwd, right = _normalize_pair(fwd, right)
    yaw, pitch = MOUSE_TO_VEC.get(mouse, (0.0, 0.0))
    yaw, pitch = _normalize_pair(yaw, pitch)
    return (fwd, right, yaw, pitch)


def batch_action_to_tensors(
    action_keys_list: list,
    action_mouse_list: list,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of action labels to (action_ids, action_vecs).

    Returns:
        action_ids:  (B,) LongTensor
        action_vecs: (B, 4) FloatTensor  [forward, right, yaw, pitch]
    """
    ids = []
    vecs = []
    for keys, mouse in zip(action_keys_list, action_mouse_list):
        ids.append(action_labels_to_id(keys, mouse))
        vecs.append(action_labels_to_vec(keys, mouse))
    action_ids = torch.tensor(ids, dtype=torch.long, device=device)
    action_vecs = torch.tensor(vecs, dtype=dtype, device=device)
    return action_ids, action_vecs


# ── Fourier / sinusoidal embedding for low-dimensional scalars ──────

class FourierFeatures(nn.Module):
    """Sinusoidal positional encoding for each scalar dimension independently.

    Given input (B, D), produces (B, D * (1 + 2*num_bands)):
      [x, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x), ...]
    """

    def __init__(self, in_dim: int = 4, num_bands: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        freqs = torch.arange(num_bands, dtype=torch.float32)
        freqs = (math.pi * (2.0 ** freqs))  # pi, 2pi, 4pi, ...
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = in_dim * (1 + 2 * num_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) -> (B, D * (1 + 2*num_bands))"""
        # x_expanded: (B, D, 1) * (num_bands,) -> (B, D, num_bands)
        angles = x.unsqueeze(-1) * self.freqs.to(x.device, x.dtype)
        sin_feats = torch.sin(angles)  # (B, D, num_bands)
        cos_feats = torch.cos(angles)  # (B, D, num_bands)
        out = torch.cat([x.unsqueeze(-1), sin_feats, cos_feats], dim=-1)  # (B, D, 1+2K)
        return out.flatten(-2)  # (B, D*(1+2K))


class ActionDirectionEmbedder(nn.Module):
    """Hybrid action embedder: structured direction Fourier + 81-class identity.

    This module produces a vector that can be added to the diffusion timestep
    embedding (temb) inside the AdaLN modulation pathway.

    Architecture:
        direction branch: action_vec (4D) -> FourierFeatures -> MLP -> (inner_dim,)
        identity branch:  action_id (81)  -> Embedding(256) -> MLP -> (inner_dim,)
        output = zero_init_proj(direction + identity)  -> (inner_dim,)
    """

    def __init__(
        self,
        inner_dim: int = 1536,
        num_action_classes: int = NUM_ACTION_CLASSES,
        action_vec_dim: int = 4,
        fourier_bands: int = 6,
        embed_dim: int = 256,
        hidden_dim: int = 256,  # 1280 chosen to match Infinite-World's MLP width for 256-dim embeddings (5x expansion), then projected to 5120 (4x) for temb addition
    ):
        super().__init__()

        # Direction branch
        self.fourier = FourierFeatures(in_dim=action_vec_dim, num_bands=fourier_bands)
        fourier_out = self.fourier.out_dim  # 4*(1+2*6) = 52
        self.dir_mlp = nn.Sequential(
            nn.Linear(fourier_out, hidden_dim),   # 52 → 1280
            nn.SiLU(),
            nn.Linear(hidden_dim, inner_dim),     # 1280 → 5120 (4x)
        )

        # Identity branch — bottleneck: embed small, project up
        self.id_embed = nn.Embedding(num_action_classes, embed_dim)
        self.id_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),     # 256 → 1280 (5x)
            nn.SiLU(),
            nn.Linear(hidden_dim, inner_dim),     # 1280 → 5120 (4x)
        )

        # Output projection — default PyTorch init (matches Infinite-World)
        self.out_proj = nn.Linear(inner_dim, inner_dim)

    def forward(
        self,
        action_id: torch.Tensor,   # (B,) LongTensor
        action_vec: torch.Tensor,   # (B, 4) FloatTensor
    ) -> torch.Tensor:
        """Returns (B, inner_dim) to be added to temb."""
        dir_feat = self.fourier(action_vec)         # (B, 52)
        dir_feat = self.dir_mlp(dir_feat)           # (B, inner_dim)
        id_feat = self.id_embed(action_id)          # (B, embed_dim)
        id_feat = self.id_proj(id_feat)             # (B, inner_dim)
        return self.out_proj(dir_feat + id_feat)    # (B, inner_dim)


class ActionBlockAdapter(nn.Module):
    """Per-block action adapter.

    Two structural variants:

    * ``simple=True`` (recommended, Astra-style, B3):
        Each block owns its own ``nn.Embedding(num_classes, dim)`` table,
        zero-initialised.  Input is the raw ``action_id`` integer tensor
        (B,) and the forward is a pure lookup -> (B, 1, dim).  No shared
        intermediate, no linear layers, no activation.

        Per-block params: ``num_classes * dim``.
        e.g. num_classes=81, dim=5120 -> ~415 K / block  (40 blocks ~= 17 M).

    * ``simple=False`` (legacy):
        Input is a shared float feature at ``dim`` precomputed by the
        old ``ActionPatchEmbedder``.  A two-layer bottleneck MLP
        ``Linear(dim, bottleneck) -> SiLU -> Linear(bottleneck, dim)``
        compresses then re-expands.  Preserved only for backward
        compatibility with old checkpoints.
        Per-block params: ``2 * dim * bottleneck + dim + bottleneck``.

    Rationale for the B3 simple branch: 81 discrete classes carry at
    most ~7 bits of information, so the intrinsic rank is bounded by 81.
    A single ``nn.Embedding(81, dim)`` per block fully saturates the
    81-d subspace of the 5120-d hidden space while letting every layer
    learn its own 81 -> dim mapping with minimal parameters.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int = 256,
        simple: bool = False,
        feature_dim: int = 256,  # kept for config compat, unused in simple
        num_action_classes: int = NUM_ACTION_CLASSES,
    ):
        super().__init__()
        self.simple = simple

        if simple:
            # Per-block embedding table, zero-init -> step-0 output = 0
            self.proj = nn.Embedding(num_action_classes, dim)
            nn.init.zeros_(self.proj.weight)
        else:
            # Legacy bottleneck MLP (input is at `dim` float)
            self.proj = nn.Sequential(
                nn.Linear(dim, bottleneck),
                nn.SiLU(),
                nn.Linear(bottleneck, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """simple=True:  x is action_id (B,) long  -> (B, 1, dim)
        simple=False: x is action_emb (B, *, dim) float -> (B, *, dim)."""
        if self.simple:
            return self.proj(x).unsqueeze(1)  # (B, dim) -> (B, 1, dim)
        return self.proj(x)


class ActionPatchEmbedder(nn.Module):
    """Scheme B: patch-level additive action conditioning.

    Produces a (B, inner_dim, 1, 1, 1) tensor that is broadcast-added to
    patchified latent features (B, inner_dim, T, H, W) before flattening
    into transformer tokens.

    Two structural variants, selected via ``simple``:

    * ``simple=False`` (legacy):
        Two-branch design -- discrete id (optional MLP bottleneck) +
        Fourier-encoded continuous direction -- summed and projected
        through an ``out_proj``.  Heavy (~30M params at inner_dim=5120).
    * ``simple=True`` (recommended, Astra-style, B3):
        Single ``nn.Embedding(num_action_classes, inner_dim)``,
        zero-initialised.  Owns its own (81, 5120) table for the Path A
        broadcast-add.  No intermediate feature, no linear layers, no
        activation.  Each of the 40 per-block ``ActionBlockAdapter``
        instances has its own (81, 5120) table, fully independent.

        Shared params: ``num_classes * inner_dim`` (81 * 5120 ~= 415 K).
    """

    def __init__(
        self,
        inner_dim: int = 5120,
        num_action_classes: int = NUM_ACTION_CLASSES,
        action_vec_dim: int = 4,
        fourier_bands: int = 6,
        hidden_dim: int = 256,
        use_dir: bool = True,
        id_embed_use_mlp: bool = True,
        simple: bool = False,
        feature_dim: int = 256,  # kept for config compat, unused in simple
    ):
        super().__init__()
        self.simple = simple
        self.use_dir = use_dir and not simple
        self.id_embed_use_mlp = id_embed_use_mlp and not simple

        if simple:
            # B3: direct 81 -> inner_dim embedding, zero-init so Path A's
            # add at step-0 is identity.
            self.id_embed = nn.Embedding(num_action_classes, inner_dim)
            nn.init.zeros_(self.id_embed.weight)
            return

        # Direction branch (continuous Fourier encoding of action_vec)
        if self.use_dir:
            self.fourier = FourierFeatures(in_dim=action_vec_dim, num_bands=fourier_bands)
            fourier_out = self.fourier.out_dim  # 4*(1+2*6) = 52
            self.dir_mlp = nn.Sequential(
                nn.Linear(fourier_out, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, inner_dim),
            )

        # Identity branch
        if self.id_embed_use_mlp:
            # Bottleneck: small embed → MLP → inner_dim
            embed_dim = hidden_dim  # 256, matching Infinite-World
            self.id_embed = nn.Embedding(num_action_classes, embed_dim)
            self.id_proj = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim * 2),  # 256 → 512
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, inner_dim),  # 512 → 5120
            )
        else:
            # Legacy: direct embed at inner_dim (matches old 8-card checkpoints)
            self.id_embed = nn.Embedding(num_action_classes, inner_dim)

        # Output projection — default PyTorch init (matches Infinite-World)
        self.out_proj = nn.Linear(inner_dim, inner_dim)

    def forward(
        self,
        action_id: torch.Tensor,   # (B,) LongTensor
        action_vec: torch.Tensor,   # (B, 4) FloatTensor  (unused in simple mode)
    ) -> torch.Tensor:
        """Returns (B, inner_dim, 1, 1, 1) for broadcast add to patch features."""
        if self.simple:
            out = self.id_embed(action_id)                    # (B, inner_dim)
            return out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        if self.id_embed_use_mlp:
            id_feat = self.id_proj(self.id_embed(action_id))  # (B,) → (B, 256) → (B, inner_dim)
        else:
            id_feat = self.id_embed(action_id)                # (B,) → (B, inner_dim) directly
        if self.use_dir:
            dir_feat = self.fourier(action_vec)     # (B, 52)
            dir_feat = self.dir_mlp(dir_feat)       # (B, inner_dim)
            out = self.out_proj(dir_feat + id_feat)
        else:
            out = self.out_proj(id_feat)
        return out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, inner_dim, 1, 1, 1)


# =====================================================================
# Dual-Memory Helios helpers (Phase 0 / 1 / 3)
# =====================================================================
#
# These utilities are plugged into the Dual-Memory extension defined in
# docs/plan_action_memory.md. They operate on per-chunk annotations that
# are already baked into seadance2 .pt files by
# tools/image_caption/encode_chunk_prompts.py:
#   - chunk_actions[i] = {"keys": str, "mouse": str}
#   - chunk_has_interaction[i] in {True, False, None}
#   - chunk_phase[i]           in PHASE_VOCAB or ""
#   - action_class             video-level str (or "None")
#
# Everything here is a pure function (no nn.Module state) so that it can
# be called from dataloader / pipeline / transformer paths without
# affecting existing baselines.

class ActionTaxonomy(str, Enum):
    """Coarse categorisation of per-chunk actions."""

    INTERACTION = "interaction"   # pickup, place, attach, toggle, ...
    NAVIGATION = "navigation"     # WASD keyboard moves (no interaction)
    CAMERA_ONLY = "camera_only"   # mouse/view changes only
    NONE = "none"                 # no action, padding, unknown


# Phase strings that appear in chunk_phase. The non-human seadance2
# subset adds "contact"; the *_chunked encoder variant additionally uses
# "reaching" (semantically close to "approaching"). We list every phase
# string that appears in the actual .pt files so EAFR importance lookup
# never falls back to the generic "unknown_phase" bucket.
PHASE_VOCAB: Tuple[str, ...] = (
    "approaching",
    "reaching",
    "contact",
    "manipulating",
    "completing",
    "post_action",
)

_PHASE_TO_IDX = {p: i for i, p in enumerate(PHASE_VOCAB)}


# Static importance prior used by Phase 1 EAFR-static. The absolute
# numbers do not matter; only their relative ordering does. We keep the
# peak on `contact`/`manipulating` because those are the causal frames
# where the world state actually changes.
IMPORTANCE_PRIOR = {
    "no_interaction": 0.10,
    "approaching":    0.50,
    "reaching":       0.50,   # seadance2 *_chunked synonym for approaching
    "contact":        1.00,
    "manipulating":   1.00,
    "completing":     0.80,
    "post_action":    0.40,
    "unknown_phase":  0.20,
}


# Mouse / keyboard actions that correspond to "interaction-likely"
# button events in the game dataset (where semantic labels are absent).
_INTERACTION_BUTTON_KEYS = {"E", "F", "R", "Q", "TAB", "SPACE", "ENTER"}
_INTERACTION_MOUSE = {"MOUSE_LEFT", "MOUSE_RIGHT", "MOUSE_MIDDLE"}


def action_taxonomy(
    keys: Optional[str],
    mouse: Optional[str],
    has_interaction: Optional[bool] = None,
) -> ActionTaxonomy:
    """Classify a (keys, mouse, has_interaction) triple.

    Rules:
      1. If seadance2 labelled the chunk as ``has_interaction=True`` we
         trust the annotation (semantic interaction).
      2. Otherwise, if keys/mouse reference a known interaction button
         we tentatively mark as ``INTERACTION`` (useful for game data).
      3. Otherwise, if keys/mouse reference WASD / mouse movement we
         mark as ``NAVIGATION`` or ``CAMERA_ONLY``.
      4. Else ``NONE``.
    """
    if has_interaction is True:
        return ActionTaxonomy.INTERACTION

    keys_str = keys or ""
    mouse_str = mouse or ""

    keys_upper = keys_str.upper()
    parts = [p for p in keys_upper.replace(" ", "").split("+") if p]
    has_interaction_button = any(p in _INTERACTION_BUTTON_KEYS for p in parts)
    has_mouse_button = any(p in _INTERACTION_MOUSE for p in parts) or any(
        m in _INTERACTION_MOUSE for m in mouse_str.upper().split()
    )

    if has_interaction_button or has_mouse_button:
        return ActionTaxonomy.INTERACTION

    wasd = {p for p in parts if p in {"W", "A", "S", "D"}}
    if wasd:
        return ActionTaxonomy.NAVIGATION

    if mouse_str and mouse_str != "\u00b7":  # middle dot (idle)
        return ActionTaxonomy.CAMERA_ONLY

    return ActionTaxonomy.NONE


def action_class_to_idx(action_class, num_classes: int) -> int:
    """Map a video-level action class to a stable index in [0, num_classes).

    Index 0 is reserved for "unknown" (``None`` / ``""`` / ``"None"``).
    Named classes are bucketed into [1, num_classes - 1] with CRC32, which
    is identical across processes. Python's built-in ``hash()`` must not be
    used here: it is salted per interpreter, so DDP ranks, resumed runs and
    inference would each see a different class -> row assignment.
    """
    if isinstance(action_class, int):
        return max(0, min(num_classes - 1, action_class))
    if action_class is None or action_class in ("", "None"):
        return 0
    return 1 + (zlib.crc32(str(action_class).encode("utf-8")) % max(1, num_classes - 1))


def phase_to_idx(phase: Optional[str]) -> int:
    """Map a phase string to a stable index in [0, |PHASE_VOCAB|]."""
    if phase is None or phase == "":
        return len(PHASE_VOCAB)  # reserved: unknown / padding
    return _PHASE_TO_IDX.get(phase, len(PHASE_VOCAB))


def phase_importance(
    has_interaction: Optional[bool],
    phase: Optional[str],
) -> float:
    """Static scalar importance derived from phase metadata."""
    if not has_interaction:
        return IMPORTANCE_PRIOR["no_interaction"]
    if phase in IMPORTANCE_PRIOR:
        return IMPORTANCE_PRIOR[phase]
    return IMPORTANCE_PRIOR["unknown_phase"]


def chunk_importance(
    has_interaction: Optional[bool],
    phase: Optional[str],
    recency: float,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> float:
    """Importance score for a history chunk.

    ``recency`` is a scalar in (0, 1] where 1.0 means "most recent"; use
    ``math.exp(-(now - t) / tau)`` at the call site.
    """
    return alpha * phase_importance(has_interaction, phase) + beta * float(recency)


class PhaseTransition(str, Enum):
    """Phase transitions that trigger an Event Memory write."""

    NONE = "none"
    ENTER_MANIP = "enter_manip"      # approaching/contact -> manipulating
    ENTER_COMPLETE = "enter_complete"  # manipulating -> completing
    RELEASE = "release"              # completing -> post_action


_ENTER_MANIP_FROM = {"approaching", "contact", ""}
_ENTER_COMPLETE_FROM = {"manipulating", "contact"}
_RELEASE_FROM = {"completing", "manipulating", "contact"}


def phase_transition_detect(
    prev_phase: Optional[str],
    cur_phase: Optional[str],
) -> PhaseTransition:
    """Detect an interaction-phase transition between two adjacent chunks."""
    prev = prev_phase or ""
    cur = cur_phase or ""

    if cur == "manipulating" and prev in _ENTER_MANIP_FROM and cur != prev:
        return PhaseTransition.ENTER_MANIP
    if cur == "completing" and prev in _ENTER_COMPLETE_FROM and cur != prev:
        return PhaseTransition.ENTER_COMPLETE
    if cur == "post_action" and prev in _RELEASE_FROM:
        return PhaseTransition.RELEASE
    return PhaseTransition.NONE


# ---------------------------------------------------------------------
# EAFR: Event-Aware Frame Re-Assignment
# ---------------------------------------------------------------------

def event_aware_split(
    num_history_frames: int,
    history_sizes: Sequence[int],
    frame_to_chunk: Sequence[int],
    chunk_has_interaction: Sequence[Optional[bool]],
    chunk_phase: Sequence[Optional[str]],
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 8.0,
) -> Tuple[List[int], List[int], List[int]]:
    """Return indices that re-assign history frames across long/mid/short.

    EAFR (Phase 1, static version). Given a flat history window of
    ``num_history_frames`` latent frames and the desired bucket sizes
    ``history_sizes = [T_long, T_mid, T_short]``, rank every frame by
    ``chunk_importance`` and return three index lists::

        long_idx   : lowest-importance frames  -> heavy compression
        mid_idx    : mid-importance frames
        short_idx  : highest-importance frames -> light compression

    The original temporal positions are preserved in ``frame_to_chunk``
    so that RoPE can continue to use real timestamps.

    Parameters
    ----------
    num_history_frames
        Number of latent frames in the history window, equal to
        ``sum(history_sizes)``.
    history_sizes
        Bucket sizes ordered as ``[T_long, T_mid, T_short]`` (same order
        as ``prepare_stage1_clean_input_from_latents`` expects).
    frame_to_chunk
        For each of the ``num_history_frames`` frames, the index of the
        annotated chunk it belongs to (``-1`` for zero-padded frames).
    chunk_has_interaction, chunk_phase
        Per-chunk labels, length ``>= max(frame_to_chunk)+1``.
    alpha, beta, tau
        Weights for phase importance, recency, and recency time-constant.

    Returns
    -------
    (long_idx, mid_idx, short_idx)
        Lists of frame indices into the original history window, each of
        length matching ``history_sizes``.
    """
    assert sum(history_sizes) == num_history_frames, (
        f"history_sizes sum {sum(history_sizes)} != num_history_frames {num_history_frames}"
    )
    T_long, T_mid, T_short = history_sizes[0], history_sizes[1], history_sizes[2]

    now = num_history_frames - 1
    importance: List[Tuple[float, int]] = []
    for i in range(num_history_frames):
        c = frame_to_chunk[i]
        if c is None or c < 0:
            # Zero-padded / missing: keep at the tail (lowest importance).
            imp = 0.0
        else:
            hi = chunk_has_interaction[c] if c < len(chunk_has_interaction) else None
            ph = chunk_phase[c] if c < len(chunk_phase) else None
            recency = math.exp(-(now - i) / max(tau, 1e-6))
            imp = chunk_importance(hi, ph, recency, alpha=alpha, beta=beta)
        importance.append((imp, i))

    # Sort by importance descending; ties broken by recency (larger i first).
    importance.sort(key=lambda x: (-x[0], -x[1]))

    short_idx = sorted(i for _, i in importance[:T_short])
    mid_idx = sorted(i for _, i in importance[T_short : T_short + T_mid])
    long_idx = sorted(i for _, i in importance[T_short + T_mid :])
    assert len(long_idx) == T_long
    return long_idx, mid_idx, short_idx


# ---------------------------------------------------------------------
# EAFR-dedup: redundancy-penalised greedy bucket fill (Phase 1b)
# ---------------------------------------------------------------------

def compute_latent_similarity_matrix(history_latents: torch.Tensor) -> torch.Tensor:
    """VAE-latent cosine similarity matrix.

    Parameters
    ----------
    history_latents
        ``(C, T_hist, H, W)`` VAE-latent slice. Layout MUST be channels-first;
        we ``movedim(1, 0)`` so the row-major reshape doesn't interleave
        channel-with-time dims (a real bug we hit during smoke-testing).

    Returns
    -------
    sim_mat
        ``(T_hist, T_hist)`` fp32, diagonal zeroed out.

    Cost: ~1ms for T_hist=19 on GPU; pure linear algebra, no encoder forward.
    Captures appearance + local structure but is **biased by VAE compression
    artefacts** — different cameras looking at the same texture can score
    artificially close. Use ``compute_clip_similarity_matrix`` for a stronger,
    semantics-aware variant.
    """
    T = history_latents.shape[1]
    flat = history_latents.float().movedim(1, 0).reshape(T, -1).contiguous()
    flat = torch.nn.functional.normalize(flat, p=2, dim=-1, eps=1e-8)
    sim_mat = (flat @ flat.T).clamp(min=-1.0, max=1.0)
    sim_mat.fill_diagonal_(0.0)
    return sim_mat


# CLIP backend is loaded lazily, mirroring StoryMem extract_keyframes.py:
# we try open_clip → openai-clip → transformers in that order, so users can
# install whichever one they already have.
class _CLIPCtx:
    model = None
    preprocess_fn = None  # callable: (T, 3, H, W) float[0,1] -> (T, 3, 224, 224) normalised
    device = None
    dtype = torch.float32
    backend = None  # "open_clip" | "clip" | "transformers"


def _imagenet_clip_normalize(x: torch.Tensor) -> torch.Tensor:
    """Bilinear resize to 224 + CLIP-mean/std normalise. Matches StoryMem.

    Input: ``(N, 3, H, W)`` in [0, 1].
    Output: ``(N, 3, 224, 224)`` ready for CLIP.encode_image.
    """
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _get_clip_model(device: str = "cuda", use_half: bool = False):
    """Lazy CLIP loader. Tries multiple backends so users with any one work.

    Returns ``(model, preprocess_fn, device, dtype, backend)``. The model
    exposes a callable ``encode_image(x) -> (N, D)`` API regardless of backend.
    """
    if _CLIPCtx.model is not None:
        return _CLIPCtx.model, _CLIPCtx.preprocess_fn, _CLIPCtx.device, _CLIPCtx.dtype, _CLIPCtx.backend

    dtype = torch.float16 if (use_half and device != "cpu") else torch.float32

    # 1) Try open_clip (most modern, simplest deps).
    try:
        import open_clip  # type: ignore
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if dtype == torch.float16:
            model = model.half()

        class _Wrap:
            def __init__(self, m):
                self.m = m
            def encode_image(self, x):
                return self.m.encode_image(x)
        _CLIPCtx.model = _Wrap(model)
        _CLIPCtx.preprocess_fn = _imagenet_clip_normalize
        _CLIPCtx.device, _CLIPCtx.dtype, _CLIPCtx.backend = device, dtype, "open_clip"
        return _CLIPCtx.model, _CLIPCtx.preprocess_fn, _CLIPCtx.device, _CLIPCtx.dtype, _CLIPCtx.backend
    except ImportError:
        pass

    # 2) Try OpenAI's official clip (StoryMem's choice).
    try:
        import clip  # type: ignore
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if dtype == torch.float16:
            model = model.half()
        _CLIPCtx.model = model
        _CLIPCtx.preprocess_fn = _imagenet_clip_normalize
        _CLIPCtx.device, _CLIPCtx.dtype, _CLIPCtx.backend = device, dtype, "clip"
        return _CLIPCtx.model, _CLIPCtx.preprocess_fn, _CLIPCtx.device, _CLIPCtx.dtype, _CLIPCtx.backend
    except ImportError:
        pass

    # 3) Fallback: HuggingFace transformers CLIP.
    try:
        from transformers import CLIPModel  # type: ignore
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if dtype == torch.float16:
            model = model.half()

        class _HFWrap:
            def __init__(self, m):
                self.m = m
            def encode_image(self, x):
                return self.m.get_image_features(pixel_values=x)
        _CLIPCtx.model = _HFWrap(model)
        _CLIPCtx.preprocess_fn = _imagenet_clip_normalize
        _CLIPCtx.device, _CLIPCtx.dtype, _CLIPCtx.backend = device, dtype, "transformers"
        return _CLIPCtx.model, _CLIPCtx.preprocess_fn, _CLIPCtx.device, _CLIPCtx.dtype, _CLIPCtx.backend
    except ImportError as e:
        raise ImportError(
            "compute_clip_similarity_matrix requires one of: open_clip, clip "
            "(openai/CLIP), or transformers (with CLIP). Install with one of:\n"
            "  pip install open_clip_torch\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  pip install transformers"
        ) from e


# DINO backend (self-supervised, fine-grained dense features).
# CLIP/SigLIP collapse 19 frames of one scene to "all look similar"; DINO does
# not — it is trained without language alignment, so within-scene pose / local
# composition differences survive in the feature. This is exactly the signal
# that EAFR-dedup wants to penalise.
#
# Default backend is DINOv3-B/16 via timm (works around the
# torch.hub-segmentor → torchmetrics → transformers transitive conflict in
# this env). The timm preset includes weights from the official
# facebook/dinov3-vitb16-pretrain-lvd1689m HF repo.
#
# Fallback DINOv2-B/14 path retained as ``backend='dinov2'`` for ablation.
class _DINOCtx:
    model = None
    device = None
    dtype = torch.float32
    backend = None  # cache key


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """Bilinear resize to 224 + standard ImageNet mean/std (DINOv2 / DINOv3)."""
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _get_dino_model(
    device: str = "cuda",
    use_half: bool = False,
    backend: str = "dinov3",
):
    """Lazy DINO loader. Returns (model, device, dtype, backend_key).

    backend
        "dinov3" — DINOv3-B/16 via timm (vit_base_patch16_dinov3.lvd1689m).
                   Recommended; matches the architecture from
                   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
                   but uses timm's loader to avoid the torch.hub
                   segmentors → torchmetrics → transformers chain that breaks
                   in this env.
        "dinov2" — DINOv2-B/14 via torch.hub. Older, slightly weaker, but a
                   reasonable fallback if timm-DINOv3 weights cannot be reached.
    """
    cache_key = backend
    if _DINOCtx.model is not None and _DINOCtx.backend == cache_key:
        return _DINOCtx.model, _DINOCtx.device, _DINOCtx.dtype, _DINOCtx.backend

    dtype = torch.float16 if (use_half and device != "cpu") else torch.float32

    if backend == "dinov3":
        try:
            import timm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "compute_dino_similarity_matrix(backend='dinov3') needs timm.  "
                "Install with: pip install timm"
            ) from e
        # num_classes=0 returns a (B, embed_dim) feature directly — no head.
        model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
        model = model.to(device).eval()
    elif backend == "dinov2":
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", source="github"
        ).to(device).eval()
    else:
        raise ValueError(f"unknown DINO backend: {backend!r}")

    for p in model.parameters():
        p.requires_grad_(False)
    if dtype == torch.float16:
        model = model.half()
    _DINOCtx.model = model
    _DINOCtx.device, _DINOCtx.dtype, _DINOCtx.backend = device, dtype, cache_key
    return model, device, dtype, cache_key


@torch.no_grad()
def compute_dino_similarity_matrix(
    pixel_frames: torch.Tensor,
    device: str = "cuda",
    use_half: bool = False,
    backend: str = "dinov3",
) -> torch.Tensor:
    """DINOv3 (default) or DINOv2 cosine similarity matrix on the [CLS] feature.

    DINO's self-supervised objective produces features that are sensitive to
    pose, local composition, and small geometric changes — exactly the
    signal you want for *intra-scene* dedup. CLIP/SigLIP collapse 19 frames
    of one scene to near-uniform similarity; DINO usually does not.

    Same API/cost as the CLIP variants. Run via the batch tool's
    ``--metrics latent dinov3 clip`` to compare.

    Parameters
    ----------
    pixel_frames
        ``(T, 3, H, W)`` float in [0, 1]. ImageNet-normalised internally.
    backend
        "dinov3" (default) or "dinov2"; see :func:`_get_dino_model`.
    """
    assert pixel_frames.dim() == 4 and pixel_frames.shape[1] == 3, (
        f"expect (T, 3, H, W), got {tuple(pixel_frames.shape)}"
    )
    model, dev, dt, _ = _get_dino_model(device=device, use_half=use_half, backend=backend)
    x = pixel_frames.to(dev).clamp(0.0, 1.0)
    x = _imagenet_normalize(x).to(dt)
    # timm vit (num_classes=0) and torch.hub DINOv2 both return (T, embed_dim).
    z = model(x)
    if isinstance(z, dict):
        # DINOv2's full forward (when not pre-pruned) may return a dict.
        z = z.get("x_norm_clstoken", z.get("cls_token", next(iter(z.values()))))
    z = torch.nn.functional.normalize(z.float(), p=2, dim=-1, eps=1e-8)
    sim_mat = (z @ z.T).clamp(min=-1.0, max=1.0)
    sim_mat.fill_diagonal_(0.0)
    return sim_mat


# SigLIP backend (Google, generally stronger than CLIP at recognition tasks).
# Loaded lazily via open_clip; uses (0.5, 0.5, 0.5) normalisation and 224
# bilinear resize for ViT-B-16-SigLIP. Webli pretrained weights.
class _SigLIPCtx:
    model = None
    device = None
    dtype = torch.float32
    backend = None


def _siglip_normalize(x: torch.Tensor) -> torch.Tensor:
    """Bilinear resize to 224 + SigLIP-style (0.5, 0.5, 0.5) normalise.

    Input: ``(N, 3, H, W)`` in [0, 1].  Output: ``(N, 3, 224, 224)``.
    """
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _get_siglip_model(device: str = "cuda", use_half: bool = False):
    """Lazy SigLIP loader via open_clip. Returns (model, device, dtype, backend).

    open_clip's ViT-B-16-SigLIP + webli is the closest analogue to ViT-B/32 CLIP
    in size; mean ~150M params, ~5ms/frame on A100. Image-tower only — no text
    encoder needed for our cosine-similarity use.
    """
    if _SigLIPCtx.model is not None:
        return _SigLIPCtx.model, _SigLIPCtx.device, _SigLIPCtx.dtype, _SigLIPCtx.backend

    dtype = torch.float16 if (use_half and device != "cpu") else torch.float32
    try:
        import open_clip  # type: ignore
    except ImportError as e:
        raise ImportError(
            "compute_siglip_similarity_matrix requires open_clip_torch. "
            "Install with: pip install open_clip_torch"
        ) from e
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if dtype == torch.float16:
        model = model.half()

    class _Wrap:
        def __init__(self, m):
            self.m = m
        def encode_image(self, x):
            return self.m.encode_image(x)

    _SigLIPCtx.model = _Wrap(model)
    _SigLIPCtx.device, _SigLIPCtx.dtype, _SigLIPCtx.backend = device, dtype, "open_clip-siglip"
    return _SigLIPCtx.model, device, dtype, "open_clip-siglip"


@torch.no_grad()
def compute_siglip_similarity_matrix(
    pixel_frames: torch.Tensor,
    device: str = "cuda",
    use_half: bool = False,
) -> torch.Tensor:
    """SigLIP ViT-B-16 (webli) cosine similarity matrix.

    SigLIP uses sigmoid loss instead of softmax-contrastive (CLIP), generally
    learns more discriminative image features. Same API/cost as
    :func:`compute_clip_similarity_matrix`; pick whichever metric your offline
    validator (``tools/eafr_dedup_compare_batch.py``) shows produces more
    consistent picks for your data.

    Parameters
    ----------
    pixel_frames
        ``(T, 3, H, W)`` float in [0, 1].
    """
    assert pixel_frames.dim() == 4 and pixel_frames.shape[1] == 3, (
        f"expect (T, 3, H, W), got {tuple(pixel_frames.shape)}"
    )
    model, dev, dt, _ = _get_siglip_model(device=device, use_half=use_half)
    x = pixel_frames.to(dev).clamp(0.0, 1.0)
    x = _siglip_normalize(x).to(dt)
    z = model.encode_image(x)
    z = torch.nn.functional.normalize(z.float(), p=2, dim=-1, eps=1e-8)
    sim_mat = (z @ z.T).clamp(min=-1.0, max=1.0)
    sim_mat.fill_diagonal_(0.0)
    return sim_mat


@torch.no_grad()
def compute_clip_similarity_matrix(
    pixel_frames: torch.Tensor,
    device: str = "cuda",
    use_half: bool = False,
) -> torch.Tensor:
    """CLIP ViT-B/32 cosine similarity matrix between frames.

    Mirrors StoryMem ``extract_keyframes.py:get_frame_sim_clip``: 224 bilinear
    resize, CLIP-mean/std normalise, encode_image, L2 normalise, dot product.
    Captures global semantics (camera-invariant up to scene change) so it's
    less fooled by VAE artefacts than the latent-space cosine.

    Parameters
    ----------
    pixel_frames
        ``(T, 3, H, W)`` float in [0, 1]. Decode VAE latents first if needed.
    device
        CUDA device for the CLIP forward.
    use_half
        Cast CLIP weights to fp16 for ~2x speedup; usually fine for sim matrices.

    Returns
    -------
    sim_mat
        ``(T, T)`` fp32, diagonal zeroed out.

    Cost: ~5ms/frame on A100 with ViT-B/32. T_hist=19 → ~100ms total. **Much
    heavier than latent cosine (<2ms) — only worth it if user has confirmed
    CLIP picks differ meaningfully via the comparison tool.**
    """
    assert pixel_frames.dim() == 4 and pixel_frames.shape[1] == 3, (
        f"expect (T, 3, H, W), got {tuple(pixel_frames.shape)}"
    )
    model, preprocess_fn, dev, dt, _ = _get_clip_model(device=device, use_half=use_half)
    x = pixel_frames.to(dev).clamp(0.0, 1.0)
    x = preprocess_fn(x).to(dt)
    z = model.encode_image(x)               # (T, D)
    z = torch.nn.functional.normalize(z.float(), p=2, dim=-1, eps=1e-8)
    sim_mat = (z @ z.T).clamp(min=-1.0, max=1.0)
    sim_mat.fill_diagonal_(0.0)
    return sim_mat


def event_aware_split_with_dedup(
    history_latents: Optional[torch.Tensor] = None,
    history_sizes: Sequence[int] = (16, 2, 1),
    frame_to_chunk: Sequence[int] = (),
    chunk_has_interaction: Sequence[Optional[bool]] = (),
    chunk_phase: Sequence[Optional[str]] = (),
    dedup_lambda: float = 0.3,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 8.0,
    sim_mat: Optional[torch.Tensor] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Greedy EAFR variant that penalises redundancy inside each bucket.

    For every bucket (short, then mid; long is the leftover), pick frames
    one at a time by maximising::

        effective_score(i) = base_importance(i)
                           - dedup_lambda * max_{j in selected_so_far} sim(i, j)

    where ``selected_so_far`` is the set of frames already routed to short
    or mid (long never enters the dedup pool: low-importance frames are
    expected to look alike).

    Two ways to provide the similarity signal:

    * Pass ``history_latents`` (the cheap default) — we compute latent-space
      cosine internally via :func:`compute_latent_similarity_matrix`.
    * Pass ``sim_mat`` directly (precomputed CLIP / DINO / arbitrary metric)
      — we skip the latent path entirely. ``history_latents`` becomes optional.

    When neither is available (or ``dedup_lambda <= 0``), falls back to the
    static :func:`event_aware_split` ranking; safe for callers gating with a
    single boolean.

    Parameters
    ----------
    history_latents
        ``(C, T_hist, H, W)`` VAE-latent slice. Optional if ``sim_mat`` is given.
    sim_mat
        ``(T_hist, T_hist)`` precomputed similarity matrix (e.g. from
        :func:`compute_clip_similarity_matrix`). Takes precedence over
        ``history_latents`` if both are passed.
    dedup_lambda
        Strength of the redundancy penalty. 0.0 reproduces
        :func:`event_aware_split`; 0.3 is the recommended starting point.

    Returns
    -------
    (long_idx, mid_idx, short_idx)
    """
    num_history_frames = int(sum(history_sizes))
    if history_latents is not None:
        assert history_latents.shape[1] == num_history_frames, (
            f"history_latents temporal dim {history_latents.shape[1]} != sum(history_sizes) {num_history_frames}"
        )
    if sim_mat is not None:
        assert sim_mat.shape == (num_history_frames, num_history_frames), (
            f"sim_mat shape {tuple(sim_mat.shape)} != ({num_history_frames}, {num_history_frames})"
        )
    T_long, T_mid, T_short = history_sizes[0], history_sizes[1], history_sizes[2]

    # 1. Base importance scores (same recipe as event_aware_split).
    now = num_history_frames - 1
    base_scores: List[float] = []
    for i in range(num_history_frames):
        c = frame_to_chunk[i]
        if c is None or c < 0:
            base_scores.append(0.0)  # padding: lowest importance
            continue
        hi = chunk_has_interaction[c] if c < len(chunk_has_interaction) else None
        ph = chunk_phase[c] if c < len(chunk_phase) else None
        recency = math.exp(-(now - i) / max(tau, 1e-6))
        base_scores.append(chunk_importance(hi, ph, recency, alpha=alpha, beta=beta))

    # 2. Resolve similarity matrix (precomputed > latent cosine > none).
    if dedup_lambda <= 0.0 or (sim_mat is None and history_latents is None):
        # No usable signal — degrade to the static ranking variant.
        ranked = sorted(range(num_history_frames), key=lambda i: (-base_scores[i], -i))
        short_idx = sorted(ranked[:T_short])
        mid_idx = sorted(ranked[T_short : T_short + T_mid])
        long_idx = sorted(ranked[T_short + T_mid :])
        return long_idx, mid_idx, short_idx

    if sim_mat is None:
        sim_mat = compute_latent_similarity_matrix(history_latents)

    # 3. Greedy fill: short first (highest priority), then mid.
    base_t = torch.tensor(base_scores, device=sim_mat.device, dtype=sim_mat.dtype)
    available_mask = torch.ones(num_history_frames, dtype=torch.bool, device=sim_mat.device)
    selected_high_priority: List[int] = []  # accumulates short ∪ mid for dedup pool

    def _greedy_pick(bucket_size: int) -> List[int]:
        picked: List[int] = []
        for _ in range(bucket_size):
            if not available_mask.any():
                break
            if selected_high_priority:
                # max similarity from each candidate to any already-picked frame
                sel_idx = torch.tensor(selected_high_priority, device=sim_mat.device, dtype=torch.long)
                # sim_to_sel: (T_hist, len(sel_idx))
                sim_to_sel = sim_mat.index_select(1, sel_idx)
                max_sim = sim_to_sel.max(dim=-1).values  # (T_hist,)
            else:
                max_sim = torch.zeros(num_history_frames, device=sim_mat.device, dtype=sim_mat.dtype)
            effective = base_t - float(dedup_lambda) * max_sim
            # Mask out already-selected frames with -inf so they can't be picked again.
            effective = torch.where(available_mask, effective, torch.full_like(effective, float("-inf")))
            # Tie-break by recency (larger index wins) by adding a tiny epsilon.
            tie_break = torch.arange(num_history_frames, device=sim_mat.device, dtype=sim_mat.dtype) * 1e-6
            best = int(torch.argmax(effective + tie_break).item())
            picked.append(best)
            selected_high_priority.append(best)
            available_mask[best] = False
        return picked

    short_picks = _greedy_pick(T_short)
    mid_picks = _greedy_pick(T_mid)
    long_picks = [i for i in range(num_history_frames) if available_mask[i].item()]

    short_idx = sorted(short_picks)
    mid_idx = sorted(mid_picks)
    long_idx = sorted(long_picks)
    assert len(long_idx) == T_long, (
        f"dedup leftover {len(long_idx)} != T_long {T_long}"
    )
    return long_idx, mid_idx, short_idx


def frame_to_chunk_map(
    choice_idx: int,
    num_history_frames: int,
    latent_window_size: int,
) -> List[int]:
    """Build a frame -> annotated-chunk index map for the history window.

    ``choice_idx`` is the current chunk index (see ``prepare_stage1_latent``).
    The history window covers the ``num_history_frames`` latent frames
    immediately preceding the current chunk. Each chunk spans exactly
    ``latent_window_size`` latent frames in the .pt file.

    Frames that fall before the video start (zero padding) get ``-1``.
    """
    out: List[int] = []
    start_frame = choice_idx * latent_window_size - num_history_frames
    for i in range(num_history_frames):
        f = start_frame + i
        if f < 0:
            out.append(-1)
        else:
            out.append(f // latent_window_size)
    return out


# ---------------------------------------------------------------------
# Gap 4: EAFR-learnable (soft weighting) --- Phase 2a
# ---------------------------------------------------------------------

class EAFRLearnableScorer(nn.Module):
    """Zero-init learnable bias + affinity used by ``event_aware_split_soft``.

    The scorer produces a per-frame scalar that is **added** to the
    static prior computed in :func:`chunk_importance`. Its last linear
    layer is zero-initialised so that step-0 behaviour is identical to
    the EAFR-static variant.

    Parameters
    ----------
    num_action_classes
        Size of the action-class vocabulary (chunk-level ``action_class``).
    bias_init
        Starting value for the learnable phase-bias table (zero by default).
    """

    def __init__(
        self,
        num_action_classes: int = 64,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_action_classes = num_action_classes

        # Per-phase bias (matches PHASE_VOCAB + 1 unknown slot).
        self.phase_bias = nn.Parameter(torch.zeros(len(PHASE_VOCAB) + 1))

        # Action-class affinity table. For a candidate history chunk
        # labelled action_class_hist, affinity[a_cur, a_hist] is added.
        self.action_affinity = nn.Parameter(
            torch.zeros(num_action_classes, num_action_classes)
        )

        # Tiny gating MLP on top of (phase_idx, interaction_flag,
        # log-recency). Zero-init last layer to preserve baseline.
        self.gate = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        phase_idx: torch.Tensor,            # (B, T_hist) long
        interaction_flag: torch.Tensor,     # (B, T_hist) float in {0, 1}
        recency: torch.Tensor,              # (B, T_hist) float in (0, 1]
        current_action_class_idx: torch.Tensor,  # (B,) long
        history_action_class_idx: torch.Tensor,  # (B, T_hist) long
    ) -> torch.Tensor:
        """Return per-frame delta to the static importance prior.

        Output shape: ``(B, T_hist)``.
        """
        # 1. Phase bias lookup.
        phase_delta = self.phase_bias[phase_idx.clamp(min=0, max=self.phase_bias.shape[0] - 1)]

        # 2. Affinity lookup: affinity[current, history].
        cur = current_action_class_idx.clamp(min=0, max=self.num_action_classes - 1)
        his = history_action_class_idx.clamp(min=0, max=self.num_action_classes - 1)
        aff = self.action_affinity[cur.unsqueeze(1).expand_as(his), his]

        # 3. Gate: (B, T_hist, 3) -> (B, T_hist, 1) -> (B, T_hist).
        log_recency = torch.log(recency.clamp_min(1e-6))
        gate_input = torch.stack(
            [phase_idx.float(), interaction_flag.float(), log_recency], dim=-1
        )
        gate_out = self.gate(gate_input.to(self.phase_bias.dtype)).squeeze(-1)

        return phase_delta + aff + gate_out


def event_aware_split_soft(
    num_history_frames: int,
    history_sizes: Sequence[int],
    frame_to_chunk: Sequence[int],
    chunk_has_interaction: Sequence[Optional[bool]],
    chunk_phase: Sequence[Optional[str]],
    chunk_action_class_idx: Sequence[int],
    current_action_class_idx: int,
    scorer: Optional[EAFRLearnableScorer] = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 8.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[int], List[int], List[int], torch.Tensor]:
    """Soft EAFR variant: hard assignment + per-frame gradient-bearing weight.

    Returns
    -------
    (long_idx, mid_idx, short_idx, per_frame_weight)
        Same three index lists as :func:`event_aware_split`, plus a
        tensor of shape ``(num_history_frames,)`` containing a ``sigmoid``
        of the importance score (normalised to match the mean of the
        baseline scores). The caller multiplies the corresponding
        patchified tokens by this weight so gradients reach the scorer.
    """
    assert sum(history_sizes) == num_history_frames, (
        f"history_sizes sum {sum(history_sizes)} != num_history_frames {num_history_frames}"
    )
    import math as _math

    now = num_history_frames - 1
    base_scores = torch.zeros(num_history_frames, dtype=dtype, device=device)
    phase_idx_t = torch.zeros(num_history_frames, dtype=torch.long, device=device)
    inter_flag_t = torch.zeros(num_history_frames, dtype=dtype, device=device)
    recency_t = torch.zeros(num_history_frames, dtype=dtype, device=device)
    hist_ac_idx_t = torch.zeros(num_history_frames, dtype=torch.long, device=device)
    for i in range(num_history_frames):
        c = frame_to_chunk[i]
        if c is None or c < 0:
            continue
        hi = chunk_has_interaction[c] if c < len(chunk_has_interaction) else None
        ph = chunk_phase[c] if c < len(chunk_phase) else None
        ac = chunk_action_class_idx[c] if c < len(chunk_action_class_idx) else 0

        recency = _math.exp(-(now - i) / max(tau, 1e-6))
        recency_t[i] = recency
        base_scores[i] = alpha * phase_importance(hi, ph) + beta * recency
        phase_idx_t[i] = phase_to_idx(ph)
        inter_flag_t[i] = 1.0 if hi else 0.0
        hist_ac_idx_t[i] = int(ac)

    if scorer is not None:
        # Scorer expects batched shapes; make a batch of size 1.
        delta = scorer(
            phase_idx=phase_idx_t.unsqueeze(0),
            interaction_flag=inter_flag_t.unsqueeze(0),
            recency=recency_t.unsqueeze(0),
            current_action_class_idx=torch.tensor(
                [int(current_action_class_idx)], dtype=torch.long, device=device
            ),
            history_action_class_idx=hist_ac_idx_t.unsqueeze(0),
        ).squeeze(0)
        scores = base_scores + delta
    else:
        scores = base_scores

    # Hard assignment uses detached scores; soft weight uses scores+grad.
    ranked = scores.detach().argsort(descending=True).tolist()
    T_long, T_mid, T_short = history_sizes[0], history_sizes[1], history_sizes[2]
    short_idx = sorted(ranked[:T_short])
    mid_idx = sorted(ranked[T_short : T_short + T_mid])
    long_idx = sorted(ranked[T_short + T_mid :])

    # Normalise to a mean-1 weight so the soft signal is gradient-only
    # and does not rescale feature magnitudes on average.
    weight = torch.sigmoid(scores)
    weight = weight * (weight.numel() / weight.sum().clamp_min(1e-6))

    return long_idx, mid_idx, short_idx, weight
