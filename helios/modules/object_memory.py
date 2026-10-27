"""
Object Memory Bank for Persistent Memory Bank (branch idea1, OIM).

Motivation
----------
Helios's Multi-Term Memory (MTM) compresses history by temporal distance.
EAFR / ACHA / Event Memory all index history by *when* something happened.
Loop-closure scenarios — pickup → walk away → walk back → check that the
apple is still the same apple — therefore degrade as soon as the relevant
chunks fall outside the latent history window.

OIM adds a second writer to the persistent-token channel that Event
Memory introduced. Where Event Memory writes "X just happened" tokens
keyed by phase transition, OIM writes "this is what object X looks like"
tokens keyed by frozen DINOv3 saliency on the chunk's keyframes. Both
writers feed the same FIFO+pin queue and share the segment-embed slot;
in the paper they collapse to one Persistent Memory Bank subsection.

Design summary (full plan: docs/plan_object_memory.md)
------------------------------------------------------

Writer (rule-based + DINO saliency, v1):
    For each interaction-bearing chunk (chunk_has_interaction=True OR
    chunk_phase ∈ {completing, post_action}), pick the top-K patches by
    DINOv3 feature norm and emit one anchor token per patch. Tokens
    carry:
      * AttnPool of the DINO patch feature (the visual signature)
      * learnable phase embedding
      * learnable action_class embedding
      * chunk index (relative to current chunk)
      * shared segment embedding (zero-init)

Reader: identical to EventMemoryBank — concat to the front of the DiT
sequence, segment embed tags origin, RoPE filled at chunk-idx (Tier-A)
or camera_pose (Tier-A+, optional).

Baseline preservation
---------------------
* Module only runs when ``enabled=True``. With ``enabled=False`` no
  parameters appear in the state_dict.
* Visual / phase / action embeds use small dimensions; the final
  ``out_proj`` is zero-initialised so a freshly enabled bank emits the
  zero vector and is bit-for-bit equivalent to disabled at step 0.
* The saliency / writer-gate logic is pure Python; pre-baked DINO
  features come from the dataloader, online DINO is invoked only at
  inference.

Implementation seam: two writer paths
-------------------------------------

* ``select_anchors_from_cached(...)`` — training path. The dataloader
  surfaces a pre-baked ``(B, num_keyframes, num_topk, dino_dim)`` tensor
  of DINO patch features plus a ``(B, num_keyframes, num_topk)``
  saliency-score tensor; the bank selects per-chunk top-K_per_chunk
  patches and produces tokens.

* ``select_anchors_from_pixels(...)`` — inference path. Frozen DINOv3
  is invoked on VAE-decoded pixel keyframes, and the same selection +
  token construction follows. The DINO model is loaded lazily via
  ``helios.utils.action_utils._get_dino_model`` (already used by
  EAFR-dedup).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from helios.utils.action_utils import (
    PHASE_VOCAB,
    phase_to_idx,
)


# Same default action-class table size as EventMemoryBank for cheap
# checkpoint compatibility — both writers reuse the same embedding table
# layout if a later refactor unifies them into one nn.Module.
DEFAULT_NUM_ACTION_CLASSES: int = 64

# Default DINOv3-B/16 feature dim. Loader instantiates
# vit_base_patch16_dinov3.lvd1689m which exposes 768-dim CLS / patch
# tokens.
DEFAULT_DINO_FEAT_DIM: int = 768

# Phases that get *pinned* into the bank (i.e. exempt from FIFO eviction)
# until the bank's video-boundary reset. Chosen so that the just-touched
# object's signature persists through the navigation chunks that follow.
DEFAULT_PIN_PHASES: Tuple[str, ...] = ("contact", "manipulating", "completing")

# Phases that are eligible for write even when chunk_has_interaction
# is False (e.g. the chunk where the object was just released).
DEFAULT_AUX_WRITE_PHASES: Tuple[str, ...] = ("completing", "post_action")

# Phase vocab size + 1 reserved "unknown / pad" slot, matching
# action_utils.phase_to_idx() and EventMemoryBank.NUM_PHASE_SLOTS.
NUM_PHASE_SLOTS: int = len(PHASE_VOCAB) + 1


@dataclass
class ObjectAnchorRecord:
    """One anchor produced by the writer.

    Created in user-space (training: dataloader; inference: pipeline)
    and consumed by ``ObjectMemoryBank.forward``.
    """

    batch_idx: int
    chunk_idx: int                       # original chunk index (relative to video)
    rel_chunk_offset: int                # offset to the *current* chunk; 0 = current
    phase: int                           # phase_to_idx(chunk_phase[chunk_idx])
    action_class: int                    # video-level action class id
    pin: bool                            # if True, exempt from FIFO eviction
    visual_feature: torch.Tensor         # (dino_feat_dim,) DINO patch feature
    saliency: float                      # raw saliency score (for tie-breaking only)


# =====================================================================
# Selection helpers (writer side, pure functions over numpy/torch)
# =====================================================================


def topk_saliency_indices(
    dino_features: torch.Tensor,         # (T_kf, P, D)
    saliency: torch.Tensor,              # (T_kf, P)
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (indices, scores) for top-k patches across all keyframes.

    Flattens (T_kf, P) -> (T_kf*P,) before top-k so that one keyframe is
    allowed to contribute multiple patches if saliency favours it.
    """
    flat = saliency.reshape(-1)
    k_eff = min(k, flat.numel())
    if k_eff == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=saliency.device),
            torch.zeros((0,), dtype=saliency.dtype, device=saliency.device),
        )
    scores, idx = torch.topk(flat, k_eff, largest=True, sorted=True)
    return idx, scores


def gather_topk_features(
    dino_features: torch.Tensor,         # (T_kf, P, D)
    flat_indices: torch.Tensor,          # (k_eff,) into (T_kf*P)
) -> torch.Tensor:
    """Gather (k_eff, D) by flat indexing across keyframes×patches."""
    if flat_indices.numel() == 0:
        return dino_features.new_zeros((0, dino_features.shape[-1]))
    flat_feats = dino_features.reshape(-1, dino_features.shape[-1])
    return flat_feats.index_select(0, flat_indices)


# =====================================================================
# Bank
# =====================================================================


class ObjectMemoryBank(nn.Module):
    """Persistent anchor bank for entity-axis structured memory.

    Parameters
    ----------
    inner_dim
        Transformer inner dimension. Output tokens are at this width.
    dino_feat_dim
        Dimensionality of the upstream DINO patch features. Default 768
        for DINOv3-B/16.
    num_action_classes
        Action-class vocabulary size; reused for the embedding table.
    max_anchors
        Bank capacity. ``K_total`` in plan §4.0. Default 16.
    per_chunk
        Max anchors written per chunk by this writer. ``K_per_chunk`` in
        plan §4.0. Default 3.
    pin_phases
        Phases whose anchors are exempt from FIFO eviction.
    aux_write_phases
        Phases that may write even when ``chunk_has_interaction=False``.
    enabled
        Master switch; when False the module is a no-op and exposes
        zero parameters to ``state_dict``.
    online_dino_backend
        Backend name forwarded to ``_get_dino_model`` when running the
        inference path. Default ``"dinov3"``; ``"dinov2"`` available as
        fallback.
    use_pose_pe
        Tier-A+ option: add a per-anchor camera-pose PE on top of the
        chunk-index PE. Requires camera_poses to be threaded through.
    """

    def __init__(
        self,
        inner_dim: int,
        dino_feat_dim: int = DEFAULT_DINO_FEAT_DIM,
        num_action_classes: int = DEFAULT_NUM_ACTION_CLASSES,
        max_anchors: int = 16,
        per_chunk: int = 3,
        pin_phases: Sequence[str] = DEFAULT_PIN_PHASES,
        aux_write_phases: Sequence[str] = DEFAULT_AUX_WRITE_PHASES,
        enabled: bool = False,
        online_dino_backend: str = "dinov3",
        use_pose_pe: bool = False,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.dino_feat_dim = dino_feat_dim
        self.num_action_classes = num_action_classes
        self.max_anchors = int(max_anchors)
        self.per_chunk = int(per_chunk)
        self.pin_phases = set(pin_phases)
        self.aux_write_phases = set(aux_write_phases)
        self.enabled = bool(enabled)
        self.online_dino_backend = str(online_dino_backend)
        self.use_pose_pe = bool(use_pose_pe)

        # Discrete embeddings (small tables).
        self.phase_embed = nn.Embedding(NUM_PHASE_SLOTS, inner_dim)
        self.action_class_embed = nn.Embedding(num_action_classes, inner_dim)

        # Visual branch: project the DINO patch feature to inner_dim.
        # Two-layer MLP with zero-init last layer so the bank emits 0
        # before training updates it.
        self.visual_proj = nn.Sequential(
            nn.Linear(dino_feat_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )
        nn.init.zeros_(self.visual_proj[-1].weight)
        nn.init.zeros_(self.visual_proj[-1].bias)

        # Final composition projection — also zero-init, mirroring
        # EventMemoryBank.out_proj. Together with the visual_proj zero-
        # init this makes the bank a strict no-op at step 0.
        self.out_proj = nn.Linear(inner_dim, inner_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Segment embedding shared with Event Memory. The transformer
        # wire-in adds the same learnable segment_embed to both writers'
        # tokens so they live in the same prepended-token slot. We keep
        # an OWN parameter here for symmetry (and so the module is
        # self-contained); the wire-in code may override.
        self.segment_embed = nn.Parameter(torch.zeros(inner_dim))

        # 1-bit per-token type tag used by the unified Persistent
        # Memory Bank to let the DiT distinguish event-vs-object tokens.
        # Object tokens add this; event tokens leave it as zero.
        self.type_tag_embed = nn.Parameter(torch.zeros(inner_dim))

        # Lazy DINO model (online inference path). Materialised on first
        # call to select_anchors_from_pixels(). Kept as a non-Module
        # attribute so it does not appear in the bank's state_dict.
        self._dino_model = None
        self._dino_device: Optional[torch.device] = None
        self._dino_dtype: Optional[torch.dtype] = None

    # ------------------------------------------------------------------
    # Writer gate — the same gate the dataloader uses at training time.
    # ------------------------------------------------------------------
    def gate_write(
        self,
        chunk_phase: Optional[str],
        chunk_has_interaction: Optional[bool],
    ) -> bool:
        """Return True iff this chunk should contribute anchors.

        Mirrors plan §4 pseudocode:
          - has_interaction=True              -> write
          - phase ∈ aux_write_phases          -> write (release residue)
          - else                              -> skip
        """
        if bool(chunk_has_interaction):
            return True
        if chunk_phase is not None and chunk_phase in self.aux_write_phases:
            return True
        return False

    def is_pin_phase(self, chunk_phase: Optional[str]) -> bool:
        if chunk_phase is None:
            return False
        return chunk_phase in self.pin_phases

    # ------------------------------------------------------------------
    # Writer: cached DINO features (training-time path)
    # ------------------------------------------------------------------
    def select_anchors_from_cached(
        self,
        chunk_idx: int,
        rel_chunk_offset: int,
        chunk_phase: Optional[str],
        chunk_has_interaction: Optional[bool],
        action_class_id: int,
        dino_features: torch.Tensor,         # (T_kf, P, D)
        saliency: torch.Tensor,              # (T_kf, P)
        batch_idx: int = 0,
    ) -> List[ObjectAnchorRecord]:
        """Select up to ``per_chunk`` anchors for this chunk."""
        if not self.gate_write(chunk_phase, chunk_has_interaction):
            return []
        flat_idx, flat_scores = topk_saliency_indices(
            dino_features, saliency, self.per_chunk
        )
        feats = gather_topk_features(dino_features, flat_idx)  # (k_eff, D)
        pin = self.is_pin_phase(chunk_phase)
        records: List[ObjectAnchorRecord] = []
        for j in range(feats.shape[0]):
            records.append(
                ObjectAnchorRecord(
                    batch_idx=int(batch_idx),
                    chunk_idx=int(chunk_idx),
                    rel_chunk_offset=int(rel_chunk_offset),
                    phase=phase_to_idx(chunk_phase),
                    action_class=int(
                        max(0, min(self.num_action_classes - 1, int(action_class_id)))
                    ),
                    pin=bool(pin),
                    visual_feature=feats[j].detach(),
                    saliency=float(flat_scores[j].detach().cpu()),
                )
            )
        return records

    # ------------------------------------------------------------------
    # Writer: online DINO on pixel keyframes (inference-time path)
    # ------------------------------------------------------------------
    def _ensure_dino_model(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._dino_model is not None:
            return
        # Lazy import to avoid hard dependency at module load.
        from helios.utils.action_utils import _get_dino_model

        model, dev, dt, _ = _get_dino_model(
            backend=self.online_dino_backend,
            device=str(device),
            use_half=(dtype == torch.float16),
        )
        # Freeze.
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self._dino_model = model
        self._dino_device = dev if isinstance(dev, torch.device) else torch.device(dev)
        self._dino_dtype = dt

    @staticmethod
    def _dino_normalize_pixels(
        keyframes: torch.Tensor,                # (T_kf, 3, H, W) in [0,1]
        target_size: int = 224,
    ) -> torch.Tensor:
        """Bilinear resize + ImageNet mean/std normalisation."""
        if keyframes.shape[-2:] != (target_size, target_size):
            keyframes = F.interpolate(
                keyframes,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=keyframes.device, dtype=keyframes.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=keyframes.device, dtype=keyframes.dtype
        ).view(1, 3, 1, 1)
        return (keyframes - mean) / std

    @torch.no_grad()
    def select_anchors_from_pixels(
        self,
        chunk_idx: int,
        rel_chunk_offset: int,
        chunk_phase: Optional[str],
        chunk_has_interaction: Optional[bool],
        action_class_id: int,
        keyframes_pixel: torch.Tensor,         # (T_kf, 3, H, W) in [0,1]
        batch_idx: int = 0,
    ) -> List[ObjectAnchorRecord]:
        """Online path: run DINO on pixels, then reuse cached selector."""
        if not self.gate_write(chunk_phase, chunk_has_interaction):
            return []
        device = keyframes_pixel.device
        dtype = keyframes_pixel.dtype
        self._ensure_dino_model(device, dtype)
        x = self._dino_normalize_pixels(keyframes_pixel.to(self._dino_dtype))
        # _get_dino_model wraps timm so .forward returns the [CLS] head
        # features. We need patch tokens, so we use forward_features +
        # the canonical timm token-norm head.
        feats = self._dino_model.forward_features(x)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_patchtokens", feats.get("x", None))
        if feats is None or feats.ndim != 3:
            raise RuntimeError(
                "Unexpected DINO output shape for online ObjectMemoryBank"
            )
        # Drop prefix tokens (CLS + optional register tokens). DINOv3-B/16
        # has num_prefix_tokens = 5 (1 CLS + 4 register); DINOv2 has 1.
        n_prefix = getattr(self._dino_model, "num_prefix_tokens", 1)
        if n_prefix > 0:
            feats = feats[:, n_prefix:, :]
        # Saliency = L2 norm of each patch feature.
        saliency = feats.norm(dim=-1)              # (T_kf, P)
        return self.select_anchors_from_cached(
            chunk_idx=chunk_idx,
            rel_chunk_offset=rel_chunk_offset,
            chunk_phase=chunk_phase,
            chunk_has_interaction=chunk_has_interaction,
            action_class_id=action_class_id,
            dino_features=feats.to(self.visual_proj[0].weight.dtype),
            saliency=saliency.to(self.visual_proj[0].weight.dtype),
            batch_idx=batch_idx,
        )

    # ------------------------------------------------------------------
    # Bank state (inference-time persistence across chunks)
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int) -> dict:
        """Return a fresh per-batch persistent state. Caller is the pipeline.

        We deliberately keep the bank state OUT of nn.Module so that
        gradient flow / state_dict semantics stay clean. The pipeline
        passes this state in/out of admit() and tokens().
        """
        return {
            "anchors": [[] for _ in range(batch_size)],   # list of List[ObjectAnchorRecord]
        }

    def admit(
        self,
        state: dict,
        new_records: List[ObjectAnchorRecord],
    ) -> dict:
        """Add records to the bank with FIFO + pin eviction."""
        if not new_records:
            return state
        for r in new_records:
            slot = state["anchors"][r.batch_idx]
            slot.append(r)
            # Evict.
            if len(slot) > self.max_anchors:
                # Evict the oldest non-pinned anchor first; if all pinned,
                # fall back to oldest.
                evicted = False
                for i, a in enumerate(slot[:-1]):  # exclude the just-admitted one
                    if not a.pin:
                        slot.pop(i)
                        evicted = True
                        break
                if not evicted:
                    slot.pop(0)
        return state

    # ------------------------------------------------------------------
    # Forward: compose object tokens for the whole batch
    # ------------------------------------------------------------------
    def forward(
        self,
        batch_records: List[List[ObjectAnchorRecord]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compose (object_tokens, object_mask).

        Returns
        -------
        object_tokens : torch.Tensor | None
            ``(B, max_anchors, inner_dim)``. ``None`` when disabled.
        object_mask : torch.Tensor | None
            ``(B, max_anchors)`` bool, ``True`` for valid positions.
            ``None`` when disabled.
        """
        if not self.enabled:
            return None, None

        batch_size = len(batch_records)
        tokens = torch.zeros(
            (batch_size, self.max_anchors, self.inner_dim),
            dtype=dtype,
            device=device,
        )
        mask = torch.zeros(
            (batch_size, self.max_anchors),
            dtype=torch.bool,
            device=device,
        )

        for b, records in enumerate(batch_records):
            for k, r in enumerate(records[: self.max_anchors]):
                visual_vec = r.visual_feature.to(device=device, dtype=dtype)
                v = self.visual_proj(visual_vec)             # (inner_dim,)
                p = self.phase_embed(
                    torch.tensor(r.phase, device=device, dtype=torch.long)
                )
                a = self.action_class_embed(
                    torch.tensor(r.action_class, device=device, dtype=torch.long)
                )
                token = (
                    v
                    + p
                    + a
                    + self.segment_embed.to(dtype)
                    + self.type_tag_embed.to(dtype)
                )
                # out_proj zero-init -> token starts at 0; gradients
                # flow from the DiT through visual_proj and embeddings.
                token = self.out_proj(token)
                tokens[b, k] = token
                mask[b, k] = True

        return tokens, mask
