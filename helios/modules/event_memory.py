"""
Event Memory Bank for Dual-Memory Helios (branch idea1, Phase 3).

Motivation
----------
Helios's Multi-Term Memory (MTM) stores *visual* history via three
patchified Conv3d branches. For interaction-heavy generation this is
insufficient: even if a pickup chunk is perfectly preserved, the model
still has to infer "apple is in hand" from pixels on every subsequent
chunk. We therefore add a *structured* second memory that explicitly
records **interaction phase transitions** detected from the per-chunk
``chunk_phase`` annotations already produced by
``tools/image_caption/encode_chunk_prompts.py``.

Design (summary from ``docs/plan_action_memory.md``)
----------------------------------------------------
Writer (rule-based in v1):
    Inspect ``chunk_phase[t-1] -> chunk_phase[t]``. When a transition is
    one of ``ENTER_MANIP``, ``ENTER_COMPLETE`` or ``RELEASE``, emit an
    event token carrying:
        * learnable transition-type embedding
        * learnable phase embedding
        * learnable action_class embedding
        * attention-pooled visual summary of that chunk (from
          ``patch_short`` features, or from ``history_latents`` when
          ``patch_short`` is not yet computed)
        * temporal position (original chunk index)

Reader (Option B in plan):
    Concatenate up to ``max_events`` event tokens to the front of the
    DiT token sequence after long/mid/short history. The event tokens
    use a dedicated segment embedding so they do not compete with the
    spatial-temporal RoPE; they participate in self-attention only
    (cross-attn with text is untouched).

Baseline preservation
---------------------
* The module only runs when ``EventMemoryBank.enabled is True``.
* With ``enabled=False``, callers receive ``None`` and all downstream
  code skips the branch. No parameters are materialised into the graph.
* With ``enabled=True`` but all inputs missing (``chunk_phase=None`` or
  game-only data), the bank emits zero events and also becomes a no-op.
* All learnable parameters that multiply with concatenated features are
  **zero-initialised** at their output projections so that a fresh
  ``enabled=True`` run starts out equivalent to ``enabled=False`` and
  only diverges as training updates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from helios.utils.action_utils import (
    PHASE_VOCAB,
    PhaseTransition,
    phase_to_idx,
    phase_transition_detect,
)


# Maximum number of distinct "action_class" identifiers we expect.
# seadance2 has ~40 interaction classes; we reserve extra slots for
# future expansion and for the "unknown" bucket at index 0.
DEFAULT_NUM_ACTION_CLASSES: int = 64

# Transition vocab size (len(PhaseTransition) includes NONE).
_TRANSITION_VOCAB = [t.value for t in PhaseTransition]
_TRANSITION_TO_IDX = {t: i for i, t in enumerate(_TRANSITION_VOCAB)}
NUM_TRANSITION_TYPES = len(_TRANSITION_VOCAB)

# Phase vocab size including an "unknown / padding" slot at the tail
# (matches action_utils.phase_to_idx).
NUM_PHASE_SLOTS = len(PHASE_VOCAB) + 1


@dataclass
class EventRecord:
    """Lightweight record produced by :class:`EventMemoryBank.detect_events`.

    Batch element ``b``, written at chunk ``chunk_idx``.
    ``visual_summary_source`` is a tensor shaped ``(C, Hs, Ws)`` carved
    out of the corresponding history chunk's latents; it is pooled and
    projected by the module at forward time.
    """

    batch_idx: int
    chunk_idx: int                # original annotated chunk index
    transition: int               # index in _TRANSITION_VOCAB
    phase: int                    # phase_to_idx(chunk_phase[chunk_idx])
    action_class: int             # index of video-level action class
    visual_summary_source: Optional[torch.Tensor]  # (C, Hs, Ws) or None


class EventMemoryBank(nn.Module):
    """Interaction-phase-triggered structured memory.

    Parameters
    ----------
    inner_dim
        Transformer inner dimension (token width). Event tokens are
        produced at this dimensionality so they concatenate cleanly with
        the existing hidden_states.
    latent_channels
        Channel count of the VAE latent ("C" in (B, C, T, H, W)). Used to
        size the visual projection.
    num_action_classes
        Size of the action-class vocabulary (e.g. 40 for seadance2 + pad).
    max_events
        Upper bound on event tokens per sample; excess events are dropped
        (oldest first). Keep small (<= 32) to avoid inflating the DiT
        sequence length.
    enabled
        Master switch. When False all forward calls return ``None``.
    """

    def __init__(
        self,
        inner_dim: int,
        latent_channels: int = 16,
        num_action_classes: int = DEFAULT_NUM_ACTION_CLASSES,
        max_events: int = 16,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.latent_channels = latent_channels
        self.num_action_classes = num_action_classes
        self.max_events = max_events
        self.enabled = enabled

        # Discrete embeddings (small tables).
        self.transition_embed = nn.Embedding(NUM_TRANSITION_TYPES, inner_dim)
        self.phase_embed = nn.Embedding(NUM_PHASE_SLOTS, inner_dim)
        self.action_class_embed = nn.Embedding(num_action_classes, inner_dim)

        # Visual branch: global-avg-pool the chunk's latent then project.
        self.visual_proj = nn.Linear(latent_channels, inner_dim)

        # Output projection for the composed event token. Zero-init the
        # weight and bias so that the first update step is a no-op: the
        # event branch emits a zero vector which, when concatenated to
        # hidden_states, acts as a learnable identity residual.
        self.out_proj = nn.Linear(inner_dim, inner_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Segment embedding added to every event token to tag its origin.
        # Zero-init keeps the early behaviour baseline-equivalent.
        self.segment_embed = nn.Parameter(torch.zeros(inner_dim))

    # ------------------------------------------------------------------
    # Writer: rule-based phase-transition detector
    # ------------------------------------------------------------------
    @staticmethod
    def detect_events(
        chunk_phase_batch: Sequence[Sequence[Optional[str]]],
        action_class_ids: Sequence[int],
        current_chunk_idx: Sequence[int],
        history_latents: Optional[torch.Tensor] = None,
        latent_window_size: int = 9,
        history_window_frames: int = 19,
        max_events: int = 16,
    ) -> List[List[EventRecord]]:
        """Scan phase strings and return the subset that triggers an event.

        Parameters mirror what the transformer receives at
        ``forward(action_memory_context=...)`` time.

        Returns
        -------
        List[List[EventRecord]]
            Outer list length = batch size; inner list length <=
            ``max_events`` (most recent transitions first).
        """
        batch_events: List[List[EventRecord]] = []
        for b, phases in enumerate(chunk_phase_batch):
            events: List[EventRecord] = []
            if not phases:
                batch_events.append(events)
                continue

            now = current_chunk_idx[b]
            window_start_frame = now * latent_window_size - history_window_frames
            window_start_chunk = max(0, window_start_frame // latent_window_size)
            scan_lo = max(1, window_start_chunk)
            scan_hi = min(now + 1, len(phases))

            for c in range(scan_lo, scan_hi):
                prev = phases[c - 1] if 0 <= c - 1 < len(phases) else None
                cur = phases[c] if 0 <= c < len(phases) else None
                transition = phase_transition_detect(prev, cur)
                if transition == PhaseTransition.NONE:
                    continue

                # Carve the corresponding visual patch from history_latents
                # if it was provided. history_latents is (B, C, T, H, W).
                visual_src = None
                if history_latents is not None:
                    T = history_latents.shape[2]
                    # frame index inside the history window
                    f_end = c * latent_window_size - window_start_frame
                    f_start = f_end - latent_window_size
                    f_start = max(0, f_start)
                    f_end = min(T, f_end)
                    if f_end > f_start:
                        visual_src = history_latents[b, :, f_start:f_end].mean(dim=1)

                events.append(
                    EventRecord(
                        batch_idx=b,
                        chunk_idx=c,
                        transition=_TRANSITION_TO_IDX[transition.value],
                        phase=phase_to_idx(cur),
                        action_class=int(action_class_ids[b]),
                        visual_summary_source=visual_src,
                    )
                )
                if len(events) >= max_events:
                    break

            # Keep most recent transitions; drop older overflow.
            events = events[-max_events:]
            batch_events.append(events)
        return batch_events

    # ------------------------------------------------------------------
    # Forward: compose event tokens for the whole batch
    # ------------------------------------------------------------------
    def forward(
        self,
        batch_events: List[List[EventRecord]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compose (event_tokens, event_mask).

        Returns
        -------
        event_tokens : torch.Tensor | None
            Shape ``(B, max_events, inner_dim)``. ``None`` when the bank
            is disabled.
        event_mask : torch.Tensor | None
            Boolean tensor shaped ``(B, max_events)`` where ``True``
            marks a valid event position and ``False`` a pad. ``None``
            when the bank is disabled.
        """
        if not self.enabled:
            return None, None

        batch_size = len(batch_events)
        tokens = torch.zeros(
            (batch_size, self.max_events, self.inner_dim),
            dtype=dtype,
            device=device,
        )
        mask = torch.zeros(
            (batch_size, self.max_events),
            dtype=torch.bool,
            device=device,
        )

        for b, events in enumerate(batch_events):
            for k, ev in enumerate(events[: self.max_events]):
                t_emb = self.transition_embed(
                    torch.tensor(ev.transition, device=device, dtype=torch.long)
                )
                p_emb = self.phase_embed(
                    torch.tensor(ev.phase, device=device, dtype=torch.long)
                )
                ac_idx = max(0, min(self.num_action_classes - 1, int(ev.action_class)))
                a_emb = self.action_class_embed(
                    torch.tensor(ac_idx, device=device, dtype=torch.long)
                )

                token = t_emb + p_emb + a_emb + self.segment_embed

                if ev.visual_summary_source is not None:
                    visual = ev.visual_summary_source.to(device=device, dtype=dtype)
                    # visual is (C, H, W); mean pool spatial dims.
                    visual_vec = visual.mean(dim=(-2, -1))  # (C,)
                    token = token + self.visual_proj(visual_vec.to(token.dtype))

                # out_proj is zero-init -> initial event token is the
                # zero vector. Gradients flow as training progresses.
                token = self.out_proj(token)
                tokens[b, k] = token
                mask[b, k] = True

        return tokens, mask
