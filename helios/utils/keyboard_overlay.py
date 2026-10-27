"""WASD + arrow-key keyboard overlay shared by demo_helios.py / infer_helios.py.

Frames may arrive as PIL.Image, uint8 numpy (H, W, 3), OR float32 numpy in
[0, 1] depending on whether the upstream pipe used `output_type='pil'` or 'np'.
``_to_pil`` handles all three; PIL.Image.fromarray refuses float32, so we
clip-and-scale before constructing the PIL Image.

Why PIL (not cv2):
  1. Handles the Unicode mouse symbols (·, ↑, ↓, ←, → and diagonal combos)
     correctly — cv2's HERSHEY fonts are ASCII-only and would render them
     as "?".
  2. Output is a list of PIL.Image, which avoids a known
     diffusers.utils.export_to_video bug: when given a list of uint8 numpy
     arrays it does `(frame * 255).astype(np.uint8)`, overflowing uint8 →
     "blue dominant + magenta speckle" RGB drift. The PIL branch of
     export_to_video skips that multiplication, so colors stay correct.

Layout:
  bottom-left:    [W]            bottom-right:    [↑]
                [A][S][D]                       [←][↓][→]
  Active keys: yellow fill + black border + black glyph.
  Inactive:    dark-gray fill + light-gray border + light-gray glyph.
"""

from __future__ import annotations
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_pil_rgb(frame) -> Image.Image:
    """Normalize a frame (PIL / uint8 ndarray / float32 ndarray in [0,1]) to PIL RGB.

    PIL.Image.fromarray rejects float dtypes; the diffusers pipeline returns
    float32 in [0, 1] when output_type='np' (which happens implicitly in some
    multi-rank paths), so we clip + scale before constructing the PIL Image.
    """
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


# Colors are RGB (PIL operates on RGB images).
_KEY_BG_INACTIVE = (35, 35, 35)
_KEY_BORDER_INACTIVE = (190, 190, 190)
_KEY_TEXT_INACTIVE = (220, 220, 220)
_KEY_BG_ACTIVE = (255, 200, 50)        # bright yellow-orange highlight
_KEY_BORDER_ACTIVE = (0, 0, 0)
_KEY_TEXT_ACTIVE = (0, 0, 0)


def _resolve_unicode_font(size: int):
    """Returns a PIL font that handles ·, ↑, ↓, ←, →. Tries DejaVuSans-Bold
    (present on virtually every Linux distro), then DejaVuSans, then PIL's
    bitmap default (ASCII-only fallback).
    """
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _parse_active_keys(keys_str: str) -> set:
    out = set()
    if not keys_str or keys_str.lower() == "none":
        return out
    for tok in keys_str.split("+"):
        tok = tok.strip().upper()
        if tok in ("W", "A", "S", "D"):
            out.add(tok)
    return out


def _parse_active_arrows(mouse_str: str) -> set:
    out = set()
    if not mouse_str or mouse_str == "·":
        return out
    mapping = {"↑": "up", "↓": "down", "←": "left", "→": "right"}
    for ch in mouse_str:
        if ch in mapping:
            out.add(mapping[ch])
    return out


def _draw_letter_key(draw, x, y, size, label, active, font):
    if active:
        bg, bd, fg = _KEY_BG_ACTIVE, _KEY_BORDER_ACTIVE, _KEY_TEXT_ACTIVE
    else:
        bg, bd, fg = _KEY_BG_INACTIVE, _KEY_BORDER_INACTIVE, _KEY_TEXT_INACTIVE
    draw.rectangle([(x, y), (x + size, y + size)], fill=bg, outline=bd, width=2)
    try:
        l, top, r, b = draw.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - top
    except AttributeError:
        tw, th = draw.textsize(label, font=font)
    cx, cy = x + size // 2, y + size // 2
    draw.text((cx - tw // 2, cy - th // 2 - 2), label, fill=fg, font=font)


def _draw_arrow_key(draw, x, y, size, direction, active):
    if active:
        bg, bd, fg = _KEY_BG_ACTIVE, _KEY_BORDER_ACTIVE, _KEY_TEXT_ACTIVE
    else:
        bg, bd, fg = _KEY_BG_INACTIVE, _KEY_BORDER_INACTIVE, _KEY_TEXT_INACTIVE
    draw.rectangle([(x, y), (x + size, y + size)], fill=bg, outline=bd, width=2)
    cx, cy = x + size // 2, y + size // 2
    s = int(size * 0.28)
    if direction == "up":
        pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
    elif direction == "down":
        pts = [(cx, cy + s), (cx - s, cy - s), (cx + s, cy - s)]
    elif direction == "left":
        pts = [(cx - s, cy), (cx + s, cy - s), (cx + s, cy + s)]
    else:  # right
        pts = [(cx + s, cy), (cx - s, cy - s), (cx - s, cy + s)]
    draw.polygon(pts, fill=fg)


def _draw_action_button(draw, x, y, w, h, label, active, font):
    """A wide rectangular button with text — for action / verb buttons that
    sit alongside the WASD/arrow grids."""
    if active:
        bg, bd, fg = _KEY_BG_ACTIVE, _KEY_BORDER_ACTIVE, _KEY_TEXT_ACTIVE
    else:
        bg, bd, fg = _KEY_BG_INACTIVE, _KEY_BORDER_INACTIVE, _KEY_TEXT_INACTIVE
    draw.rectangle([(x, y), (x + w, y + h)], fill=bg, outline=bd, width=2)
    try:
        l, top, r, b = draw.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - top
    except AttributeError:
        tw, th = draw.textsize(label, font=font)
    cx, cy = x + w // 2, y + h // 2
    draw.text((cx - tw // 2, cy - th // 2 - 2), label, fill=fg, font=font)


def render_keyboard_overlay_pil(
    frames_pil: Sequence,
    keys_per_chunk: Sequence[str],
    mouse_per_chunk: Sequence[str],
    frames_per_chunk: int,
    *,
    key_size: int = 44,
    margin: int = 14,
    gap: int = 4,
    # ── Optional action overlay (matrix-game-3 style) ───────────────────────
    # When provided, ``actions_per_chunk[i]`` is the canonical action label
    # active in chunk ``i`` (e.g. "pickup"). Per chunk the label is matched
    # against ``action_button_labels`` and that single button is highlighted.
    # An empty string in the list means "no action this chunk" (e.g. pure
    # camera/movement chunk) → all action buttons stay dim.
    action_button_labels: Optional[Sequence[str]] = None,
    actions_per_chunk: Optional[Sequence[str]] = None,
    # When True, on chunks that *do* have an active action, force every
    # WASD/arrow button to render dim regardless of keys/mouse — the frame
    # only emphasises the verb. Conversely, on chunks with no action, the
    # action buttons are forced dim. Mirrors the Matrix-Game-3 single-mode
    # look. Default False = both layers light up independently.
    action_exclusive: bool = False,
) -> List[Image.Image]:
    """Burns WASD (bottom-left) + arrow keys (bottom-right) — and optionally
    a row of action / verb buttons (bottom-center) — into every frame.

    Returns a NEW list of PIL.Image. Pass directly to
    diffusers.utils.export_to_video.
    """
    if not frames_pil:
        return list(frames_pil)
    first = _to_pil_rgb(frames_pil[0])
    W, H = first.size
    s = key_size

    wasd_top = H - margin - 2 * s - gap
    wasd_left = margin
    wasd_origins = {
        "W": (wasd_left + s + gap,         wasd_top),
        "A": (wasd_left,                   wasd_top + s + gap),
        "S": (wasd_left + s + gap,         wasd_top + s + gap),
        "D": (wasd_left + 2 * (s + gap),   wasd_top + s + gap),
    }
    arr_left = W - margin - 3 * s - 2 * gap
    arr_origins = {
        "up":    (arr_left + s + gap,         wasd_top),
        "left":  (arr_left,                   wasd_top + s + gap),
        "down":  (arr_left + s + gap,         wasd_top + s + gap),
        "right": (arr_left + 2 * (s + gap),   wasd_top + s + gap),
    }

    letter_font = _resolve_unicode_font(max(20, int(s * 0.55)))
    action_font = _resolve_unicode_font(max(14, int(s * 0.42)))

    # Pre-compute action-button geometry once (sizes don't change per frame).
    action_layout: List[tuple] = []  # list of (x, y, w, h, label)
    if action_button_labels:
        labels = [str(lab) for lab in action_button_labels]
        # Width auto-fits the longest label; padded; one button per label;
        # row sits aligned with the bottom WASD row (same y as ASD / ←↓→).
        # Height = key_size; width = max(text_w + 2*pad, key_size*1.2).
        # Use a throwaway draw to measure text widths without an image yet.
        from PIL import Image as _Img, ImageDraw as _Draw
        _probe = _Draw.Draw(_Img.new("RGB", (10, 10)))
        pad_x = max(8, int(s * 0.25))
        widths: List[int] = []
        for lab in labels:
            try:
                l, t, r, b = _probe.textbbox((0, 0), lab, font=action_font)
                widths.append(r - l)
            except AttributeError:
                widths.append(_probe.textsize(lab, font=action_font)[0])
        btn_widths = [max(int(s * 1.2), tw + 2 * pad_x) for tw in widths]
        total_w = sum(btn_widths) + (len(labels) - 1) * gap
        # Place buttons centered horizontally; y aligned with the bottom
        # WASD row so they read as one keyboard row across the frame.
        action_y = wasd_top + s + gap
        action_x = (W - total_w) // 2
        x = action_x
        for lab, bw in zip(labels, btn_widths):
            action_layout.append((x, action_y, bw, s, lab))
            x += bw + gap

    out: List[Image.Image] = []
    n = max(len(keys_per_chunk), 1)
    for t, src in enumerate(frames_pil):
        chunk_idx = min(t // max(frames_per_chunk, 1), n - 1)
        active_letters = _parse_active_keys(keys_per_chunk[chunk_idx])
        active_arrows = _parse_active_arrows(mouse_per_chunk[chunk_idx])

        active_action_label: Optional[str] = None
        if actions_per_chunk is not None and chunk_idx < len(actions_per_chunk):
            cand = str(actions_per_chunk[chunk_idx]) if actions_per_chunk[chunk_idx] else ""
            if cand:
                active_action_label = cand

        # Exclusive mode: an active action suppresses keys/mouse, and absence
        # of an action keeps the verb row dim. Off by default.
        if action_exclusive:
            if active_action_label is not None:
                active_letters = set()
                active_arrows = set()
            else:
                # actions all stay dim because active_action_label is None
                pass

        img = _to_pil_rgb(src).copy()
        draw = ImageDraw.Draw(img)

        for letter, (x, y) in wasd_origins.items():
            _draw_letter_key(draw, x, y, s, letter,
                             letter in active_letters, letter_font)
        for d, (x, y) in arr_origins.items():
            _draw_arrow_key(draw, x, y, s, d, d in active_arrows)
        for x, y, w, h, lab in action_layout:
            _draw_action_button(
                draw, x, y, w, h, lab,
                active=(lab == active_action_label),
                font=action_font,
            )

        out.append(img)
    return out
