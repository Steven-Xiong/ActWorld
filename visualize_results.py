"""Generate a self-contained HTML visualization for Helios inference / validation results.

Supports three modes:

1. **Eval mode** (inference from CSV or TSV):
   Supports both formats:
   - CSV: columns id, prompt, image_name, action_class  (test_eval.csv)
   - TSV: columns action_class, prompt, video_name      (world_model_action12_*.tsv)
   File format is auto-detected from extension.

   python visualize_results.py --mode eval \
       --video_dir ./output_helios/eval_seadance2_20260317_190752 \
       --label_path data/seadance2_yume_test_12classv2/test_eval.csv \
       --first_frame_dir data/seadance2_yume_test_12classv2/first_frame

2. **Validation mode** (training checkpoint videos):
   Scans --video_dir for global_step{N}_*_video_{i}_{prompt}.mp4, grouped by step.

   python visualize_results.py --mode validation \
       --video_dir ./output_helios/stage3_action_20260318_211239

3. **Scan-all mode** (batch, entire output_helios/ tree):
   Auto-detects each subdir as eval or validation and generates html/results.html in each.

   python visualize_results.py --mode scan_all \
       --output_base_dir ./output_helios \
       --label_path data/seadance2_yume_test_12classv2/test_eval.csv \
       --first_frame_dir data/seadance2_yume_test_12classv2/first_frame

Videos and images are base64-encoded so the HTML is fully self-contained (works offline).
"""

import argparse
import base64
import json
import os
import re

import pandas as pd


def encode_file_b64(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


HTML_HEADER = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 24px; }}
    h1 {{ text-align: center; margin-bottom: 8px; font-size: 22px; color: #fff; }}
    .meta {{ text-align: center; margin-bottom: 24px; color: #888; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
    th {{ background: #1a1a1a; padding: 12px 16px; text-align: left; font-size: 13px; color: #aaa; border-bottom: 2px solid #333; position: sticky; top: 0; z-index: 1; }}
    tr {{ border-bottom: 1px solid #222; }}
    tr:hover {{ background: #1a1a1a; }}
    td {{ padding: 16px; vertical-align: top; }}
    .img-cell img {{ max-width: 260px; border-radius: 4px; }}
    .prompt-cell {{ max-width: 520px; }}
    .sample-id {{ font-weight: 700; font-size: 14px; color: #7cb3ff; margin-bottom: 6px; }}
    .action-class {{ display: inline-block; background: #2a3a50; color: #8ec8ff; font-size: 11px; padding: 2px 8px; border-radius: 3px; margin-bottom: 6px; }}
    .prompt {{ font-size: 13px; line-height: 1.5; color: #ccc; }}
    .chunk-actions {{ margin-top: 8px; font-size: 12px; line-height: 1.8; }}
    .chunk-actions summary {{ cursor: pointer; color: #8ec8ff; font-size: 12px; }}
    .chunk {{ display: inline-block; background: #1e2a36; border-radius: 3px; padding: 1px 6px; margin: 2px 2px; font-family: monospace; font-size: 11px; }}
    .chunk .key {{ color: #ffcc66; }}
    .chunk .mouse {{ color: #88ddaa; }}
    .chunk .clause {{ color: #999; font-style: italic; font-family: inherit; font-size: 10px; }}
    .chunk-prompts {{ margin-top: 8px; font-size: 12px; line-height: 1.5; }}
    .chunk-prompts summary {{ cursor: pointer; color: #d8a070; font-size: 12px; }}
    .chunk-prompt {{ background: #1a2218; border-left: 2px solid #5a8060; border-radius: 3px; padding: 4px 8px; margin: 3px 0; font-size: 11px; color: #c8d0c0; }}
    .chunk-prompt .cp-idx {{ color: #88ddaa; font-family: monospace; font-weight: 700; margin-right: 6px; }}
    .chunk-prompt .cp-text {{ color: #c0c8b8; }}
    .chunk-prompt.empty {{ opacity: 0.4; font-style: italic; }}
    .meta-tags {{ margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }}
    .meta-tag {{ display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 3px; }}
    .meta-tag.loc {{ background: #2a3a2a; color: #a0d8a0; }}
    .meta-tag.weather {{ background: #3a3a2a; color: #d8d8a0; }}
    .meta-tag.tod {{ background: #2a2a3a; color: #a0a0d8; }}
    .prompt-original {{ margin-top: 6px; font-size: 12px; line-height: 1.5; }}
    .prompt-original summary {{ cursor: pointer; color: #c8a070; font-size: 12px; }}
    .prompt-original .orig-text {{ color: #a08060; margin-top: 4px; }}
    .video-cell video {{ max-width: 480px; border-radius: 4px; }}
    .fname {{ font-size: 11px; color: #666; margin-top: 4px; }}
    .na {{ color: #555; }}
    .step-header {{ font-size: 18px; color: #7cb3ff; margin: 32px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #333; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 24px; margin-bottom: 40px; }}
    .card {{ background: #1a1a1a; border-radius: 8px; padding: 16px; }}
    .card video {{ width: 100%; border-radius: 4px; }}
    .card .prompt {{ margin-top: 8px; }}
</style>
</head>
<body>
"""

HTML_FOOTER = """\
</body>
</html>"""


# ─── Eval mode ───────────────────────────────────────────────────────────────

def _load_label_file(label_path: str) -> list[dict]:
    """Load CSV or TSV label file(s) and normalize to list of dicts with keys:
    sample_id, prompt, action_class, image_name (may be empty string).

    Supported formats (auto-detected):
      TSV:       action_class, prompt, video_name, [action_keys, action_mouse, ...]
      CSV:       id, prompt, image_name, [action_class, ...]
      Sekai CSV: videoFile, caption, [cameraFile, location, scene, ...]

    Multiple paths can be comma-separated (same as --image_prompt_csv_path in infer_helios.py).
    """
    paths = [p.strip() for p in label_path.split(",")]
    frames = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in {".tsv", ".tab"}:
            frames.append(pd.read_csv(p, sep="\t"))
        elif ext == ".csv":
            frames.append(pd.read_csv(p, sep=","))
        else:
            frames.append(pd.read_csv(p, sep=None, engine="python"))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Sekai CSV: videoFile → video_name, caption → prompt
    if "videoFile" in df.columns:
        df = df.rename(columns={"videoFile": "video_name", "caption": "prompt"})

    # TSV / Sekai: video_name → id / image_name
    if "video_name" in df.columns and "id" not in df.columns:
        stems = df["video_name"].astype(str).map(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        df = df.copy()
        df["id"] = stems
        if "image_name" not in df.columns:
            df["image_name"] = stems + ".jpg"

    if "image_name" not in df.columns:
        df["image_name"] = ""

    # Preserve prompt_original / caption_original if present (Sekai global CSVs)
    if "caption_original" in df.columns and "prompt_original" not in df.columns:
        df = df.rename(columns={"caption_original": "prompt_original"})

    records = []
    for _, row in df.iterrows():
        records.append({
            "sample_id": str(row["id"]),
            "prompt": str(row.get("refined_prompt") or row["prompt"]),
            "prompt_original": str(row["prompt_original"]) if "prompt_original" in df.columns and pd.notna(row.get("prompt_original")) else "",
            "action_class": str(row.get("action_class", row.get("scene", ""))),
            "image_name": str(row.get("image_name", "")),
            "location": str(row.get("location", "")),
            "weather": str(row.get("weather", "")),
            "timeOfDay": str(row.get("timeOfDay", "")),
        })
    return records


def _render_chunk_prompts_html(descs: list[str]) -> str:
    """Render per-chunk text prompts (chunk_descs) as a vertical list."""
    parts = []
    for i, desc in enumerate(descs):
        text = "" if desc is None else str(desc)
        if not text:
            parts.append(f'<div class="chunk-prompt empty"><span class="cp-idx">[{i}]</span><span class="cp-text">(empty)</span></div>')
            continue
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f'<div class="chunk-prompt"><span class="cp-idx">[{i}]</span>'
            f'<span class="cp-text">{escaped}</span></div>'
        )
    return "".join(parts)


def _render_chunk_actions_html(actions: list[list[str]]) -> str:
    """Render per-chunk [keys, mouse, clause_text?] as styled inline badges."""
    parts = []
    for i, entry in enumerate(actions):
        keys, mouse = entry[0], entry[1]
        clause = entry[2] if len(entry) > 2 and entry[2] else ""
        escaped = clause.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        title_attr = f' title="{escaped}"' if escaped else ""
        clause_span = f' <span class="clause">{escaped}</span>' if escaped else ""
        parts.append(
            f'<span class="chunk"{title_attr}>{i}: '
            f'<span class="key">{keys}</span> '
            f'<span class="mouse">{mouse}</span>'
            f'{clause_span}</span>'
        )
    return "".join(parts)


def build_eval_html(
    video_dir: str,
    label_path: str,
    first_frame_dir: str | None,
    output_path: str,
    chunk_actions_path: str | None = None,
):
    records = _load_label_file(label_path)

    chunk_actions_map: dict[str, list] = {}
    if chunk_actions_path and os.path.isfile(chunk_actions_path):
        with open(chunk_actions_path, "r") as f:
            chunk_actions_map = json.load(f)

    inference_meta: dict = {}
    meta_path = os.path.join(video_dir, "inference_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            inference_meta = json.load(f)

    meta_params = inference_meta.get("params", {})
    meta_samples = inference_meta.get("samples", {})

    rows_html = []
    for rec in records:
        sample_id = rec["sample_id"]
        sample_meta = meta_samples.get(sample_id, {})

        # Prefer prompt from inference_meta (what was actually fed to T5)
        prompt = sample_meta.get("prompt", rec["prompt"])
        if isinstance(prompt, list):
            prompt = " | ".join(prompt)
        prompt_original = sample_meta.get("prompt_original", rec.get("prompt_original", ""))
        action_source = sample_meta.get("action_source", "")
        action_class = rec["action_class"]
        image_name = rec["image_name"]
        location = rec.get("location", "")
        weather = rec.get("weather", "")
        time_of_day = rec.get("timeOfDay", "")

        video_path = os.path.join(video_dir, f"{sample_id}.mp4")
        if not os.path.isfile(video_path):
            continue

        vid_b64 = encode_file_b64(video_path, "video/mp4")

        img_b64 = ""
        if first_frame_dir and image_name:
            img_path = os.path.join(first_frame_dir, image_name)
            if not os.path.isfile(img_path):
                img_path = os.path.join(first_frame_dir, os.path.splitext(image_name)[0] + ".png")
            if os.path.isfile(img_path):
                mime = "image/png" if img_path.endswith(".png") else "image/jpeg"
                img_b64 = encode_file_b64(img_path, mime)

        action_badge = (
            f'<span class="action-class">{action_class}</span>'
            if action_class and action_class != "nan"
            else ""
        )
        escaped_prompt = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Metadata tags (location, weather, timeOfDay)
        meta_html = ""
        tag_parts = []
        if location and location != "nan":
            tag_parts.append(f'<span class="meta-tag loc">{location}</span>')
        if weather and weather != "nan":
            tag_parts.append(f'<span class="meta-tag weather">{weather}</span>')
        if time_of_day and time_of_day != "nan":
            tag_parts.append(f'<span class="meta-tag tod">{time_of_day}</span>')
        if tag_parts:
            meta_html = f'<div class="meta-tags">{"".join(tag_parts)}</div>'

        # Original prompt (before camera-movement stripping)
        orig_html = ""
        if prompt_original and prompt_original != "nan" and prompt_original != prompt:
            escaped_orig = prompt_original.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            orig_html = (
                f'<div class="prompt-original">'
                f'<details><summary>Original Caption (action source)</summary>'
                f'<div class="orig-text">{escaped_orig}</div>'
                f'</details></div>'
            )

        # Action source badge
        source_html = ""
        if action_source:
            source_html = f'<span class="meta-tag" style="background:#3a2a3a;color:#d8a0d8;">action: {action_source}</span>'

        # Chunk actions (prefer inference_meta over chunk_actions.json)
        ca = sample_meta.get("chunk_actions") or chunk_actions_map.get(sample_id)
        chunk_html = ""
        if ca:
            embeds_injected = meta_params.get("action_embeds_injected", True)
            if embeds_injected:
                ca_label = f"Chunk Actions ({len(ca)} chunks)"
                ca_label_style = ""
            else:
                ca_label = f"Chunk Actions ({len(ca)} chunks) — parsed only, not injected"
                ca_label_style = ' style="color:#a0a0a0;"'
            badges = _render_chunk_actions_html(ca)
            chunk_html = (
                f'<div class="chunk-actions">'
                f'<details><summary{ca_label_style}>{ca_label}</summary>'
                f'{badges}</details></div>'
            )

        # Chunk prompts (per-chunk text descriptions encoded into UMT5)
        cp = sample_meta.get("chunk_descs") or meta_params.get("chunk_descs")
        chunk_prompt_html = ""
        if cp:
            n_nonempty = sum(1 for x in cp if x)
            badges = _render_chunk_prompts_html(cp)
            chunk_prompt_html = (
                f'<div class="chunk-prompts">'
                f'<details><summary>Chunk Prompts ({n_nonempty}/{len(cp)} chunks)</summary>'
                f'{badges}</details></div>'
            )

        rows_html.append(f"""
        <tr>
            <td class="img-cell">
                {'<img src="' + img_b64 + '">' if img_b64 else '<span class="na">No image</span>'}
            </td>
            <td class="prompt-cell">
                <div class="sample-id">{sample_id}</div>
                {action_badge}
                {source_html}
                {meta_html}
                <div class="prompt">{escaped_prompt}</div>
                {orig_html}
                {chunk_html}
                {chunk_prompt_html}
            </td>
            <td class="video-cell">
                <video controls preload="metadata">
                    <source src="{vid_b64}" type="video/mp4">
                </video>
                <div class="fname">{sample_id}.mp4</div>
            </td>
        </tr>""")

    title = f"Helios Eval — {os.path.basename(video_dir)}"
    html = HTML_HEADER.format(title=title)
    html += f'<h1>{title}</h1>\n'
    html += f'<div class="meta">{len(rows_html)} samples &bull; {video_dir}</div>\n'

    if meta_params:
        param_parts = []
        key_order = [
            ("height", "H"), ("width", "W"), ("num_frames", "Frames"),
            ("num_inference_steps", "Steps"), ("guidance_scale", "CFG"),
            ("seed", "Seed"), ("num_latent_frames_per_chunk", "ChunkFrames"),
            ("is_enable_stage2", "Stage2"), ("use_zero_init", "ZeroInit"),
            ("zero_steps", "ZeroSteps"),
            ("image_noise_sigma_min", "ImgNoiseMin"), ("image_noise_sigma_max", "ImgNoiseMax"),
            ("use_interpolate_prompt", "InterpPrompt"),
        ]
        for key, label in key_order:
            if key in meta_params:
                param_parts.append(f'<span style="margin:0 6px;"><b>{label}:</b> {meta_params[key]}</span>')
        neg = meta_params.get("negative_prompt", "")
        if neg:
            escaped_neg = neg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            param_parts.append(f'<span style="margin:0 6px;"><b>NegPrompt:</b> <i>{escaped_neg}</i></span>')
        pyramid = meta_params.get("pyramid_num_inference_steps_list")
        if pyramid:
            param_parts.append(f'<span style="margin:0 6px;"><b>Pyramid:</b> {pyramid}</span>')
        if "action_embeds_injected" in meta_params:
            injected = meta_params["action_embeds_injected"]
            color = "#88ddaa" if injected else "#ff8888"
            label = "✓ ActionEmbed" if injected else "✗ ActionEmbed (ablation)"
            param_parts.append(f'<span style="margin:0 6px;color:{color};"><b>{label}</b></span>')
        html += (
            '<div style="text-align:center;margin:8px auto 16px;padding:10px 16px;'
            'background:#1a1a2a;border-radius:6px;max-width:1100px;font-size:12px;color:#aaa;'
            'line-height:2;">'
            '<b style="color:#7cb3ff;">Generation Params</b><br>'
            + "".join(param_parts)
            + '</div>\n'
        )

    html += '<table>\n<thead><tr><th>First Frame</th><th>Prompt &amp; Model Inputs</th><th>Generated Video</th></tr></thead>\n<tbody>\n'
    html += "".join(rows_html)
    html += "\n</tbody>\n</table>\n"
    html += HTML_FOOTER

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[visualize] HTML saved → {output_path} ({len(rows_html)} samples)")


# ─── Validation mode ─────────────────────────────────────────────────────────

_VAL_PAT = re.compile(
    r"^global_step(\d+)_((?:final_step_)?validation)_video_(\d+)_(.+)\.mp4$"
)


def build_validation_html(video_dir: str, output_path: str):
    entries = []
    for fname in os.listdir(video_dir):
        m = _VAL_PAT.match(fname)
        if not m:
            continue
        step, phase, vid_idx, prompt_slug = (
            int(m.group(1)),
            m.group(2),
            int(m.group(3)),
            m.group(4).replace("_", " "),
        )
        entries.append((step, phase, vid_idx, prompt_slug, fname))

    entries.sort(key=lambda e: (e[0], e[1], e[2]))

    body_parts = []
    current_step = None
    for step, phase, vid_idx, prompt_slug, fname in entries:
        if step != current_step:
            if current_step is not None:
                body_parts.append("</div>")  # close grid
            body_parts.append(f'<div class="step-header">Step {step}</div>')
            body_parts.append('<div class="grid">')
            current_step = step

        video_path = os.path.join(video_dir, fname)
        vid_b64 = encode_file_b64(video_path, "video/mp4")
        phase_label = phase.replace("_", " ").title()

        body_parts.append(f"""
        <div class="card">
            <video controls preload="metadata">
                <source src="{vid_b64}" type="video/mp4">
            </video>
            <div class="sample-id">#{vid_idx} — {phase_label}</div>
            <div class="prompt">{prompt_slug}</div>
            <div class="fname">{fname}</div>
        </div>""")

    if current_step is not None:
        body_parts.append("</div>")

    title = f"Helios Validation — {os.path.basename(video_dir)}"
    html = HTML_HEADER.format(title=title)
    html += f'<h1>{title}</h1>\n'
    html += f'<div class="meta">{len(entries)} videos &bull; {video_dir}</div>\n'
    html += "".join(body_parts)
    html += HTML_FOOTER

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[visualize] HTML saved → {output_path} ({len(entries)} videos)")


# ─── Scan-all mode ───────────────────────────────────────────────────────────

_VAL_PAT_QUICK = re.compile(r"^global_step\d+_")


def _is_validation_dir(d: str) -> bool:
    """Return True if directory contains training-validation mp4s."""
    try:
        return any(_VAL_PAT_QUICK.match(f) for f in os.listdir(d) if f.endswith(".mp4"))
    except OSError:
        return False


def _has_eval_mp4s(d: str) -> bool:
    """Return True if directory contains plain (non-validation) mp4 files."""
    try:
        return any(
            f.endswith(".mp4") and not _VAL_PAT_QUICK.match(f)
            for f in os.listdir(d)
        )
    except OSError:
        return False


def scan_all_and_build(
    output_base_dir: str,
    label_path: str | None,
    first_frame_dir: str | None,
    force: bool = False,
):
    """Scan every immediate subdirectory of output_base_dir and generate HTML."""
    try:
        subdirs = sorted(
            d for d in os.listdir(output_base_dir)
            if os.path.isdir(os.path.join(output_base_dir, d))
        )
    except OSError as e:
        print(f"[visualize] Cannot list {output_base_dir}: {e}")
        return

    total = 0
    skipped = 0
    for name in subdirs:
        d = os.path.join(output_base_dir, name)
        html_path = os.path.join(d, "html", "results.html")

        if os.path.isfile(html_path) and not force:
            print(f"[visualize] skip (already exists) → {html_path}")
            skipped += 1
            continue

        if _is_validation_dir(d):
            try:
                build_validation_html(d, html_path)
                total += 1
            except Exception as e:
                print(f"[visualize] FAILED {name}: {e}")
        elif _has_eval_mp4s(d):
            if label_path is None:
                print(f"[visualize] skip eval dir (no --label_path): {name}")
                continue
            ca_path = os.path.join(d, "chunk_actions.json")
            try:
                build_eval_html(d, label_path, first_frame_dir, html_path,
                                chunk_actions_path=ca_path if os.path.isfile(ca_path) else None)
                total += 1
            except Exception as e:
                print(f"[visualize] FAILED {name}: {e}")
        else:
            print(f"[visualize] skip (no mp4s found): {name}")

    print(f"\n[visualize] Done. Generated {total} HTML files, skipped {skipped}.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML visualization for Helios results")
    parser.add_argument(
        "--mode", choices=["eval", "validation", "scan_all"], required=True,
        help="eval: single eval dir | validation: training val dir | scan_all: whole output_helios/",
    )
    # single-dir modes
    parser.add_argument("--video_dir", default=None, help="[eval/validation] Directory containing .mp4 files")
    # scan_all mode
    parser.add_argument("--output_base_dir", default=None, help="[scan_all] Root dir, e.g. ./output_helios")
    parser.add_argument("--force", action="store_true", help="[scan_all] Overwrite existing html/results.html files")
    # shared label options
    parser.add_argument(
        "--label_path", "--csv_path", dest="label_path", default=None,
        help="[eval/scan_all] CSV or TSV with sample labels (auto-detected by extension)",
    )
    parser.add_argument("--first_frame_dir", default=None, help="[eval/scan_all] Directory with first-frame images")
    parser.add_argument("--chunk_actions", default=None, help="[eval] chunk_actions.json path (auto-detected in video_dir if omitted)")
    parser.add_argument("--output", default=None, help="[eval/validation] Output HTML path")
    args = parser.parse_args()

    if args.mode == "eval":
        if args.video_dir is None:
            parser.error("--video_dir is required for eval mode")
        if args.label_path is None:
            parser.error("--label_path is required for eval mode")
        output = args.output or os.path.join(args.video_dir, "html", "results.html")
        ca_path = args.chunk_actions
        if ca_path is None:
            auto = os.path.join(args.video_dir, "chunk_actions.json")
            if os.path.isfile(auto):
                ca_path = auto
        build_eval_html(args.video_dir, args.label_path, args.first_frame_dir, output,
                        chunk_actions_path=ca_path)

    elif args.mode == "validation":
        if args.video_dir is None:
            parser.error("--video_dir is required for validation mode")
        output = args.output or os.path.join(args.video_dir, "html", "results.html")
        build_validation_html(args.video_dir, output)

    else:  # scan_all
        base = args.output_base_dir or "./output_helios"
        scan_all_and_build(base, args.label_path, args.first_frame_dir, force=args.force)
