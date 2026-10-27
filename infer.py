import functools
import glob
import importlib
import os
import re


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

# Inference backend selection. The diffusers-version pipeline + transformer
# is the default (faster ContextParallel path) but does NOT implement OIM /
# EventMemory / ACHA / EAFR-learnable hooks. When training a checkpoint with
# any of those memory modules, set INFER_BACKEND=modules so the modules-
# version pipeline + transformer (which `train_helios.py` itself uses) is
# loaded — these classes share state_dict layout, so the same LoRA +
# extra-weights checkpoint loads cleanly on either backend.
_INFER_BACKEND = os.environ.get("INFER_BACKEND", "modules").strip().lower()
if _INFER_BACKEND not in ("diffusers", "modules"):
    raise ValueError(
        f"Unknown INFER_BACKEND={_INFER_BACKEND!r}; expected 'diffusers' or 'modules'."
    )
if _INFER_BACKEND == "modules":
    from helios.pipelines.pipeline_helios import HeliosPipeline
    from helios.scheduler.scheduling_helios import HeliosScheduler
    from helios.modules.transformer_helios import HeliosTransformer3DModel
else:
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
    from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
print(f"[infer-backend] {_INFER_BACKEND}")

# Modules pipeline renames the pyramid kwarg; pick the right name once at
# import time so call sites can spread it via **{}.
_PIPE_PYRAMID_KWARG = (
    "stage2_num_inference_steps_list" if _INFER_BACKEND == "modules"
    else "pyramid_num_inference_steps_list"
)

# Training-aligned scheduler kwargs (modules backend only).
#
# train_helios.py validation calls helios.pipelines.HeliosPipeline (modules
# backend) and forwards --use_dynamic_shifting + --time_shift_type "linear"
# (see run_cluster.sh:768-769, train_helios.py:3473-3474). Without these
# flags the modules pipeline falls back to a uniform 0.999->0 sigma curve
# and Euler stage1 step, both of which diverge from the training noise
# distribution and produce blurry rollouts.
#
# - use_dynamic_shifting=True  ->  pipeline applies apply_schedule_shift to
#   build sigmas with token-count-aware mu (matches diffusers backend's
#   internal mu shift, which is permanently on).
# - time_shift_type="linear"   ->  matches CLI in run_cluster.sh / run.sh.
# - scheduler_type="unipc"     ->  picks UniPC stage1 step instead of the
#   modules scheduler's hard-wired Euler. Matches scheduler_config.json
#   ("scheduler_type": "unipc") and the diffusers backend stage1 path.
#
# The diffusers backend ignores these knobs (it always does mu shift and
# routes step() by config.scheduler_type internally) and would error on
# unknown kwargs, so we only pass them for modules.
_PIPE_TRAIN_ALIGN_KWARGS = (
    {
        "use_dynamic_shifting": True,
        "time_shift_type": "linear",
        "scheduler_type": "unipc",
    }
    if _INFER_BACKEND == "modules"
    else {}
)

from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.utils.keyboard_overlay import render_keyboard_overlay_pil
from helios.utils.utils_base import load_extra_components

from diffusers import ContextParallelConfig
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image, load_video


def _export_with_optional_overlay(output_frames, output_path, *, fps,
                                  keys_list=None, mouse_list=None,
                                  frames_per_chunk=33, overlay_enabled=False):
    """Export pipe output to mp4. When overlay_enabled and per-chunk
    (keys, mouse) are known, burn the WASD/arrow keyboard overlay in first.
    Otherwise the call degenerates to plain export_to_video.

    pipe(...).frames[0] returns one of:
      - list of PIL.Image (output_type='pil', the default)
      - numpy float32 array in [0, 1] (output_type='np')
      - numpy uint8 array in [0, 255] (rare)
    The conversion below handles all three uniformly.
    """
    if overlay_enabled and keys_list and mouse_list:
        from PIL import Image as _Image

        def _to_pil(f):
            if isinstance(f, _Image.Image):
                return f
            arr = np.asarray(f)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            return _Image.fromarray(arr)

        pil_frames = [_to_pil(f) for f in output_frames]
        overlaid = render_keyboard_overlay_pil(
            pil_frames, list(keys_list), list(mouse_list),
            frames_per_chunk=frames_per_chunk,
        )
        export_to_video(overlaid, output_path, fps=fps)
    else:
        export_to_video(output_frames, output_path, fps=fps)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Base")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="BestWishYsh/Helios-Base",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--partial_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_folder", type=str, default="./output_helios")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="auto",
        choices=["auto", "_flash_3", "flash", "_native_flash", "native"],
        help="Attention backend. 'auto' tries _flash_3 → flash → _native_flash → native.",
    )

    # === Generation parameters ===
    # environment
    parser.add_argument(
        "--sample_type",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "v2v"],
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    # base
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=99)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    # cfg zero
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)
    # stage 1
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    # stage 2
    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--pyramid_num_inference_steps_list", type=int, nargs="+", default=[20, 20, 20])
    # stage 3
    parser.add_argument("--is_skip_first_chunk", action="store_true")
    parser.add_argument("--is_amplify_first_chunk", action="store_true")

    # === Prompts ===
    parser.add_argument("--use_interpolate_prompt", action="store_true")
    parser.add_argument("--interpolation_steps", type=int, default=3)
    parser.add_argument("--interpolate_time", type=int, default=7)
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_noise_sigma_min", type=float, default=0.111, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--image_noise_sigma_max", type=float, default=0.135, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--video_noise_sigma_min", type=float, default=0.111, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--video_noise_sigma_max", type=float, default=0.135, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )
    parser.add_argument(
        "--prompt_txt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base_image_prompt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_prompt_csv_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--interactive_prompt_csv_path",
        type=str,
        default=None,
    )

    # === Action Conditioning ===
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to run (0 = all)")
    parser.add_argument("--random_sample", action="store_true",
                        help="Randomly sample max_samples rows (uses --seed as random_state) "
                             "instead of df.head(max_samples). Stable across trajectories when "
                             "--seed is fixed, so cross-trajectory comparison still uses the same "
                             "first-frames.")
    parser.add_argument("--action_embeds_cache", type=str, default=None, help="Path to action_embeds_cache.pt")
    parser.add_argument("--action_keys", type=str, default=None, help="Keyboard action (e.g. W, W+A, None)")
    parser.add_argument("--action_mouse", type=str, default=None, help="Mouse action (e.g. →, ↑←, ·)")
    parser.add_argument("--action_cond_mode", type=str, default="none",
                        choices=["none", "adaln", "patch_add", "patch_add_perblock"],
                        help="Action conditioning mode (must match training)")
    parser.add_argument("--action_patch_use_dir", type=int, default=1,
                        help="Enable Fourier direction branch in ActionPatchEmbedder (0=off, 1=on)")
    parser.add_argument("--action_patch_id_use_mlp", type=int, default=1,
                        help="ActionPatchEmbedder id branch: 1=MLP bottleneck (new), 0=direct embed (legacy 8-card)")
    parser.add_argument("--action_patch_simple", type=int, default=0,
                        help="Simplified (Astra-style) action path: 1=small shared feature + single-Linear per-block adapter; 0=legacy two-branch + bottleneck MLP")
    parser.add_argument("--action_feature_dim", type=int, default=256,
                        help="Shared low-rank action feature dim for simple mode (default 256)")
    parser.add_argument("--action_adapter_bottleneck", type=int, default=1024,
                        help="Bottleneck dim for legacy per-block ActionBlockAdapter (only used when action_patch_simple=0)")
    # idea2 Plucker ray FiLM
    parser.add_argument("--use_plucker", action="store_true",
                        help="Enable shared PluckerFiLM head at inference (must match training checkpoint).")
    parser.add_argument("--plucker_only_rays_d", action="store_true",
                        help="Use 3D plucker (rays_d only); default is 6D [rays_o, rays_d].")
    parser.add_argument("--plucker_vae_space_down", type=int, default=8,
                        help="Wan VAE spatial downsample factor (must match training).")
    parser.add_argument("--camera_pt_dir", type=str, default=None,
                        help="Optional directory of .pt files containing per-video "
                             "'camera_poses' + 'camera_intrinsics' to feed plucker FiLM. "
                             "When provided, the inference rollout uses the corresponding "
                             "video's camera trajectory. Without it, plucker stays inactive.")
    parser.add_argument("--camera_traj_dir", type=str, default=None,
                        help="Optional directory of pre-recorded camera trajectories. "
                             "Layout: <traj_dir>/<id>/poses.npy (N,4,4) [+ intrinsics.npy "
                             "(4,) or (N,4)]. Built by tools/build_camera_traj_library.py.")
    parser.add_argument("--camera_traj_default_name", type=str, default=None,
                        help="Fallback trajectory name (subdir of --camera_traj_dir) when "
                             "the per-id directory is absent. Useful for sweeping a CSV "
                             "with a single canned trajectory like 'forward'.")
    parser.add_argument("--camera_action_string", type=str, default=None,
                        help="lingbot-style action string DSL (e.g. 'w-33,wl-33,none-33,s-33'). "
                             "Synthesises a c2w trajectory via the WASD/IJKL kinematics. "
                             "Has lower priority than --camera_pt_dir / --camera_traj_dir.")
    parser.add_argument("--camera_from_helios_actions", action="store_true",
                        help="Derive a per-chunk c2w trajectory from the same per-chunk "
                             "(action_keys, action_mouse) schedule used by AdaLN/cross-attn. "
                             "Lowest priority — only kicks in when no other camera source "
                             "is available and --use_plucker is set.")
    parser.add_argument("--camera_move_speed", type=float, default=None,
                        help="Override DEFAULT_MOVE_SPEED (0.05) for action→c2w conversion. "
                             "Translation gets normalised per chunk so this is mostly cosmetic.")
    parser.add_argument("--camera_rotate_speed_deg", type=float, default=None,
                        help="Override DEFAULT_ROTATE_SPEED_RAD (0.5°/frame). Calibrated "
                             "against seadance2 training stats: q75 per-chunk rotation ≈ 13°.")
    parser.add_argument("--camera_intrinsics_from_pt_dir", type=str, default=None,
                        help="When set with Mode B/C/D (canned/DSL camera), replace the "
                             "constant DEFAULT_INTRINSICS_384x640 with intrinsics sampled "
                             "from this directory of training-format .pt files. Each video "
                             "id is hashed to a deterministic file pick, so the same id "
                             "always gets the same fx/fy across runs. Recovers the training "
                             "FOV distribution (q25-q75 = 30°-90° range) instead of a fixed "
                             "65°. No effect on Mode A (which already uses real per-video Ks).")
    parser.add_argument("--action_txt_path", type=str, default=None,
                        help="Per-chunk action txt. Each line = one chunk, with (keys) and (mouse) in parentheses.")
    parser.add_argument("--zero_prompt_embeds", action="store_true",
                        help="Zero out T5 prompt_embeds at inference (matches training global_prompt_dropout_p=1.0). "
                             "Keeps attention mask intact; both positive and negative embeds become zero so CFG text delta=0.")
    parser.add_argument("--overlay_keyboard", action="store_true",
                        help="Burn WASD (bottom-left) + arrow-key (bottom-right) glyph overlay into "
                             "the per-sample mp4 using the resolved per-chunk (keys, mouse). "
                             "Default OFF so existing eval pipelines (VBench / VLM judge) keep "
                             "scoring clean frames; opt in when you want shareable demo clips.")

    # === Dual-Memory Helios (branch idea1) ===
    # These mirror scripts/training/configs/*.yaml -> training_config.action_memory.
    # Must match the flags used at training time for the checkpoint being loaded.
    parser.add_argument("--eafr_enabled", action="store_true",
                        help="Enable Phase 1 Event-Aware Frame Re-assignment at inference.")
    parser.add_argument("--eafr_alpha", type=float, default=1.0)
    parser.add_argument("--eafr_beta", type=float, default=0.5)
    parser.add_argument("--eafr_tau", type=float, default=8.0)
    parser.add_argument("--eafr_learnable", action="store_true",
                        help="Enable Phase 2a learnable EAFR scorer (requires --eafr_enabled).")
    # Phase 1b dedup — penalises picking redundant frames during the greedy
    # bucket fill. Must match training (training_config.action_memory.eafr_dedup_*)
    # otherwise long/mid/short bucket assignment drifts from training distribution.
    parser.add_argument("--eafr_dedup_enabled", action="store_true",
                        help="Enable Phase 1b EAFR redundancy penalty (requires --eafr_enabled).")
    parser.add_argument("--eafr_dedup_lambda", type=float, default=0.3,
                        help="Strength of the dedup penalty. 0.0 reproduces Phase 1; 0.2-0.5 typical.")
    parser.add_argument("--eafr_dedup_metric", type=str, default="latent",
                        choices=["latent", "clip"],
                        help="Similarity backend for dedup. 'latent' (cheap) or 'clip' "
                             "(needs precomputed sim_mat; falls back to latent if absent).")
    # IS_AMPLIFY_HISTORY must be passed at construction time. Without it,
    # HeliosAttention is built with is_amplify_history=False -> no history_key_scale
    # parameter -> enable_action_memory() can't attach ACHA's action_delta_mlp
    # (the for-loop at transformer_helios.py:1382-1384 skips every block).
    # Same root cause as the merge_lora_action.py is_amplify_history bug.
    parser.add_argument("--is_amplify_history", action="store_true",
                        help="Build HeliosAttention with is_amplify_history=True. Required "
                             "for any checkpoint trained with IS_AMPLIFY_HISTORY=true (which "
                             "includes the paper Table 4 'full' config and is a prerequisite for ACHA).")
    parser.add_argument("--history_scale_mode", type=str, default="per_head",
                        choices=["scalar", "per_head"],
                        help="Must match training-time HISTORY_SCALE_MODE; only consulted "
                             "when --is_amplify_history is set.")
    parser.add_argument("--acha_enabled", action="store_true",
                        help="Enable Phase 2b ACHA (requires --is_amplify_history and a checkpoint "
                             "trained with is_amplify_history=True + acha_enabled=true).")
    parser.add_argument("--acha_bottleneck", type=int, default=256)
    parser.add_argument("--event_memory_enabled", action="store_true",
                        help="Enable Phase 3 Event Memory Bank at inference.")
    parser.add_argument("--event_memory_k", type=int, default=16)
    parser.add_argument("--event_memory_num_action_classes", type=int, default=64)
    # ---- Persistent Memory Bank: Object writer (OIM) ----
    parser.add_argument("--object_memory_enabled", action="store_true",
                        help="Enable Phase 6 Persistent Memory Bank Object writer at inference. "
                             "Requires the checkpoint to have been built with object_memory_enabled=True.")
    parser.add_argument("--object_memory_k", type=int, default=16,
                        help="Bank capacity (max anchors retained). Default 16.")
    parser.add_argument("--object_memory_per_chunk", type=int, default=3,
                        help="Anchors written per qualifying chunk (writer gate). Default 3.")
    parser.add_argument("--object_memory_dino_backend", type=str, default="dinov3",
                        choices=["dinov3", "dinov2"])
    parser.add_argument("--object_memory_keyframes_per_chunk", type=int, default=5,
                        help="Keyframes sampled per chunk for online DINO at inference.")
    parser.add_argument("--chunk_phase_txt_path", type=str, default=None,
                        help="Optional path to a per-chunk phase file. Each line = one chunk, one of "
                             "approaching / contact / manipulating / completing / post_action. "
                             "When omitted the pipeline falls back to the keys/mouse heuristic.")
    parser.add_argument("--chunk_prompt_pt_path", type=str, default=None,
                        help="Optional path to a .pt carrying 'chunk_prompt_embeds' (list of "
                             "(seq_len, dim) UMT5 tensors, same format as produced by "
                             "tools/image_caption/encode_chunk_prompts.py). When provided, the "
                             "pipeline swaps prompt_embeds per chunk to list[k] during rollout.")
    parser.add_argument("--chunk_annotations_dir", type=str, default=None,
                        help="Optional dir of <id>.json carrying per-sample per-chunk "
                             "annotations (e.g. data/seadance2_Ibench/chunk_annotations). "
                             "Each JSON's chunks[i].description is written into "
                             "inference_meta['samples'][id]['chunk_descs'] (truncated to "
                             "num_chunks) so visualize_results renders the per-chunk prompt "
                             "in the HTML. Cosmetic only; does not affect generation.")
    parser.add_argument("--prompt_mode", type=str, default="auto",
                        choices=["auto", "global", "chunk"],
                        help="Which prompt to feed into the model:\n"
                             "  global = use the CSV's `prompt` column (single video-level "
                             "caption shared across all chunks). Same conditioning every "
                             "baseline gets — pick this for paper-fair comparison.\n"
                             "  chunk  = use per-chunk UMT5 embeds from the camera_pt_dir's "
                             "<vid>.pt 'chunk_prompt_embeds' field (10 separate embeds per "
                             "video, one per 33-frame chunk). Errors out if any video lacks "
                             "the field.\n"
                             "  auto   = chunk when available (camera_pt_dir set + .pt has "
                             "chunk_prompt_embeds), else global. Default.")

    # === Context parallelism ===
    # Please refer to https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#context-parallelism
    parser.add_argument("--enable_parallelism", action="store_true")
    parser.add_argument(
        "--cp_backend",
        type=str,
        choices=["ring", "ulysses", "unified", "ulysses_anything"],
        default="ulysses",
        help="Context parallel backend to use.",
    )

    # === Group-Offloading ===
    # Please refer to https://huggingface.co/docs/diffusers/main/en/optimization/memory#group-offloading
    parser.add_argument("--enable_low_vram_mode", action="store_true")
    parser.add_argument(
        "--group_offloading_type",
        type=str,
        choices=["leaf_level", "block_level"],
        default="leaf_level",
        help="Specifies the granularity for group CPU offloading. Choose between 'leaf_level' (individual modules) or 'block_level' (entire blocks).",
    )
    parser.add_argument(
        "--num_blocks_per_group",
        type=str,
        default="4",
        help="The number of blocks to bundle together in each offloading group. Only relevant when using block-level offloading.",
    )

    return parser.parse_args()


@functools.lru_cache(maxsize=1)
def _resolve_attention_backend() -> str:
    """Probe once per process and cache the best available attention backend."""
    import torch as _torch
    cuda_major = _torch.cuda.get_device_capability()[0]
    candidates = (
        ["_flash_3", "flash", "_native_flash", "native"] if cuda_major >= 9
        else ["flash", "_native_flash", "native"]
    )
    for backend in candidates:
        try:
            # Import diffusers check function directly to avoid building a full model
            from diffusers.models.attention_dispatch import _check_attention_backend_requirements
            from diffusers.models.attention_dispatch import AttentionBackendName
            _check_attention_backend_requirements(AttentionBackendName(backend))
            return backend
        except Exception:
            continue
    return "native"


def read_prompt_table(path: str) -> pd.DataFrame:
    """Read a prompt table (CSV or TSV) and normalize to a common schema.

    Output always has at least:
      - id         : unique sample identifier (str)
      - prompt     : generation prompt (str)
      - image_name : first-frame filename relative to base_image_prompt_path (str, may be empty)

    Supported input formats:
      CSV  columns: id, prompt, image_name, [action_class, action_keys, action_mouse, ...]
      TSV  columns: action_class, prompt, video_name, [action_keys, action_mouse, ...]
      Sekai CSV   : videoFile, caption, [cameraFile, location, scene, ...]

    Multiple paths can be separated by commas to merge several CSV/TSV files.
    """
    paths = [p.strip() for p in path.split(",")]
    frames = []
    for p in paths:
        suffix = os.path.splitext(p)[1].lower()
        if suffix in {".tsv", ".tab"}:
            frames.append(pd.read_csv(p, sep="\t"))
        elif suffix == ".csv":
            frames.append(pd.read_csv(p, sep=","))
        else:
            frames.append(pd.read_csv(p, sep=None, engine="python"))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Sekai CSV format: videoFile → video_name, caption → prompt
    if "videoFile" in df.columns:
        rename_map = {"videoFile": "video_name", "caption": "prompt"}
        if "caption_original" in df.columns:
            rename_map["caption_original"] = "prompt_original"
        df = df.rename(columns=rename_map)

    # TSV / Sekai format uses video_name instead of id / image_name — normalize here
    if "video_name" in df.columns and "id" not in df.columns:
        stems = df["video_name"].astype(str).map(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        df = df.copy()
        df["id"] = stems
        if "image_name" not in df.columns:
            df["image_name"] = stems + ".jpg"

    # Ensure image_name column always exists
    if "image_name" not in df.columns:
        df["image_name"] = ""

    return df


def main():
    args = parse_args()

    assert not (args.enable_low_vram_mode and args.enable_compile), (
        "enable_low_vram_mode and enable_compile cannot be used together."
    )

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_available() and "RANK" in os.environ:
        if args.cp_backend == "ulysses_anything":
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        else:
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        world_size = dist.get_world_size()
        torch.cuda.set_device(device)
        assert world_size == 1 or not args.enable_low_vram_mode, "enable_low_vram_mode is only for single GPU."
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    prompt = None
    image_path = None
    video_path = None
    interpolate_time_list = None
    if args.sample_type == "t2v" and args.prompt is None:
        prompt = "An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt , he wears a brown beret and glasses and has a very professorial appearance, and the end he offers a subtle closed-mouth smile as if he found the answer to the mystery of life, the lighting is very cinematic with the golden light and the Parisian streets and city in the background, depth of field, cinematic 35mm film."
    elif args.sample_type == "i2v" and (args.image_path is None and args.prompt is None):
        image_path = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        )
        prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    elif args.sample_type == "v2v" and (args.video_path is None and args.prompt is None):
        video_path = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        )
        prompt = "A robot standing on a mountain top. The sun is setting in the background."
    else:
        image_path = args.image_path
        video_path = args.video_path
        prompt = args.prompt

    # Auto-detect whether transformer weights are in a "transformer/" subfolder or
    # directly at transformer_path (e.g. merged_transformer dirs that are already flat).
    _has_subfolder = os.path.isdir(os.path.join(args.transformer_path, "transformer"))
    # Tensor names stored in the transformer checkpoint (headers only). Merged
    # ActWorld weights already contain Plücker FiLM and history_key_scale, which
    # the re-initialisation steps below must then leave untouched.
    from safetensors import safe_open

    _ckpt_keys = set()
    _ckpt_dir = os.path.join(args.transformer_path, "transformer") if _has_subfolder else args.transformer_path
    for _f in glob.glob(os.path.join(_ckpt_dir, "*.safetensors")):
        with safe_open(_f, "pt") as _h:
            _ckpt_keys.update(_h.keys())
    _transformer_kwargs = {}
    if args.action_cond_mode != "none":
        _transformer_kwargs["action_cond_mode"] = args.action_cond_mode
        _transformer_kwargs["action_patch_use_dir"] = bool(args.action_patch_use_dir)
        _transformer_kwargs["action_patch_id_use_mlp"] = bool(args.action_patch_id_use_mlp)
        _transformer_kwargs["action_patch_simple"] = bool(args.action_patch_simple)
        _transformer_kwargs["action_feature_dim"] = int(args.action_feature_dim)
        if args.action_cond_mode == "patch_add_perblock":
            _transformer_kwargs["action_adapter_bottleneck"] = args.action_adapter_bottleneck
    if args.use_plucker:
        _transformer_kwargs["use_plucker"] = True
        _transformer_kwargs["plucker_only_rays_d"] = bool(args.plucker_only_rays_d)
        _transformer_kwargs["plucker_vae_space_down"] = int(args.plucker_vae_space_down)
    # IS_AMPLIFY_HISTORY: must be in kwargs so attention layers get
    # history_key_scale = ones(heads). enable_action_memory() then hooks ACHA
    # onto these blocks. See note above the argparse entry.
    if args.is_amplify_history:
        _transformer_kwargs["is_amplify_history"] = True
        _transformer_kwargs["history_scale_mode"] = args.history_scale_mode
    # NOTE: helios.modules.transformer_helios.HeliosTransformer3DModel.from_pretrained
    # is a custom override that does NOT accept arbitrary **kwargs — config
    # overrides like use_plucker / action_cond_mode must be passed via the
    # ``transformer_additional_kwargs=`` named arg.  The diffusers-version
    # backend (helios.diffusers_version.*) inherits the standard ModelMixin
    # from_pretrained which DOES accept **kwargs, so we pass both forms
    # and the one the active backend supports wins.
    if _INFER_BACKEND == "modules":
        transformer = HeliosTransformer3DModel.from_pretrained(
            args.transformer_path,
            subfolder="transformer" if _has_subfolder else None,
            torch_dtype=args.weight_dtype,
            transformer_additional_kwargs=_transformer_kwargs,
        )
    else:
        transformer = HeliosTransformer3DModel.from_pretrained(
            args.transformer_path,
            subfolder="transformer" if _has_subfolder else None,
            torch_dtype=args.weight_dtype,
            **_transformer_kwargs,
        )
    # Materialize action modules from meta → CPU (not in base checkpoint,
    # weights will be loaded from LoRA file below)
    if hasattr(transformer, "action_patch_embedder"):
        transformer.action_patch_embedder.to_empty(device="cpu")
        for p in transformer.action_patch_embedder.parameters():
            torch.nn.init.zeros_(p)
        print("[Action] Materialized action_patch_embedder from meta to CPU")
    if args.action_cond_mode == "patch_add_perblock":
        for _blk in transformer.blocks:
            if hasattr(_blk, "action_adapter"):
                _blk.action_adapter.to_empty(device="cpu")
                for p in _blk.action_adapter.parameters():
                    torch.nn.init.zeros_(p)
        print("[Action] Materialized per-block action_adapter from meta to CPU")
    # Plucker FiLM is also added via use_plucker kwarg, not in the base ckpt;
    # materialize it the same way so weight loading from extras can populate it.
    # Cast to args.weight_dtype so it matches the rest of the transformer
    # (otherwise the Linear matmul inside modulate() raises "mat1 and mat2 must
    # have the same dtype, but got Float and BFloat16").
    if args.use_plucker and hasattr(transformer, "plucker_film") and transformer.plucker_film is not None:
        if any(k.startswith("plucker_film.") for k in _ckpt_keys):
            transformer.plucker_film.to(dtype=args.weight_dtype)
            print(f"[Plucker] plucker_film loaded from {_ckpt_dir} (dtype={args.weight_dtype})")
        else:
            transformer.plucker_film.to_empty(device="cpu")
            transformer.plucker_film._init_weights()
            transformer.plucker_film.to(dtype=args.weight_dtype)
            print(f"[Plucker] Materialized plucker_film (dtype={args.weight_dtype}) from meta to CPU")
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    if args.attention_backend != "auto":
        transformer.set_attention_backend(args.attention_backend)
        print(f"[Attention] Using user-specified backend: {args.attention_backend}")
    else:
        transformer.set_attention_backend(_resolve_attention_backend())
        print(f"[Attention] Using backend: {_resolve_attention_backend()}")

    # --- Dual-Memory Helios (branch idea1): instantiate optional modules ---
    # Must happen BEFORE the LoRA weights load below, otherwise the safetensors
    # keys for action_delta_mlp / event_memory_bank / eafr_scorer have no home.
    # NOTE: pure EAFR-static + dedup (Phase 1 / 1b) needs no learnable module —
    # the scoring lives entirely inside the pipeline and consumes runtime
    # kwargs (eafr_alpha/beta/tau, eafr_dedup_*). Only ACHA / EventMemory /
    # ObjectMemory / EAFR-learnable need enable_action_memory().
    # The diffusers-version transformer used at inference does not implement
    # enable_action_memory(); attempting to call it for pure EAFR-static
    # would AttributeError. Gate accordingly.
    _need_enable_action_memory = (
        args.acha_enabled or args.event_memory_enabled
        or args.eafr_learnable or args.object_memory_enabled
    )
    if _need_enable_action_memory and _INFER_BACKEND != "modules":
        raise RuntimeError(
            "Memory modules (ACHA / EventMemory / ObjectMemory / EAFR-learnable) "
            "are only implemented on the modules-version pipeline. "
            "Re-run with INFER_BACKEND=modules. Current backend: "
            f"{_INFER_BACKEND}."
        )
    if _need_enable_action_memory:
        _kept_history_scale = {
            n: p.detach().clone()
            for n, p in transformer.named_parameters()
            if n.endswith("history_key_scale") and n in _ckpt_keys
        }
        transformer.enable_action_memory(
            acha_enabled=args.acha_enabled,
            acha_bottleneck=args.acha_bottleneck,
            event_memory_enabled=args.event_memory_enabled,
            event_memory_k=args.event_memory_k,
            event_memory_num_action_classes=args.event_memory_num_action_classes,
            eafr_enabled=args.eafr_enabled,
            eafr_learnable=args.eafr_learnable,
            # --- Persistent Memory Bank: Object writer (OIM) ---
            object_memory_enabled=args.object_memory_enabled,
            object_memory_k=args.object_memory_k,
            object_memory_per_chunk=args.object_memory_per_chunk,
            object_memory_dino_backend=args.object_memory_dino_backend,
        )
        if _kept_history_scale:
            with torch.no_grad():
                for n, p in transformer.named_parameters():
                    if n in _kept_history_scale:
                        p.copy_(_kept_history_scale[n])
            print(f"[Action-Memory] kept {len(_kept_history_scale)} history_key_scale tensors from {_ckpt_dir}")
        # Cast every action-memory module the same way Plücker FiLM is
        # cast above: nn.Module ctors default to fp32, while the rest of
        # the transformer runs in args.weight_dtype (bf16 / fp16).  Without
        # this cast the bank's Linear layers fire with float32 weights vs
        # bfloat16 inputs and torch raises:
        #   RuntimeError: Input type (float) and bias type (c10::BFloat16)
        #                 should be the same
        for _attr in ("object_memory_bank", "event_memory_bank", "eafr_scorer", "acha_action_embed"):
            _mod = getattr(transformer, _attr, None)
            if _mod is not None:
                _mod.to(dtype=args.weight_dtype)
                print(f"[Action-Memory] cast {_attr} to {args.weight_dtype}")
        # ACHA's per-block action_delta_mlp also needs casting.
        for _i, _blk in enumerate(getattr(transformer, "blocks", []) or []):
            _attn = getattr(_blk, "attn1", None)
            _mlp = getattr(_attn, "action_delta_mlp", None) if _attn is not None else None
            if _mlp is not None:
                _mlp.to(dtype=args.weight_dtype)
        # Segment embeds + type-tag are nn.Parameter (not nn.Module): the
        # outer transformer.to(...) covers them, but if it has already run
        # we cast explicitly to be safe.
        for _attr in ("event_segment_embed", "object_segment_embed"):
            _p = getattr(transformer, _attr, None)
            if _p is not None and hasattr(_p, "data"):
                _p.data = _p.data.to(dtype=args.weight_dtype)
    if (args.eafr_enabled or _need_enable_action_memory):
        print(
            f"[Action-Memory] enabled: "
            f"EAFR={args.eafr_enabled} (learnable={args.eafr_learnable}, "
            f"dedup={args.eafr_dedup_enabled}) "
            f"ACHA={args.acha_enabled} EventMem={args.event_memory_enabled} "
            f"OIM={args.object_memory_enabled}"
        )

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    scheduler = HeliosScheduler.from_pretrained(
        args.base_model_path,
        subfolder="scheduler",
    )
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=args.weight_dtype,
    )

    if args.lora_path is not None:
        import safetensors.torch
        lora_file = os.path.join(args.lora_path, "pytorch_lora_weights.safetensors")
        if os.path.isfile(args.lora_path):
            lora_file = args.lora_path
        full_state = safetensors.torch.load_file(lora_file)

        lora_state = {k: v for k, v in full_state.items() if "lora" in k}
        extra_state = {k: v for k, v in full_state.items() if "lora" not in k}
        _action_keywords = ("action_embedder", "action_proj", "action_scale_shift_table", "action_patch_embedder")
        _n_action = sum(1 for k in extra_state if any(w in k for w in _action_keywords))
        _n_norm = len(extra_state) - _n_action
        print(f"Loaded checkpoint: {len(lora_state)} LoRA, {_n_norm} norm, {_n_action} action keys")

        if lora_state:
            from peft import LoraConfig, set_peft_model_state_dict
            lora_rank = lora_state[next(k for k in lora_state if "lora_A" in k)].shape[0]
            raw_exclude = ["down", "up", "action_embedder", "action_proj", "action_patch_embedder",
                           "action_adapter", "plucker_film"]
            # Mirror train_helios.py: dual-memory modules must also be
            # excluded from LoRA at inference, otherwise PEFT will try to
            # wrap their Linears (which were trained as full-precision in
            # the extras file) and produce "missing LoRA keys" + load
            # mismatch on extras_state. Only add modules that were
            # actually instantiated via enable_action_memory().
            if hasattr(transformer, "object_memory_bank") and transformer.object_memory_bank is not None:
                raw_exclude.append("object_memory_bank")
            if hasattr(transformer, "event_memory_bank") and transformer.event_memory_bank is not None:
                raw_exclude.append("event_memory_bank")
            if hasattr(transformer, "eafr_scorer") and transformer.eafr_scorer is not None:
                raw_exclude.append("eafr_scorer")
            # ACHA: per-block delta MLP — exclude each one.
            for _i, _blk in enumerate(getattr(transformer, "blocks", []) or []):
                _attn = getattr(_blk, "attn1", None)
                if _attn is not None and getattr(_attn, "action_delta_mlp", None) is not None:
                    raw_exclude.append(f"blocks.{_i}.attn1.action_delta_mlp")
            infer_exclude = []
            all_mod_names = {n for n, _ in transformer.named_modules()}
            for exc in raw_exclude:
                if exc in all_mod_names:
                    children = [
                        n for n, m in transformer.named_modules()
                        if (n == exc or n.startswith(exc + "."))
                        and isinstance(m, (torch.nn.Linear, torch.nn.Embedding))
                    ]
                    infer_exclude.extend(children if children else [exc])
                else:
                    infer_exclude.append(exc)
            # Mirror training (train_helios.py:472-486): build explicit
            # target_modules from all nn.Linear, then append "patch_embedding"
            # (Conv3d input projection — wrapped via is_train_lora_patch_embedding
            # at training, but PEFT's "all-linear" string never matches Conv3d so
            # the trained patch_embedding LoRA delta would otherwise be dropped).
            target_modules = sorted({
                n for n, m in transformer.named_modules()
                if isinstance(m, torch.nn.Linear)
                and not any(exc in n for exc in infer_exclude)
                and "norm" not in n
            })
            # Append patch_embedding only if the checkpoint actually trained a
            # LoRA on it (presence of patch_embedding.lora_* keys in lora_state).
            if any("patch_embedding.lora_" in k for k in lora_state):
                target_modules.append("patch_embedding")
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
                exclude_modules=infer_exclude,
            )
            transformer.add_adapter(lora_config, adapter_name="default")
            transformer_key_prefix = "transformer."
            peft_state = {k[len(transformer_key_prefix):]: v for k, v in lora_state.items()
                          if k.startswith(transformer_key_prefix)}
            incompatible = set_peft_model_state_dict(transformer, peft_state, adapter_name="default")
            # Filter to only true LoRA keys (PEFT also reports base norm/bias
            # params as "missing" which is just noise — those are init-state).
            _missing_lora = [k for k in incompatible.missing_keys if ".lora_" in k]
            _unexpected_lora = [k for k in incompatible.unexpected_keys if ".lora_" in k]
            if _missing_lora:
                # Missing LoRA pairs stay at init: lora_A=Gaussian, lora_B=zeros
                # → delta = lora_B @ lora_A = 0, identity to base model. Safe.
                _mods = sorted({k.rsplit(".lora_", 1)[0] for k in _missing_lora})
                print(f"  Warning: {len(_missing_lora)} missing LoRA keys "
                      f"({len(_mods)} unique modules; identity at init, so safe). "
                      f"e.g. {_mods[:3]}")
            if _unexpected_lora:
                # Unexpected = trained LoRA deltas with no destination on the
                # inference model. These are silently dropped — investigate.
                print(f"  Warning: {len(_unexpected_lora)} unexpected LoRA keys "
                      f"(trained deltas DROPPED — fix LoraConfig target_modules): "
                      f"{_unexpected_lora}")

        if extra_state:
            strip_prefix = "transformer.transformer."
            extra_renamed = {}
            for k, v in extra_state.items():
                nk = k[len(strip_prefix):] if k.startswith(strip_prefix) else k
                extra_renamed[nk] = v
            info = transformer.load_state_dict(extra_renamed, strict=False)
            loaded_count = len(extra_renamed) - len(info.unexpected_keys)
            print(f"  Loaded {loaded_count} extra weights (norm + action) into transformer")
            if info.unexpected_keys:
                print(f"  Warning: {len(info.unexpected_keys)} keys not loaded: "
                      f"{info.unexpected_keys[:5]}{'...' if len(info.unexpected_keys) > 5 else ''}")

        # New format: unified pytorch_extra_weights.safetensors (all full-train modules).
        # Bare-path keys, direct load_state_dict into transformer.
        extras_path = os.path.join(os.path.dirname(lora_file), "pytorch_extra_weights.safetensors")
        if os.path.exists(extras_path):
            extras_state = safetensors.torch.load_file(extras_path)
            info2 = transformer.load_state_dict(extras_state, strict=False)
            loaded_count2 = len(extras_state) - len(info2.unexpected_keys)
            print(f"  Loaded {loaded_count2} extras from {extras_path}")
            if info2.unexpected_keys:
                print(f"  Warning: {len(info2.unexpected_keys)} extras keys not loaded: "
                      f"{info2.unexpected_keys[:5]}{'...' if len(info2.unexpected_keys) > 5 else ''}")
            # Loaded extras state_dict may carry fp32 tensors (training was
            # in mixed precision and saved untouched).  Re-cast every
            # action-memory module after load so dtypes match args.weight_dtype.
            for _attr in ("object_memory_bank", "event_memory_bank", "eafr_scorer", "acha_action_embed"):
                _mod = getattr(transformer, _attr, None)
                if _mod is not None:
                    _mod.to(dtype=args.weight_dtype)
            for _blk in getattr(transformer, "blocks", []) or []:
                _attn = getattr(_blk, "attn1", None)
                _mlp = getattr(_attn, "action_delta_mlp", None) if _attn is not None else None
                if _mlp is not None:
                    _mlp.to(dtype=args.weight_dtype)
            for _attr in ("event_segment_embed", "object_segment_embed"):
                _p = getattr(transformer, _attr, None)
                if _p is not None and hasattr(_p, "data"):
                    _p.data = _p.data.to(dtype=args.weight_dtype)

        if args.partial_path is not None:
            if not hasattr(args, "training_config"):
                from argparse import Namespace

                args.training_config = Namespace()
            args.training_config.is_enable_stage1 = True
            args.training_config.restrict_self_attn = True
            args.training_config.is_amplify_history = True
            args.training_config.is_use_gan = True
            load_extra_components(args, transformer, args.partial_path)
    else:
        # Merged weights: ACHA MLPs, memory banks, EAFR scorer and segment embeds
        # are created by enable_action_memory() after from_pretrained, so load
        # them from the extras file next to the shards (or --partial_path).
        _extras_src = args.partial_path or _ckpt_dir
        if os.path.exists(os.path.join(_extras_src, "pytorch_extra_weights.safetensors")) or os.path.isfile(_extras_src):
            load_extra_components(args, transformer, _extras_src)

    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.enable_low_vram_mode:
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type=args.group_offloading_type,
            num_blocks_per_group=args.num_blocks_per_group if args.group_offloading_type == "block_level" else None,
            use_stream=True,
            record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    if args.zero_prompt_embeds:
        _orig_t5 = pipe._get_t5_prompt_embeds
        def _zero_t5(prompt, num_videos_per_prompt=1, max_sequence_length=226, device=None, dtype=None):
            embeds, mask = _orig_t5(prompt, num_videos_per_prompt, max_sequence_length, device, dtype)
            return torch.zeros_like(embeds), mask
        pipe._get_t5_prompt_embeds = _zero_t5
        if rank == 0:
            print("[zero_prompt_embeds] T5 prompt_embeds zeroed (matches global_prompt_dropout_p=1.0)")

    if world_size > 1 and args.enable_parallelism:
        if args.cp_backend == "ring":
            cp_config = ContextParallelConfig(ring_degree=world_size)
        elif args.cp_backend == "unified":
            cp_config = ContextParallelConfig(ring_degree=world_size // 2, ulysses_degree=world_size // 2)
        elif args.cp_backend == "ulysses":
            cp_config = ContextParallelConfig(ulysses_degree=world_size)
        elif args.cp_backend == "ulysses_anything":
            cp_config = ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True)
        else:
            raise ValueError(f"Unsupported cp_backend: {args.cp_backend}")

        pipe.transformer.enable_parallelism(config=cp_config)

    # Load action embedding cache if provided
    action_embeds_cache = None
    if args.action_embeds_cache is not None and os.path.exists(args.action_embeds_cache):
        action_embeds_cache = torch.load(args.action_embeds_cache, map_location="cpu", weights_only=False)
        print(f"Loaded action embedding cache with {len(action_embeds_cache)} entries")

    # ── Prompt → (keys, mouse) parsing (mirrors YUME generate_per_sample_captions) ──
    #
    # Sekai captions use diverse phrasing ("advances steadily forward",
    # "pans slightly to the right", "the drone glides forward",
    # "continues to move forward", "pans from left to right"), so we use
    # multiple complementary patterns to maximise coverage.
    _MOVE_VERBS = r"moves|advances|progresses|glides|proceeds|pushes|follows|tracks|navigates"
    _PAN_VERBS = r"pans|shifts|turns|sweeps|rotates"
    _TILT_VERBS = r"tilts|angles"
    _ALL_VERBS = f"{_MOVE_VERBS}|{_PAN_VERBS}|{_TILT_VERBS}"
    _ADVERBS = r"(?:\s+(?:steadily|smoothly|slightly|gently|gradually|subtly|then|occasionally|slowly|briefly|further|continuously|rapidly|quickly))*"
    _PREPS = r"(?:\s+(?:to the|toward the|towards the))?"
    _DIRS = r"forward|backward|upward|downward|left|right|up|down"
    _SUBJECT = r"the (?:camera|viewer(?:'s (?:perspective|gaze|view))?|drone)(?:\s+angle)?"

    _CLAUSE_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'({_ALL_VERBS}){_ADVERBS}{_PREPS}\s+'
        rf'({_DIRS})',
        re.IGNORECASE,
    )

    # "continues to move/pan/tilt forward" or "continues moving forward"
    _CONTINUES_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'continues{_ADVERBS}\s+(?:'
        rf'(?:to\s+)?(?:move|pan|tilt|advance|progress|glide|shift|push|turn|track|navigate)'
        rf'|moving|panning|tilting|advancing|progressing|gliding|shifting|pushing|turning'
        rf'){_ADVERBS}{_PREPS}\s+({_DIRS})',
        re.IGNORECASE,
    )

    # "continues its [steady] forward journey" / "continues its upward climb"
    _CONTINUES_ITS_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'continues\s+(?:its|the)\s+(?:steady\s+|smooth\s+)?({_DIRS})',
        re.IGNORECASE,
    )

    # "pans smoothly from left to right"
    _FROM_TO_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'(?:{_ALL_VERBS}){_ADVERBS}\s+'
        rf'from\s+(?:the\s+)?({_DIRS})\s+to\s+(?:the\s+)?({_DIRS})',
        re.IGNORECASE,
    )

    # "the camera/viewer moving forward" (gerund without finite verb)
    _GERUND_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'(moving|panning|tilting|advancing|progressing|gliding|shifting|pushing|turning)'
        rf'{_ADVERBS}{_PREPS}\s+({_DIRS})',
        re.IGNORECASE,
    )

    _DIR_NORMALIZE = {"upward": "up", "downward": "down"}
    _MOVE_TO_KEY = {"forward": "W", "backward": "S", "left": "A", "right": "D"}
    _PAN_VERBS_SET = {"pans", "shifts", "turns", "sweeps", "rotates",
                      "panning", "shifting", "turning"}
    _TILT_VERBS_SET = {"tilts", "angles", "tilting"}
    _ROT_TO_MOUSE = {
        "left": "←", "right": "→",
        "up": "↑", "down": "↓",
    }

    def _classify_verb(verb):
        """Classify a camera verb into 'move' (translation) or 'rotate' (pan/tilt)."""
        v = verb.lower()
        if v in _PAN_VERBS_SET or v in _TILT_VERBS_SET:
            return "rotate"
        return "move"

    def _verb_direction_to_action(verb, direction):
        """Map (verb, direction) → (keys, mouse)."""
        direction = _DIR_NORMALIZE.get(direction, direction)
        if _classify_verb(verb) == "move":
            return _MOVE_TO_KEY.get(direction, "None"), "·"
        else:
            return "None", _ROT_TO_MOUSE.get(direction, "·")

    def _collect_all_clause_matches(prompt_text):
        """Gather directional clause matches from all patterns, deduplicated.

        Each result carries .start, .end, .verb, .direction, and .text (the
        matched substring from the original prompt).
        """
        Match = type("Match", (), {})

        results = []
        for m in _CLAUSE_PAT.finditer(prompt_text):
            o = Match()
            o.start, o.end = m.start(), m.end()
            o.verb, o.direction = m.group(1).lower(), m.group(2).lower()
            o.text = prompt_text[m.start():m.end()].strip(", ")
            results.append(o)

        def _overlaps(new_start, new_end):
            return any(r.start <= new_start < r.end for r in results)

        for m in _CONTINUES_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                o.verb, o.direction = "moves", m.group(1).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        for m in _CONTINUES_ITS_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                o.verb, o.direction = "moves", m.group(1).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        for m in _FROM_TO_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                o.verb, o.direction = "pans", m.group(2).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        for m in _GERUND_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                gerund = m.group(1).lower()
                o.verb = gerund.rstrip("ing") + "s" if gerund not in _PAN_VERBS_SET and gerund not in _TILT_VERBS_SET else gerund
                o.direction = m.group(2).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        results.sort(key=lambda r: r.start)
        return results

    def parse_prompt_to_chunk_actions(prompt_text, num_chunks):
        """Parse camera clauses from prompt into per-chunk (keys, mouse, clause) triples.

        Adjacent move+rotation clauses (no significant text between them) are
        merged into a single combined action, e.g. "the camera moves forward,
        the camera pans left" → ("W", "←", "...") instead of two separate chunks.
        Non-camera text gaps (>=20 chars) become ("None", "·", "") segments.
        Segments are distributed evenly across num_chunks.
        """
        matches = _collect_all_clause_matches(prompt_text)
        if not matches:
            return [("None", "·", "")] * num_chunks

        MIN_ACTION_LEN = 20

        raw_clauses = []
        prev_end = 0
        for m in matches:
            gap = prompt_text[prev_end:m.start].strip(", ")
            raw_clauses.append((gap, m.verb, m.direction, m.text))
            prev_end = m.end
        trailing = prompt_text[prev_end:].strip(", .")

        segments = []
        i = 0
        while i < len(raw_clauses):
            gap, verb, direction, clause_text = raw_clauses[i]

            if len(gap) >= MIN_ACTION_LEN:
                segments.append(("None", "·", ""))

            keys, mouse = _verb_direction_to_action(verb, direction)

            if i + 1 < len(raw_clauses):
                next_gap, next_verb, next_dir, next_clause = raw_clauses[i + 1]
                can_merge = len(next_gap) < MIN_ACTION_LEN
                if can_merge:
                    is_move = _classify_verb(verb) == "move"
                    next_is_move = _classify_verb(next_verb) == "move"
                    if is_move and not next_is_move:
                        _, mouse = _verb_direction_to_action(next_verb, next_dir)
                        merged_text = f"{clause_text} + {next_clause}"
                        i += 2
                        segments.append((keys, mouse, merged_text))
                        continue
                    elif not is_move and next_is_move:
                        keys, _ = _verb_direction_to_action(next_verb, next_dir)
                        merged_text = f"{clause_text} + {next_clause}"
                        i += 2
                        segments.append((keys, mouse, merged_text))
                        continue

            segments.append((keys, mouse, clause_text))
            i += 1

        if len(trailing) >= MIN_ACTION_LEN:
            segments.append(("None", "·", ""))

        if not segments:
            return [("None", "·", "")] * num_chunks

        n = len(segments)
        base = num_chunks // n
        remainder = num_chunks % n
        actions = []
        for i, seg in enumerate(segments):
            steps = base + (1 if i < remainder else 0)
            actions.extend([seg] * steps)
        return actions

    # ── Per-chunk action override from txt file ──
    _PARENS_PAT = re.compile(r"\(([^)]*)\)")

    def parse_action_txt(path: str):
        """Parse a per-chunk action txt file.

        Each line has two parenthesised tokens:
          "... (W). ... (·). ..."  → ("W", "·")
          "... (D). ... (→). ..."  → ("D", "→")
        Returns list of (keys, mouse) tuples, one per line.
        """
        actions = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = _PARENS_PAT.findall(line)
                if len(tokens) >= 2:
                    actions.append((tokens[0], tokens[1]))
                elif len(tokens) == 1:
                    actions.append((tokens[0], "·"))
                else:
                    actions.append(("None", "·"))
        return actions

    # ── Unified action resolution ──

    def _pad_or_truncate(actions, num_chunks):
        """Pad (repeat last) or truncate an action list to exactly num_chunks."""
        if len(actions) < num_chunks:
            return actions + [actions[-1]] * (num_chunks - len(actions))
        return actions[:num_chunks]

    def _parse_chunk_actions_json(raw):
        """Parse a chunk_actions_json string → list of (keys, mouse) tuples."""
        import json as _json
        parsed = _json.loads(raw)
        return [(str(c[0] if isinstance(c, (list, tuple)) else c.get("keys", "None")),
                 str(c[1] if isinstance(c, (list, tuple)) else c.get("mouse", "·")))
                for c in parsed]

    def _load_chunk_prompt_embeds_from_pt(row_id):
        """Pull per-chunk UMT5 prompt embeddings from a training-format .pt.

        Looks under args.camera_pt_dir for `<id>.pt` or `<id>_*.pt`. The .pt's
        ``chunk_prompt_embeds`` field is a list of (seq_len, dim) tensors (one
        per chunk in the source video). Returns (embeds_list, descs_list) or
        (None, None) when no file is found or the field is missing/empty.

        This is the per-video equivalent of ``--chunk_prompt_pt_path`` (which
        loads a single shared .pt for the whole batch). When camera_pt_dir is
        set we prefer per-video embeds — same behavior as training.
        """
        if not args.camera_pt_dir:
            return None, None
        import glob as _glob
        pt_path = os.path.join(args.camera_pt_dir, f"{row_id}.pt")
        if not os.path.isfile(pt_path):
            _matches = sorted(_glob.glob(os.path.join(args.camera_pt_dir, f"{row_id}_*.pt")))
            pt_path = _matches[0] if _matches else None
        if pt_path is None or not os.path.isfile(pt_path):
            return None, None
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            return None, None
        embeds = blob.get("chunk_prompt_embeds", None)
        if not (isinstance(embeds, list) and embeds):
            return None, None
        descs = blob.get("chunk_descs", None)
        if isinstance(descs, list) and descs:
            descs = [str(x) for x in descs]
        else:
            descs = None
        return embeds, descs

    def _encode_chunk_descs_from_annotations(row_id):
        """Encode per-chunk captions from <chunk_annotations_dir>/<id>.json.

        Uses the same UMT5 call as tools/image_caption/encode_chunk_prompts.py,
        which produced the chunk_prompt_embeds stored in training .pt files, so
        the result matches those embeddings. Returns None when the file or its
        chunk descriptions are missing.
        """
        if not args.chunk_annotations_dir:
            return None
        ann_path = os.path.join(args.chunk_annotations_dir, f"{row_id}.json")
        if not os.path.isfile(ann_path):
            return None
        with open(ann_path) as _f:
            _ann = json.load(_f)
        descs = [str(c.get("description") or "") for c in _ann.get("chunks", [])]
        if not any(descs):
            return None
        from helios.utils.utils_base import encode_prompt

        with torch.no_grad():
            embeds, _ = encode_prompt(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                prompt=descs,
                device=pipe._execution_device,
            )
        return [e.cpu() for e in embeds]

    def _load_chunk_labels(row_id):
        """Per-video memory labels: (chunk_phase, chunk_has_interaction, action_class).

        These drive EAFR's phase prior, the Event / Object bank writers and
        ACHA. Read from the training-format .pt under --camera_pt_dir (same
        fields the training dataloader uses); fall back to
        --chunk_annotations_dir/<id>.json. Any field that cannot be found is
        returned as None, which the pipeline treats as "unlabelled".
        """
        import glob as _glob
        phases, has_int, action_class = None, None, None
        if args.camera_pt_dir:
            pt_path = os.path.join(args.camera_pt_dir, f"{row_id}.pt")
            if not os.path.isfile(pt_path):
                _matches = sorted(_glob.glob(os.path.join(args.camera_pt_dir, f"{row_id}_*.pt")))
                pt_path = _matches[0] if _matches else None
            if pt_path is not None and os.path.isfile(pt_path):
                try:
                    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
                except Exception:
                    blob = {}
                if isinstance(blob.get("chunk_phase"), list) and blob["chunk_phase"]:
                    phases = [("" if p is None else str(p)) for p in blob["chunk_phase"]]
                if isinstance(blob.get("chunk_has_interaction"), list) and blob["chunk_has_interaction"]:
                    has_int = [None if h is None else bool(h) for h in blob["chunk_has_interaction"]]
                if blob.get("action_class") not in (None, "", "None"):
                    action_class = str(blob["action_class"])
        if args.chunk_annotations_dir and (phases is None or has_int is None or action_class is None):
            ann_path = os.path.join(args.chunk_annotations_dir, f"{row_id}.json")
            if os.path.isfile(ann_path):
                try:
                    with open(ann_path) as _f:
                        _ann = json.load(_f)
                except Exception:
                    _ann = {}
                _chunks = _ann.get("chunks", []) or []
                if phases is None and _chunks:
                    phases = [str(c.get("phase") or "") for c in _chunks]
                if has_int is None and _chunks:
                    has_int = [bool(c.get("has_interaction")) for c in _chunks]
                if action_class is None and _ann.get("action_class") not in (None, "", "None"):
                    action_class = str(_ann["action_class"])
        return phases, has_int, action_class

    def _load_chunk_actions_from_pt(row_id, num_chunks):
        """Pull per-chunk (keys, mouse) tuples from a training-format .pt.

        Looks under args.camera_pt_dir for `<id>.pt` or `<id>_*.pt`. The .pt's
        ``chunk_actions`` field is a list of {"keys": ..., "mouse": ...} dicts
        (one per chunk in the source video). Returns None when no file is
        found or the field is missing.
        """
        if not args.camera_pt_dir:
            return None
        import glob as _glob
        pt_path = os.path.join(args.camera_pt_dir, f"{row_id}.pt")
        if not os.path.isfile(pt_path):
            _matches = sorted(_glob.glob(os.path.join(args.camera_pt_dir, f"{row_id}_*.pt")))
            pt_path = _matches[0] if _matches else None
        if pt_path is None or not os.path.isfile(pt_path):
            return None
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        ca = blob.get("chunk_actions", None)
        if not isinstance(ca, list) or not ca:
            return None
        out = []
        for c in ca:
            if isinstance(c, dict):
                out.append((str(c.get("keys", "None")), str(c.get("mouse", "·"))))
            elif isinstance(c, (list, tuple)) and len(c) >= 2:
                out.append((str(c[0]), str(c[1])))
            else:
                out.append(("None", "·"))
        return out

    def resolve_row_actions(row, df, num_chunks, action_txt_actions):
        """Resolve per-chunk (keys, mouse) actions for a single CSV row.

        Returns (actions, source) where:
          - actions: list of exactly num_chunks (keys, mouse) tuples
          - source: str label indicating which priority was used

        Resolution order:
          1. camera_pt_dir's <id>.pt -> chunk_actions  (training GT — best match)
          2. chunk_actions_json column  — GT per-chunk actions exported from PT files
          3. --action_txt_path          — fixed sequence shared across all videos
          4. action_keys/action_mouse   — single action repeated to all chunks
          5. prompt_original column     — regex-parsed from original caption
          6. prompt column              — fallback regex parse
        """
        # 1. Per-video chunk_actions from the matching training .pt — when
        # --camera_pt_dir is set we mirror exactly what the training dataloader
        # used (chunk_actions field), so action_embeds_cache lookup at inference
        # gets the same (keys, mouse) → embedding mapping seen during training.
        row_id = str(row["id"])
        pt_actions = _load_chunk_actions_from_pt(row_id, num_chunks)
        if pt_actions is not None:
            return _pad_or_truncate(pt_actions, num_chunks), f"camera_pt_dir({row_id}).chunk_actions"

        # 2. chunk_actions_json — ground truth from PT files (CSV-exported)
        ca_json = row.get("chunk_actions_json", "")
        if isinstance(ca_json, str) and ca_json.strip():
            return _pad_or_truncate(_parse_chunk_actions_json(ca_json), num_chunks), "chunk_actions_json"

        # 3. --action_txt_path — manually specified, shared across all videos
        if action_txt_actions is not None:
            return _pad_or_truncate(list(action_txt_actions), num_chunks), "action_txt_path"

        # 4. CSV action_keys/action_mouse — single action repeated
        if "action_keys" in df.columns and "action_mouse" in df.columns:
            return [(str(row["action_keys"]), str(row["action_mouse"]))] * num_chunks, "csv_action_columns"

        # 5. prompt_original — regex parse from original (movement-containing) caption
        orig = row.get("prompt_original")
        if isinstance(orig, str) and orig.strip():
            return parse_prompt_to_chunk_actions(orig, num_chunks), "prompt_original_regex"

        # 6. prompt — fallback
        p = row.get("refined_prompt") or row.get("prompt", "")
        return parse_prompt_to_chunk_actions(p, num_chunks), "prompt_regex"

    def actions_to_embeds(actions, cache):
        """Convert list of (keys, mouse) tuples → list of action embeddings."""
        if cache is None:
            return None
        default = cache.get(("None", "·"))
        return [cache.get((a[0], a[1]), default) for a in actions]

    # ── Plucker camera trajectory resolution ──
    # Returns (poses_per_chunk, intrinsics_per_chunk, source_label) or
    # (None, None, "inactive") when no source is available. Pre-loaded at
    # the global scope: a per-row resolver runs inside the CSV loops.
    _camera_move_speed = (
        args.camera_move_speed
        if args.camera_move_speed is not None
        else None  # let helper apply DEFAULT_MOVE_SPEED
    )
    _camera_rotate_rad = (
        float(np.deg2rad(args.camera_rotate_speed_deg))
        if args.camera_rotate_speed_deg is not None
        else None
    )

    def _kwargs_for_action_to_camera():
        kw = {}
        if _camera_move_speed is not None:
            kw["move_speed"] = _camera_move_speed
        if _camera_rotate_rad is not None:
            kw["rotate_speed_rad"] = _camera_rotate_rad
        return kw

    # Pre-parse a global action_string once if provided.
    _global_action_string_keys = None
    if args.use_plucker and args.camera_action_string is not None:
        from helios.utils.action_to_camera import action_string_to_frame_keys
        _global_action_string_keys = action_string_to_frame_keys(
            args.camera_action_string)
        if rank == 0:
            print(f"[plucker] camera_action_string expands to "
                  f"{len(_global_action_string_keys)} per-frame key sets")

    def _trajectory_from_global_action_string(num_chunks, frames_per_chunk=33):
        from helios.utils.action_to_camera import (
            DEFAULT_INTRINSICS_384x640, generate_c2ws_from_keys,
        )
        kw = _kwargs_for_action_to_camera()
        # Pad / truncate to exactly num_chunks * frames_per_chunk per-frame keys.
        target = num_chunks * frames_per_chunk
        keys = list(_global_action_string_keys)
        if len(keys) < target:
            keys = keys + [keys[-1]] * (target - len(keys))
        else:
            keys = keys[:target]
        c2ws_full, _, _ = generate_c2ws_from_keys(keys, **kw)
        # Drop the leading identity so we land on T = num_chunks * frames_per_chunk
        c2ws_full = c2ws_full[1:]
        K = torch.tensor(DEFAULT_INTRINSICS_384x640, dtype=torch.float32)
        cp_list, ci_list = [], []
        for k in range(num_chunks):
            a = k * frames_per_chunk
            b = a + frames_per_chunk
            cp_list.append(torch.from_numpy(c2ws_full[a:b]).to(torch.float32))
            ci_list.append(K.clone())
        return cp_list, ci_list

    def _maybe_swap_intrinsics(ci_list, row_id):
        """Optionally replace canned intrinsics with one sampled from training .pt
        files. Per-row deterministic via md5(row_id) hashing. No-op when the
        flag is unset OR when ci_list comes from camera_pt_dir (Mode A — already
        the real Ks). Recovers training FOV distribution for Mode B/C/D.
        """
        if not args.camera_intrinsics_from_pt_dir:
            return ci_list
        from helios.utils.action_to_camera import sample_intrinsics_from_pt_dir
        try:
            sampled_ks = sample_intrinsics_from_pt_dir(
                args.camera_intrinsics_from_pt_dir,
                sample_id=str(row_id),
                seed=args.seed,
            )
        except Exception as _e:
            if rank == 0:
                print(f"[plucker] intrinsics_from_pt_dir sampling failed: {_e}")
            return ci_list
        return [sampled_ks.clone() for _ in ci_list]

    def resolve_camera_trajectory(row_id, row_actions, num_chunks,
                                   frames_per_chunk=33):
        """Build (camera_poses_per_chunk, camera_intrinsics_per_chunk, src).

        Resolution priority (matches docstring of --camera_* flags):
          1. <camera_pt_dir>/<row_id>.pt           (training schema)
          2. <camera_traj_dir>/<row_id>/poses.npy  (per-id npy)
          3. <camera_traj_dir>/<default_name>/...  (shared canned npy)
          4. --camera_action_string                (lingbot DSL)
          5. --camera_from_helios_actions          (rebuild from row_actions)
          6. (None, None, "inactive")              (zero-init identity)

        Modes 2-5 use the canned DEFAULT_INTRINSICS_384x640 by default, but
        when --camera_intrinsics_from_pt_dir is set we swap intrinsics for
        ones sampled from the training latents (per-row deterministic).
        """
        if not args.use_plucker:
            return None, None, "inactive"
        from helios.utils.action_to_camera import (
            build_camera_trajectory_from_actions,
            load_trajectory_from_npy,
            load_trajectory_from_pt,
        )

        # 1. training-format .pt — match either `<id>.pt` or `<id>_*.pt`
        # (training .pt files often carry a size suffix like pickup_0001_145_384_640.pt).
        if args.camera_pt_dir:
            import glob as _glob
            pt_path = os.path.join(args.camera_pt_dir, f"{row_id}.pt")
            if not os.path.isfile(pt_path):
                _matches = sorted(_glob.glob(os.path.join(args.camera_pt_dir, f"{row_id}_*.pt")))
                pt_path = _matches[0] if _matches else None
            if pt_path is not None and os.path.isfile(pt_path):
                cp, ci = load_trajectory_from_pt(
                    pt_path, num_chunks=num_chunks,
                    frames_per_chunk=frames_per_chunk)
                return cp, ci, f"camera_pt_dir({os.path.basename(pt_path)})"

        # 2. per-id npy directory
        if args.camera_traj_dir:
            id_dir = os.path.join(args.camera_traj_dir, str(row_id))
            poses_npy = os.path.join(id_dir, "poses.npy")
            if os.path.isfile(poses_npy):
                ks_npy = os.path.join(id_dir, "intrinsics.npy")
                cp, ci = load_trajectory_from_npy(
                    poses_npy,
                    intrinsics_npy=ks_npy if os.path.isfile(ks_npy) else None,
                    num_chunks=num_chunks,
                    frames_per_chunk=frames_per_chunk,
                )
                ci = _maybe_swap_intrinsics(ci, row_id)
                return cp, ci, f"camera_traj_dir({row_id})"

            # 3. shared canned trajectory
            if args.camera_traj_default_name:
                def_dir = os.path.join(args.camera_traj_dir,
                                       args.camera_traj_default_name)
                poses_npy = os.path.join(def_dir, "poses.npy")
                if os.path.isfile(poses_npy):
                    ks_npy = os.path.join(def_dir, "intrinsics.npy")
                    cp, ci = load_trajectory_from_npy(
                        poses_npy,
                        intrinsics_npy=ks_npy if os.path.isfile(ks_npy) else None,
                        num_chunks=num_chunks,
                        frames_per_chunk=frames_per_chunk,
                    )
                    ci = _maybe_swap_intrinsics(ci, row_id)
                    return cp, ci, f"camera_traj_default({args.camera_traj_default_name})"

        # 4. global action string
        if _global_action_string_keys is not None:
            cp, ci = _trajectory_from_global_action_string(
                num_chunks, frames_per_chunk=frames_per_chunk)
            ci = _maybe_swap_intrinsics(ci, row_id)
            return cp, ci, "camera_action_string"

        # 5. derive from per-chunk Helios actions
        if args.camera_from_helios_actions and row_actions:
            cp, ci = build_camera_trajectory_from_actions(
                row_actions[:num_chunks],
                frames_per_chunk=frames_per_chunk,
                **_kwargs_for_action_to_camera(),
            )
            ci = _maybe_swap_intrinsics(ci, row_id)
            return cp, ci, "helios_actions"

        return None, None, "inactive"

    # Pre-parse action txt if provided
    action_txt_actions = None
    if args.action_txt_path is not None:
        action_txt_actions = parse_action_txt(args.action_txt_path)
        if rank == 0:
            print(f"Loaded {len(action_txt_actions)} per-chunk actions from {args.action_txt_path}")
            for i, (k, m) in enumerate(action_txt_actions):
                print(f"  chunk {i}: keys={k}, mouse={m}")

    # --- Dual-Memory Helios: parse optional chunk_phase_txt and bundle kwargs ---
    chunk_phase_list_cli = None
    if args.chunk_phase_txt_path is not None:
        with open(args.chunk_phase_txt_path) as _f:
            chunk_phase_list_cli = [line.strip() for line in _f if line.strip()]
        if rank == 0:
            print(f"Loaded {len(chunk_phase_list_cli)} per-chunk phases from {args.chunk_phase_txt_path}")

    # Banner: which prompt mode is active for this run.
    if rank == 0:
        if args.prompt_mode == "global":
            print("=" * 70)
            print(f"[prompt_mode=global] Using CSV `prompt` column for all chunks.")
            print(f"  Per-chunk UMT5 embeds (chunk_prompt_embeds in .pt) IGNORED.")
            print(f"  → Paper-fair comparison: all baselines see only global caption.")
            print("=" * 70)
        elif args.prompt_mode == "chunk":
            print("=" * 70)
            print(f"[prompt_mode=chunk] REQUIRE per-vid chunk_prompt_embeds.")
            print(f"  Source: camera_pt_dir = {args.camera_pt_dir or '<unset>'}")
            print(f"          (or chunk_annotations_dir = {args.chunk_annotations_dir or '<unset>'})")
            print(f"          (or --chunk_prompt_pt_path = {args.chunk_prompt_pt_path or '<unset>'})")
            print(f"  Will fail loudly if any video lacks chunk_prompt_embeds.")
            print("=" * 70)
        else:  # auto
            print("=" * 70)
            print(f"[prompt_mode=auto] Per-vid chunk embeds when available, else global.")
            print(f"  Source priority: camera_pt_dir/<vid>.pt > chunk_annotations_dir/<vid>.json descriptions"
                  f" > --chunk_prompt_pt_path > global.")
            print(f"  camera_pt_dir = {args.camera_pt_dir or '<unset>'}")
            print("=" * 70)

    # Phase 5: per-chunk UMT5 embeddings loaded once from a .pt
    chunk_prompt_embeds_list_cli = None
    chunk_descs_cli = None
    if args.chunk_prompt_pt_path is not None:
        _cp_blob = torch.load(args.chunk_prompt_pt_path, map_location="cpu", weights_only=False)
        if isinstance(_cp_blob, dict):
            _cp_descs = _cp_blob.get("chunk_descs", None)
            if isinstance(_cp_descs, list) and _cp_descs:
                chunk_descs_cli = [str(x) for x in _cp_descs]
            _cp_blob = _cp_blob.get("chunk_prompt_embeds", None)
        if isinstance(_cp_blob, list) and _cp_blob:
            chunk_prompt_embeds_list_cli = _cp_blob
            if rank == 0:
                desc_msg = f", {len(chunk_descs_cli)} chunk_descs" if chunk_descs_cli else ""
                print(
                    f"[chunk-prompt] loaded {len(chunk_prompt_embeds_list_cli)} per-chunk "
                    f"UMT5 embeddings{desc_msg} from {args.chunk_prompt_pt_path}"
                )
        else:
            if rank == 0:
                print(f"[chunk-prompt] WARNING: {args.chunk_prompt_pt_path} does not contain a valid "
                      f"chunk_prompt_embeds list; ignoring.")

    action_memory_kwargs = {}
    if (args.eafr_enabled or args.event_memory_enabled
            or args.acha_enabled or args.eafr_learnable):
        action_memory_kwargs = {
            "action_memory_enabled": True,
            "eafr_alpha": args.eafr_alpha,
            "eafr_beta": args.eafr_beta,
            "eafr_tau": args.eafr_tau,
            "eafr_dedup_enabled": args.eafr_dedup_enabled,
            "eafr_dedup_lambda": args.eafr_dedup_lambda,
            "eafr_dedup_metric": args.eafr_dedup_metric,
            # Defaults for unlabelled inputs; per-video labels are filled in
            # from the .pt / chunk_annotations by _load_chunk_labels().
            "chunk_has_interaction_list": None,
            "chunk_phase_list": chunk_phase_list_cli,
            "action_class": None,
        }
    # OIM toggles independently of EAFR/Event Memory; merged into the
    # same kwargs blob so a single ``**action_memory_kwargs`` covers all
    # persistent-memory features at the call site.
    if args.object_memory_enabled:
        action_memory_kwargs["object_memory_enabled"] = True
        action_memory_kwargs["object_memory_keyframes_per_chunk"] = (
            args.object_memory_keyframes_per_chunk
        )
        # OIM also wants chunk_phase / chunk_has_interaction at the
        # per-call level so its writer gate fires correctly.
        action_memory_kwargs.setdefault("chunk_phase_list", chunk_phase_list_cli)
        action_memory_kwargs.setdefault("chunk_has_interaction_list", None)
        action_memory_kwargs.setdefault("action_class", None)
    # chunk_prompt_embeds_list is independent of action_memory flags.
    if chunk_prompt_embeds_list_cli is not None:
        action_memory_kwargs["chunk_prompt_embeds_list"] = chunk_prompt_embeds_list_cli

    # Per-sample metadata log, saved to inference_meta.json for HTML visualization
    chunk_actions_log = {}
    inference_meta = {"params": {}, "samples": {}}

    if args.prompt_txt_path is not None:
        with open(args.prompt_txt_path, "r") as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
        if not args.enable_parallelism:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]
            prompt_list_with_idx = prompt_list_with_idx[rank::world_size]
        else:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]

        for idx, prompt in tqdm(prompt_list_with_idx, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{idx}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        **{_PIPE_PYRAMID_KWARG: args.pyramid_num_inference_steps_list},
                        # training-aligned scheduler (modules backend only;
                        # empty dict on diffusers backend so no-op there).
                        **_PIPE_TRAIN_ALIGN_KWARGS,
                        # stage 3
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        # v2v
                        video=load_video(video_path) if video_path is not None else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue
            if not args.enable_parallelism or rank == 0:
                # prompt_txt_path branch has no per-chunk action context;
                # overlay would have nothing to display. Always plain export.
                export_to_video(output, output_path, fps=24)
    elif args.image_prompt_csv_path is not None:
        df = read_prompt_table(args.image_prompt_csv_path)
        if args.max_samples > 0:
            if args.random_sample and len(df) > args.max_samples:
                # random_state=args.seed → deterministic across trajectories
                df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
            else:
                df = df.head(args.max_samples)

        num_chunks = max(1, args.num_frames // 33)
        for _, r in df.iterrows():
            ra, src = resolve_row_actions(r, df, num_chunks, action_txt_actions)
            sid = str(r["id"])
            chunk_actions_log[sid] = [list(a) for a in ra]
            actual_prompt = str(r.get("refined_prompt") or r["prompt"])
            sample_meta = {
                "prompt": actual_prompt,
                "prompt_original": str(r["prompt_original"]) if "prompt_original" in df.columns and pd.notna(r.get("prompt_original")) else "",
                "action_source": src,
                "chunk_actions": [list(a) for a in ra],
            }
            if args.chunk_annotations_dir is not None:
                ann_path = os.path.join(args.chunk_annotations_dir, f"{sid}.json")
                if os.path.isfile(ann_path):
                    try:
                        with open(ann_path) as _f:
                            _ann = json.load(_f)
                        descs = [str(c.get("description", "")) for c in _ann.get("chunks", [])]
                        sample_meta["chunk_descs"] = descs[:num_chunks]
                    except Exception as _e:
                        if rank == 0:
                            print(f"  [{sid}] failed to load chunk_annotations {ann_path}: {_e}")
            inference_meta["samples"][sid] = sample_meta

        if not args.enable_parallelism:
            df = df.iloc[rank::world_size]

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{row['id']}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            prompt = row.get("refined_prompt") or row["prompt"]
            image_path = (
                os.path.join(args.base_image_prompt_path, str(row["image_name"]))
                if args.base_image_prompt_path and row["image_name"]
                else None
            )

            row_actions, _ = resolve_row_actions(row, df, num_chunks, action_txt_actions)
            row_action_embeds = actions_to_embeds(row_actions, action_embeds_cache)
            cp_list, ci_list, cam_src = resolve_camera_trajectory(
                str(row["id"]), row_actions, num_chunks)

            # Per-video chunk_prompt_embeds resolution, gated by --prompt_mode:
            #   global → never use chunk embeds (always feed CSV `prompt` col)
            #   chunk  → REQUIRE per-vid chunk embeds; fail loudly if missing
            #   auto   → per-vid if available, else cli-pt, else global
            row_chunk_prompt_embeds = None
            cp_src = "global"  # label for logs
            if args.prompt_mode in ("auto", "chunk"):
                _emb, _ = _load_chunk_prompt_embeds_from_pt(str(row["id"]))
                _ann_emb = None if _emb is not None else _encode_chunk_descs_from_annotations(str(row["id"]))
                if _emb is not None:
                    row_chunk_prompt_embeds = _emb
                    cp_src = f"per-vid-pt (n={len(_emb)})"
                elif _ann_emb is not None:
                    row_chunk_prompt_embeds = _ann_emb
                    cp_src = f"chunk_annotations (n={len(_ann_emb)})"
                elif chunk_prompt_embeds_list_cli is not None:
                    row_chunk_prompt_embeds = chunk_prompt_embeds_list_cli
                    cp_src = f"cli-pt (n={len(chunk_prompt_embeds_list_cli)})"
                elif args.prompt_mode == "chunk":
                    raise RuntimeError(
                        f"[{row['id']}] --prompt_mode=chunk but no chunk_prompt_embeds found "
                        f"in camera_pt_dir, chunk_annotations_dir or --chunk_prompt_pt_path; cannot proceed."
                    )
                # auto mode silently falls through to "global"
            row_action_memory_kwargs = dict(action_memory_kwargs)
            if row_chunk_prompt_embeds is not None:
                row_action_memory_kwargs["chunk_prompt_embeds_list"] = row_chunk_prompt_embeds
            else:
                # ensure no stale cli value leaks through when mode=global
                row_action_memory_kwargs.pop("chunk_prompt_embeds_list", None)

            # Per-video memory labels (phase / has_interaction / action_class).
            # --chunk_phase_txt_path, when given, still overrides the phases.
            lbl_src = "none"
            if row_action_memory_kwargs.get("action_memory_enabled") or row_action_memory_kwargs.get(
                "object_memory_enabled"
            ):
                _ph, _hi, _ac = _load_chunk_labels(str(row["id"]))
                if _ph is not None and chunk_phase_list_cli is None:
                    row_action_memory_kwargs["chunk_phase_list"] = _ph
                if _hi is not None:
                    row_action_memory_kwargs["chunk_has_interaction_list"] = _hi
                if _ac is not None:
                    row_action_memory_kwargs["action_class"] = _ac
                lbl_src = (
                    f"phase={'y' if row_action_memory_kwargs.get('chunk_phase_list') else 'n'},"
                    f"has_int={'y' if _hi is not None else 'n'},"
                    f"action_class={_ac if _ac is not None else 'n'}"
                )

            if rank == 0:
                print(
                    f"  [{row['id']}] chunks={num_chunks}, "
                    f"actions={chunk_actions_log.get(str(row['id']))}, "
                    f"plucker_camera={cam_src}, "
                    f"prompt_mode={args.prompt_mode}->{cp_src}, "
                    f"memory_labels=[{lbl_src}]"
                )

            with torch.no_grad():
                # import pdb; pdb.set_trace()
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        **{_PIPE_PYRAMID_KWARG: args.pyramid_num_inference_steps_list},
                        # training-aligned scheduler (modules backend only;
                        # empty dict on diffusers backend so no-op there).
                        **_PIPE_TRAIN_ALIGN_KWARGS,
                        # stage 3
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        # v2v
                        video=load_video(video_path) if video_path is not None else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                        # action conditioning
                        action_embeds_list=row_action_embeds,
                        action_keys_list=[a[0] for a in row_actions] if row_actions else None,
                        action_mouse_list=[a[1] for a in row_actions] if row_actions else None,
                        # plucker camera (idea2)
                        camera_poses_per_chunk=cp_list,
                        camera_intrinsics_per_chunk=ci_list,
                        **row_action_memory_kwargs,
                    ).frames[0]
                except Exception as e:
                    print(f"Error processing {row['id']}: {e}")
                    continue
            if not args.enable_parallelism or rank == 0:
                _export_with_optional_overlay(
                    output, output_path, fps=24,
                    keys_list=[a[0] for a in row_actions] if row_actions else None,
                    mouse_list=[a[1] for a in row_actions] if row_actions else None,
                    frames_per_chunk=33,
                    overlay_enabled=args.overlay_keyboard,
                )
    elif args.interactive_prompt_csv_path is not None:
        df = read_prompt_table(args.interactive_prompt_csv_path)

        df = df.sort_values(by=["id", "prompt_index"])
        all_video_ids = df["id"].unique()
        if args.max_samples > 0:
            if args.random_sample and len(all_video_ids) > args.max_samples:
                import numpy as _np
                _rng = _np.random.RandomState(args.seed)
                _idx = _rng.choice(len(all_video_ids), size=args.max_samples, replace=False)
                _idx.sort()
                all_video_ids = all_video_ids[_idx]
            else:
                all_video_ids = all_video_ids[:args.max_samples]

        num_chunks = max(1, args.num_frames // 33)
        for vid in all_video_ids:
            group = df[df["id"] == vid]
            first_row = group.iloc[0]
            ra, src = resolve_row_actions(first_row, df, num_chunks, action_txt_actions)
            sid = str(vid)
            chunk_actions_log[sid] = [list(a) for a in ra]
            if "refined_prompt" in df.columns:
                prompts = group["refined_prompt"].fillna(group["prompt"]).tolist()
            else:
                prompts = group["prompt"].tolist()
            inference_meta["samples"][sid] = {
                "prompt": prompts[0] if len(prompts) == 1 else prompts,
                "prompt_original": str(first_row["prompt_original"]) if "prompt_original" in df.columns and pd.notna(first_row.get("prompt_original")) else "",
                "action_source": src,
                "chunk_actions": [list(a) for a in ra],
            }

        if not args.enable_parallelism:
            my_video_ids = all_video_ids[rank::world_size]
        else:
            my_video_ids = all_video_ids

        for video_id in tqdm(my_video_ids, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{video_id}.mp4")

            if os.path.exists(output_path):
                print(f"skipping {output_path}!")
                continue

            group_df = df[df["id"] == video_id]

            if "refined_prompt" in df.columns:
                prompt_list = group_df["refined_prompt"].fillna(group_df["prompt"]).tolist()
            else:
                prompt_list = group_df["prompt"].tolist()
            interpolate_time_list = [args.interpolate_time] * len(prompt_list)

            row_actions, _ = resolve_row_actions(group_df.iloc[0], df, num_chunks, action_txt_actions)
            csv_action_embeds = actions_to_embeds(row_actions, action_embeds_cache)
            cp_list, ci_list, _ = resolve_camera_trajectory(
                str(video_id), row_actions, num_chunks)

            # Per-video chunk_prompt_embeds, gated by --prompt_mode (mirrors i2v branch).
            row_chunk_prompt_embeds = None
            cp_src = "global"
            if args.prompt_mode in ("auto", "chunk"):
                _emb, _ = _load_chunk_prompt_embeds_from_pt(str(video_id))
                _ann_emb = None if _emb is not None else _encode_chunk_descs_from_annotations(str(video_id))
                if _emb is not None:
                    row_chunk_prompt_embeds = _emb
                    cp_src = f"per-vid-pt (n={len(_emb)})"
                elif _ann_emb is not None:
                    row_chunk_prompt_embeds = _ann_emb
                    cp_src = f"chunk_annotations (n={len(_ann_emb)})"
                elif chunk_prompt_embeds_list_cli is not None:
                    row_chunk_prompt_embeds = chunk_prompt_embeds_list_cli
                    cp_src = f"cli-pt (n={len(chunk_prompt_embeds_list_cli)})"
                elif args.prompt_mode == "chunk":
                    raise RuntimeError(
                        f"[{video_id}] --prompt_mode=chunk but no chunk_prompt_embeds found "
                        f"in camera_pt_dir, chunk_annotations_dir or --chunk_prompt_pt_path; cannot proceed."
                    )
            row_action_memory_kwargs = dict(action_memory_kwargs)
            if row_chunk_prompt_embeds is not None:
                row_action_memory_kwargs["chunk_prompt_embeds_list"] = row_chunk_prompt_embeds
            else:
                row_action_memory_kwargs.pop("chunk_prompt_embeds_list", None)
            if rank == 0:
                print(f"  [{video_id}] prompt_mode={args.prompt_mode}->{cp_src}")

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt_list,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        **{_PIPE_PYRAMID_KWARG: args.pyramid_num_inference_steps_list},
                        # training-aligned scheduler (modules backend only;
                        # empty dict on diffusers backend so no-op there).
                        **_PIPE_TRAIN_ALIGN_KWARGS,
                        # stage 3
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        # v2v
                        video=load_video(video_path) if video_path is not None else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                        # action conditioning
                        action_embeds_list=csv_action_embeds,
                        action_keys_list=[a[0] for a in row_actions] if row_actions else None,
                        action_mouse_list=[a[1] for a in row_actions] if row_actions else None,
                        # plucker camera (idea2)
                        camera_poses_per_chunk=cp_list,
                        camera_intrinsics_per_chunk=ci_list,
                        **row_action_memory_kwargs,
                    ).frames[0]
                except Exception:
                    continue
            if not args.enable_parallelism or rank == 0:
                _export_with_optional_overlay(
                    output, output_path, fps=24,
                    keys_list=[a[0] for a in row_actions] if row_actions else None,
                    mouse_list=[a[1] for a in row_actions] if row_actions else None,
                    frames_per_chunk=33,
                    overlay_enabled=args.overlay_keyboard,
                )
    else:
        num_chunks = max(1, args.num_frames // 33)
        if action_txt_actions is not None:
            single_actions = _pad_or_truncate(list(action_txt_actions), num_chunks)
            _single_src = "action_txt_path"
        elif args.action_keys is not None and args.action_mouse is not None:
            single_actions = [(args.action_keys, args.action_mouse)] * num_chunks
            _single_src = "cli_action_keys"
        else:
            single_actions = parse_prompt_to_chunk_actions(prompt, num_chunks)
            _single_src = "prompt_regex"
        single_action_embeds = actions_to_embeds(single_actions, action_embeds_cache)
        single_cp, single_ci, single_cam_src = resolve_camera_trajectory(
            "single", single_actions, num_chunks)
        if rank == 0 and single_cam_src != "inactive":
            print(f"[plucker] single-prompt camera source: {single_cam_src}")
        inference_meta["samples"]["single"] = {
            "prompt": prompt,
            "prompt_original": "",
            "action_source": _single_src,
            "chunk_actions": [list(a) for a in single_actions],
        }

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                # stage 1
                history_sizes=[16, 2, 1],
                num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                keep_first_frame=True,
                # stage 2
                is_enable_stage2=args.is_enable_stage2,
                **{_PIPE_PYRAMID_KWARG: args.pyramid_num_inference_steps_list},
                # training-aligned scheduler (modules backend only;
                # empty dict on diffusers backend so no-op there).
                **_PIPE_TRAIN_ALIGN_KWARGS,
                # stage 3
                is_skip_first_chunk=args.is_skip_first_chunk,
                is_amplify_first_chunk=args.is_amplify_first_chunk,
                # cfg zero
                use_zero_init=args.use_zero_init,
                zero_steps=args.zero_steps,
                # i2v
                image=load_image(image_path).resize((args.width, args.height)) if image_path is not None else None,
                image_noise_sigma_min=args.image_noise_sigma_min,
                image_noise_sigma_max=args.image_noise_sigma_max,
                # v2v
                video=load_video(video_path) if video_path is not None else None,
                video_noise_sigma_min=args.video_noise_sigma_min,
                video_noise_sigma_max=args.video_noise_sigma_max,
                # interpolate_prompt
                use_interpolate_prompt=args.use_interpolate_prompt,
                interpolation_steps=args.interpolation_steps,
                interpolate_time_list=interpolate_time_list,
                # action conditioning
                action_embeds_list=single_action_embeds,
                action_keys_list=[a[0] for a in single_actions] if single_actions else None,
                action_mouse_list=[a[1] for a in single_actions] if single_actions else None,
                # plucker camera (idea2)
                camera_poses_per_chunk=single_cp,
                camera_intrinsics_per_chunk=single_ci,
                **action_memory_kwargs,
            ).frames[0]
            # elapsed_time = time.time() - start_time
            # print(f"Inference time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

        if not args.enable_parallelism or rank == 0:
            file_count = len(
                [f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))]
            )
            output_path = os.path.join(
                args.output_folder, f"{file_count:04d}_{args.sample_type}_{int(time.time())}.mp4"
            )
            _export_with_optional_overlay(
                output, output_path, fps=24,
                keys_list=[a[0] for a in single_actions] if single_actions else None,
                mouse_list=[a[1] for a in single_actions] if single_actions else None,
                frames_per_chunk=33,
                overlay_enabled=args.overlay_keyboard,
            )

    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")

    # ── Save chunk actions log & inference metadata (rank 0 only) ──
    chunk_actions_path = None
    if rank == 0 and chunk_actions_log:
        chunk_actions_path = os.path.join(args.output_folder, "chunk_actions.json")
        with open(chunk_actions_path, "w") as f:
            json.dump(chunk_actions_log, f, ensure_ascii=False, indent=1)
        print(f"Saved chunk actions for {len(chunk_actions_log)} samples → {chunk_actions_path}")

    if rank == 0 and inference_meta["samples"]:
        inference_meta["params"] = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "negative_prompt": args.negative_prompt or "",
            "num_latent_frames_per_chunk": args.num_latent_frames_per_chunk,
            "is_enable_stage2": args.is_enable_stage2,
            "pyramid_num_inference_steps_list": args.pyramid_num_inference_steps_list,
            "use_zero_init": args.use_zero_init,
            "zero_steps": args.zero_steps,
            "image_noise_sigma_min": args.image_noise_sigma_min,
            "image_noise_sigma_max": args.image_noise_sigma_max,
            "use_interpolate_prompt": args.use_interpolate_prompt,
            "action_embeds_injected": action_embeds_cache is not None,
        }
        if chunk_descs_cli is not None:
            inference_meta["params"]["chunk_descs"] = chunk_descs_cli
            inference_meta["params"]["chunk_prompt_pt_path"] = args.chunk_prompt_pt_path
        meta_path = os.path.join(args.output_folder, "inference_meta.json")
        with open(meta_path, "w") as f:
            json.dump(inference_meta, f, ensure_ascii=False, indent=1)
        print(f"Saved inference metadata → {meta_path}")

    if rank == 0 and args.image_prompt_csv_path is not None:
        from visualize_results import build_eval_html

        html_dir = os.path.join(args.output_folder, "html")
        html_path = os.path.join(html_dir, "results.html")
        print(f"Generating HTML visualization → {html_path}")
        build_eval_html(
            video_dir=args.output_folder,
            label_path=args.image_prompt_csv_path,
            first_frame_dir=args.base_image_prompt_path,
            output_path=html_path,
            chunk_actions_path=chunk_actions_path,
        )
    elif rank == 0 and args.interactive_prompt_csv_path is not None:
        from visualize_results import build_eval_html

        html_dir = os.path.join(args.output_folder, "html")
        html_path = os.path.join(html_dir, "results.html")
        print(f"Generating HTML visualization → {html_path}")
        build_eval_html(
            video_dir=args.output_folder,
            label_path=args.interactive_prompt_csv_path,
            first_frame_dir=None,
            output_path=html_path,
            chunk_actions_path=chunk_actions_path,
        )


if __name__ == "__main__":
    main()
