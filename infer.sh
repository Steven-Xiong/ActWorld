#!/usr/bin/env bash
# ActWorld inference.
#
#   hf download BestWishYsh/Helios-Base --local-dir checkpoints/Helios-Base
#   hf download xzhexiao/ActWorld --include "merged/*" "action_embeds_cache.pt" --local-dir checkpoints/ActWorld
#   bash infer.sh                         # runs the samples in example/
#   INPUT_DIR=my_inputs bash infer.sh     # your own inputs, same layout as example/
#
# INPUT_DIR layout (see README):
#   prompts.csv                  id, prompt, image_name, chunk_actions_json
#   first_frames/<image_name>
#   annotations/<id>.json        optional: per-chunk description / phase / has_interaction
#   cameras/<id>/poses.npy       optional: camera trajectory (+ intrinsics.npy); otherwise
#                                the trajectory is derived from the keyboard/mouse commands
#
# Weights: ACTWORLD/merged (the full transformer) is used when present;
# otherwise ACTWORLD/lora (LoRA + extras applied on top of Helios-Base).
# Every variable below can be overridden from the environment.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-checkpoints/Helios-Base}"     # VAE, text encoder, scheduler
ACTWORLD="${ACTWORLD:-checkpoints/ActWorld}"            # merged/ or lora/, + action_embeds_cache.pt
INPUT_DIR="${INPUT_DIR:-example}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/actworld}"
NUM_GPUS="${NUM_GPUS:-1}"                               # data-parallel: one sample per GPU
NUM_FRAMES="${NUM_FRAMES:-330}"                         # 33 frames per chunk
MAX_SAMPLES="${MAX_SAMPLES:-0}"                         # 0 = all rows
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [ -d "$ACTWORLD/merged" ]; then
    WEIGHT_FLAGS=(--transformer_path "$ACTWORLD/merged")
    echo "[infer.sh] weights: $ACTWORLD/merged"
elif [ -d "$ACTWORLD/lora" ]; then
    WEIGHT_FLAGS=(--transformer_path "$BASE_MODEL" --lora_path "$ACTWORLD/lora" --partial_path "$ACTWORLD/lora")
    echo "[infer.sh] weights: $BASE_MODEL + LoRA from $ACTWORLD/lora"
else
    echo "[infer.sh] no ActWorld weights in $ACTWORLD (expected merged/ or lora/)" >&2
    exit 1
fi

if [ ! -f "$INPUT_DIR/prompts.csv" ]; then
    echo "[infer.sh] $INPUT_DIR/prompts.csv not found" >&2
    exit 1
fi
# Per-chunk captions and interaction labels, when provided.
INPUT_FLAGS=()
if [ -d "$INPUT_DIR/annotations" ]; then
    INPUT_FLAGS+=(--chunk_annotations_dir "$INPUT_DIR/annotations")
fi
# Camera trajectory for the Plücker branch: cameras/<id>/poses.npy when present,
# otherwise rebuilt from that sample's per-chunk keyboard/mouse commands.
if [ -d "$INPUT_DIR/cameras" ]; then
    INPUT_FLAGS+=(--camera_traj_dir "$INPUT_DIR/cameras")
fi

# The persistent memory bank stays disabled (no --event_memory_enabled /
# --object_memory_enabled) for this checkpoint.
INFER_BACKEND=modules torchrun --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" infer.py \
    --base_model_path "$BASE_MODEL" \
    "${WEIGHT_FLAGS[@]}" \
    --action_embeds_cache "$ACTWORLD/action_embeds_cache.pt" \
    --sample_type i2v \
    --image_prompt_csv_path "$INPUT_DIR/prompts.csv" \
    --base_image_prompt_path "$INPUT_DIR/first_frames" \
    --output_folder "$OUTPUT_DIR" \
    --prompt_mode auto \
    --max_samples "$MAX_SAMPLES" \
    --seed "$SEED" \
    --num_frames "$NUM_FRAMES" --height 384 --width 640 --fps 24 \
    --num_inference_steps 50 --guidance_scale 5.0 \
    --use_plucker --plucker_vae_space_down 8 --camera_from_helios_actions \
    --eafr_enabled --eafr_dedup_enabled \
    --is_amplify_history --acha_enabled \
    ${INPUT_FLAGS[@]+"${INPUT_FLAGS[@]}"}
