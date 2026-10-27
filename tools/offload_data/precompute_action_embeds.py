"""
Pre-compute T5 embeddings for all 81 (keyboard x mouse) action combinations.

Output: action_embeds_cache.pt containing a dict:
    {(keys_label, mouse_label): tensor(1, seq_len, 4096), ...}

Usage:
    python precompute_action_embeds.py \
        --pretrained_model_name_or_path BestWishYsh/Helios-Base \
        --output_path action_embeds_cache.pt
"""

import argparse
import itertools

import torch
from helios.utils.utils_base import encode_prompt
from transformers import AutoTokenizer, UMT5EncoderModel


VOCAB_KEYBOARD = {
    "W": "Person moves forward (W).",
    "A": "Person moves left (A).",
    "S": "Person moves backward (S).",
    "D": "Person moves right (D).",
    "W+A": "Person moves forward and left (W+A).",
    "W+D": "Person moves forward and right (W+D).",
    "S+D": "Person moves backward and right (S+D).",
    "S+A": "Person moves backward and left (S+A).",
    "None": "Person stands still (·).",
}

VOCAB_MOUSE = {
    "→": "Camera turns right (→).",
    "←": "Camera turns left (←).",
    "↑": "Camera tilts up (↑).",
    "↓": "Camera tilts down (↓).",
    "↑→": "Camera tilts up and turns right (↑→).",
    "↑←": "Camera tilts up and turns left (↑←).",
    "↓→": "Camera tilts down and turns right (↓→).",
    "↓←": "Camera tilts down and turns left (↓←).",
    "·": "Camera remains still (·).",
}


def main():
    parser = argparse.ArgumentParser(description="Pre-compute T5 action embeddings cache")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="BestWishYsh/Helios-Base", help="Helios model path"
    )
    parser.add_argument("--output_path", type=str, default="action_embeds_cache.pt", help="Output cache file")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    weight_dtype = torch.bfloat16
    device = args.device

    print("Loading tokenizer and text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    text_encoder.eval().requires_grad_(False).to(device)

    cache = {}
    all_combos = list(itertools.product(VOCAB_KEYBOARD.keys(), VOCAB_MOUSE.keys()))
    print(f"Encoding {len(all_combos)} action combinations...")

    with torch.no_grad():
        for keys_label, mouse_label in all_combos:
            action_text = VOCAB_KEYBOARD[keys_label] + " " + VOCAB_MOUSE[mouse_label]
            prompt_embed, _ = encode_prompt(
                tokenizer=tokenizer, text_encoder=text_encoder, prompt=[action_text], device=device
            )
            cache[(keys_label, mouse_label)] = prompt_embed[0].cpu()  # (seq_len, 4096)
            print(f"  ({keys_label}, {mouse_label}) -> shape {prompt_embed[0].shape}")

    torch.save(cache, args.output_path)
    print(f"Saved {len(cache)} action embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
