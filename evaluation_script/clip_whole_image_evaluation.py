#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import sys
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_similarity(img_path1, img_path2):
    try:
        image1 = preprocess(Image.open(img_path1).convert("RGB")).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(img_path2).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Failed to load images: {img_path1}, {img_path2} — {e}")
        return None

    with torch.no_grad():
        feat1 = model.encode_image(image1)
        feat2 = model.encode_image(image2)
        feat1 /= feat1.norm(dim=-1, keepdim=True)
        feat2 /= feat2.norm(dim=-1, keepdim=True)
        return (feat1 @ feat2.T).item()

def process_json(input_path):
    try:
        with open(input_path, "r") as f:
            entries = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file {input_path}: {e}")
        return

    scores = []

    for entry in tqdm(entries, desc="Computing CLIP scores"):
        original_path = entry.get("image")
        edited_path = entry.get("edited_image_path")

        if not original_path or not edited_path or not os.path.exists(original_path) or not os.path.exists(edited_path):
            print(f"Missing or invalid image paths for entry: {entry.get('edit_instruction', '')}")
            entry["clip_score_whole_image"] = None
            continue

        score = compute_clip_similarity(original_path, edited_path)
        if score is not None:
            entry["clip_score_whole_image"] = score
            scores.append(score)
        else:
            entry["clip_score_whole_image"] = None

    output_path = input_path.replace(".json", "_with_clip_whole_image.json")
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"\nDone. Output written to: {output_path}")
    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"Mean CLIP Score (whole image): {mean_score:.4f}")
    else:
        print("No valid CLIP scores computed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluation_script/clip_whole_image_evaluation.py <input_json>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not input_file.endswith(".json") or not os.path.isfile(input_file):
        print(f"Invalid input file: {input_file}")
        sys.exit(1)

    process_json(input_file)
