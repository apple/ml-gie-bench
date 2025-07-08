#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

def align_images(original, edited):
    gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(gray1, None)
    k2, d2 = sift.detectAndCompute(gray2, None)
    if d1 is None or d2 is None:
        return None
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 4:
        return None
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)
    if matrix is None:
        return None
    h, w = original.shape[:2]
    return cv2.warpAffine(edited, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def compute_psnr_outside_mask(original, aligned, mask_array):
    if aligned is None or mask_array is None:
        return None
    if aligned.shape != original.shape:
        aligned = cv2.resize(aligned, (original.shape[1], original.shape[0]))
    mask = np.array(mask_array, dtype=np.uint8)
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    inv_mask = (mask < 128).astype(np.uint8)
    diff = (original.astype(np.float32) - aligned.astype(np.float32)) ** 2
    mse = np.sum(diff * inv_mask[..., None]) / (np.sum(inv_mask) * 3 + 1e-10)
    return float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)

def compute_scores_for_json(json_path, save_path=None):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[{json_path}] Corrupt JSON. Trying line-by-line recovery...")
        with open(json_path, 'r') as f:
            raw = f.read().strip()
            if raw.startswith('[') and raw.endswith(']'):
                raw = raw[1:-1]
            entries = raw.split("},")
            data = []
            for i, line in enumerate(entries):
                try:
                    if not line.endswith('}'):
                        line += '}'
                    data.append(json.loads(line))
                except:
                    print(f"Skipped corrupted entry {i}")

    for entry in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):
        original_path = entry.get("image")
        edited_path = entry.get("edited_image_path")
        mask_array = entry.get("object_mask")
        if not (original_path and edited_path and mask_array) or not all(os.path.exists(p) for p in [original_path, edited_path]):
            entry["aligned_psnr"] = None
            continue
        try:
            original = cv2.imread(original_path)
            edited = cv2.imread(edited_path)
            aligned = align_images(original, edited)
            entry["aligned_psnr"] = compute_psnr_outside_mask(original, aligned, mask_array)
        except Exception as e:
            print(f"Error processing entry: {e}")
            entry["aligned_psnr"] = None

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    return data

def summarize_aligned_psnr(json_data):
    scores = [entry.get("aligned_psnr") for entry in json_data if isinstance(entry.get("aligned_psnr"), (int, float))]
    if not scores:
        return None
    scores = np.array(scores)
    return {
        "count": len(scores),
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluation_script/masked_psnr_evaluation.py path/to/json_file.json")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    output_path = input_path.replace(".json", "_with_aligned_psnr.json")
    scored_data = compute_scores_for_json(input_path, save_path=output_path)
    stats = summarize_aligned_psnr(scored_data)

    if stats:
        print("\nAligned PSNR Summary:")
        print(tabulate([[
            os.path.basename(input_path),
            stats["count"],
            f"{stats['mean']:.2f}",
            f"{stats['std']:.2f}",
            f"{stats['min']:.2f}",
            f"{stats['max']:.2f}",
        ]], headers=["File", "Count", "Mean", "Std Dev", "Min", "Max"], tablefmt="grid"))
    else:
        print("No valid scores computed.")
