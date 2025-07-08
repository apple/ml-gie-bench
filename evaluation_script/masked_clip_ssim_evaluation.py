#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import sys
import json
import clip
import torch
import numpy as np
import torchvision.transforms as T
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from PIL import Image
from tabulate import tabulate
from einops import rearrange

model, preprocess = clip.load("ViT-B/32", device="cpu")


def apply_mask(spatial_tokens, mask, patch_size):
    # Convert mask to torch tensor, same dtype and device as spatial_tokens
    mask_tensor = torch.from_numpy(mask).float().to(spatial_tokens.device)
    if mask_tensor.shape != (224, 224):
        raise ValueError(f"Expected mask shape (224, 224), but got {mask_tensor.shape}")

    # Rearrange to match patch layout
    patch_mask = rearrange(mask_tensor, '(h p1) (w p2) -> h w p1 p2', p1=patch_size, p2=patch_size)
    patch_mask = patch_mask.any(dim=(-1, -2))  # shape: [H', W']
    patch_mask = patch_mask.reshape(1, -1, 1).float()  # shape: [1, H*W, 1]

    x = spatial_tokens * patch_mask  # [B, H*W, C]
    summed = x.sum(dim=1)  # [B, C]
    norm = patch_mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    x = summed / norm  # [B, C]
    return x



def get_visual_output(image, object_mask, rest_mask, patch_size):
    x = model.visual.conv1(image)  # shape = [*, width, grid, grid]
        
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                device=x.device), x],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD [1, 50, 768]

    # apply mask
    x = x[:, 1:, :]  # [B, H*W, C]

    # Masked average pooling over spatial tokens
    object_x = apply_mask(x, object_mask, patch_size)
    rest_x = apply_mask(x, rest_mask, patch_size)
    object_x = model.visual.ln_post(object_x)
    rest_x = model.visual.ln_post(rest_x)

    if model.visual.proj is not None:
        object_x = object_x @ model.visual.proj
        rest_x = rest_x @ model.visual.proj
    return object_x, rest_x


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224), Image.LANCZOS)
    image = np.array(image)
    image = Image.fromarray(image)
    return image


def generate_image_embedding(image_info, image_key):
    image = load_image(image_info[image_key])
    image = preprocess(image).unsqueeze(0).to("cpu")
    object_mask = np.array(image_info["object_mask"])
    rest_mask = np.ones(object_mask.shape, dtype=object_mask.dtype) - object_mask
    object_emb, rest_emb = get_visual_output(image, object_mask, rest_mask, patch_size=32)
    emb_data = image_info.copy()
    emb_data[image_key + "_object_emb"] = object_emb.detach().cpu().numpy().tolist()
    emb_data[image_key + "_rest_emb"] = rest_emb.detach().cpu().numpy().tolist()
    return emb_data



def calculate_clip_similarity(image_info):
    def normalize(x):
        return x / np.linalg.norm(x)

    object_emb = np.squeeze(normalize(image_info["image_object_emb"]))
    rest_emb = np.squeeze(normalize(image_info["image_rest_emb"]))
    edited_object_emb = np.squeeze(normalize(image_info["edit_image_object_emb"]))
    edited_rest_emb = np.squeeze(normalize(image_info["edit_image_rest_emb"]))

    object_sim = np.dot(object_emb, edited_object_emb)
    rest_sim = np.dot(rest_emb, edited_rest_emb)

    sim_data = image_info.copy()
    sim_data["object_clip_sim"] = object_sim
    sim_data["rest_clip_sim"] = rest_sim
    return sim_data


def calculate_ssim(image_info):
    def normalize_image(x):
        return np.array(x, dtype=np.float32) / 255.0

    image = load_image(image_info["image"])
    edit_image = load_image(image_info["edit_image"])

    object_mask = np.array(image_info["object_mask"])
    rest_mask = np.ones(object_mask.shape, dtype=object_mask.dtype) - object_mask
    # convert to grayscale
    image = T.Grayscale()(image)
    edit_image = T.Grayscale()(edit_image)
    # normalize
    image = normalize_image(image)
    edit_image = normalize_image(edit_image)
    # apply mask
    image = image * rest_mask
    edit_image = edit_image * rest_mask
    # compute ssim
    sim, _ = ssim(image, edit_image, data_range=1.0, full=True)
    ssim_data = image_info.copy()
    ssim_data["ssim_similarity_outside_mask"] = sim
    return ssim_data


def output_clip_ssim_scores(input_filename, output_filename):
    image_info_dict = {}

    # extract raw image filename, edited image filename, object mask and rest mask
    with open(input_filename, "r") as file:
        data = json.load(file)
        for d in data:
            image_filename = d['image']
            image_path = d['image']
            edit_image_path = d['edited_image_path']

            if not os.path.exists(image_path) or not os.path.exists(edit_image_path):
                print(f"Skip image: {image_path}, edit_image: {edit_image_path}")
                continue
            image_info_dict[d["edit_instruction"] + image_filename] = {
                "edit_instruction": d["edit_instruction"],
                "image": image_path,
                "edit_image": edit_image_path,
                "object_mask": d["object_mask"],
            }

    # extract clip embedding of object area and rest area
    clip_embeddings = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # extract clip embedding of raw image
        futures = {executor.submit(generate_image_embedding, image_info_dict[k], "image") for k in image_info_dict}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            emb_data = future.result()
            clip_embeddings[emb_data["edit_instruction"] + emb_data["image"]] = emb_data
        # extract clip embedding of edited image
        futures = {executor.submit(generate_image_embedding, clip_embeddings[k], "edit_image") for k in clip_embeddings}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            emb_data = future.result()
            clip_embeddings[emb_data["edit_instruction"] + emb_data["image"]] = emb_data

    clip_sims = {}
    # calculate clip similarity
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(calculate_clip_similarity, clip_embeddings[k]) for k in clip_embeddings}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            clip_sim_data = future.result()
            clip_sims[clip_sim_data["edit_instruction"] + clip_sim_data["image"]] = clip_sim_data

    # calculate ssim similarity
    ssim_scores = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(calculate_ssim, clip_sims[k]) for k in clip_sims}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            ssim_data = future.result()
            ssim_scores[ssim_data["edit_instruction"] + ssim_data["image"]] = ssim_data

    # generate output content
    output_content = []
    with open(input_filename, "r") as file:
        data = json.load(file)
        for d in data:
            image_filename = d['image']
            image_path = d['image']
            if d["edit_instruction"] + image_path not in ssim_scores:
                continue
            image_data = ssim_scores[d["edit_instruction"] + image_path]
            d["object_clip_sim"] = image_data["object_clip_sim"]
            d["rest_clip_sim"] = image_data["rest_clip_sim"]
            d["ssim_similarity_outside_mask"] = image_data["ssim_similarity_outside_mask"]
            output_content.append(d)

    # compute and print summary table
    def summarize_scores(entries):
        def safe_mean(key):
            values = [d[key] for d in entries if isinstance(d.get(key), (float, int))]
            return np.mean(values) if values else None

        summary = [
            ["object_clip_sim", f"{safe_mean('object_clip_sim'):.4f}" if safe_mean('object_clip_sim') is not None else "N/A"],
            ["rest_clip_sim", f"{safe_mean('rest_clip_sim'):.4f}" if safe_mean('rest_clip_sim') is not None else "N/A"],
            ["ssim_similarity_outside_mask", f"{safe_mean('ssim_similarity_outside_mask'):.4f}" if safe_mean('ssim_similarity_outside_mask') is not None else "N/A"]
        ]

        print("\nEvaluation Summary:")
        print(tabulate(summary, headers=["Metric", "Mean Value"], tablefmt="grid"))

    summarize_scores(output_content)


    with open(output_filename, "w") as fout:
        json.dump(output_content, fout, indent=2)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python evaluation_script/masked_clip_ssim_evaluation.py <input_json>")
        sys.exit(1)

    input_filename = sys.argv[1]
    if not input_filename.endswith(".json") or not os.path.exists(input_filename):
        print(f"Invalid input: {input_filename}")
        sys.exit(1)

    output_filename = input_filename.replace(".json", "_with_masked_clip_ssim.json")
    output_clip_ssim_scores(input_filename, output_filename)

