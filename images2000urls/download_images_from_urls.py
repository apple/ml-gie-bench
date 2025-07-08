import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

def download_and_save_images(json_dir, output_base_dir):
    json_dir = Path(json_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for json_file in json_dir.glob("pexels_image_*.json"):
        content_type = json_file.stem.split("_image_")[1].split("100")[0]  # get 'accessories' from 'pexels_image_accessories100'
        save_folder = output_base_dir / content_type
        save_folder.mkdir(parents=True, exist_ok=True)

        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"Downloading images for: {content_type}")
        for entry in tqdm(data, desc=f"Processing {json_file.name}"):
            image_name = entry.get("image_name")
            image_url = entry.get("image_url")
            image_path = save_folder / image_name

            if image_path.exists():
                continue  # Skip if already downloaded

            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                with open(image_path, "wb") as img_file:
                    img_file.write(response.content)
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")

if __name__ == "__main__":
    json_directory = "images2000urls"       # Replace with your path
    output_directory = "/mnt/task_runtime/images2000/"    # Replace with your desired output path
    download_and_save_images(json_directory, output_directory)