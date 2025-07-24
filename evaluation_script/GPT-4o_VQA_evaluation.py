#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import json
import base64
import mimetypes
import openai
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Initialize OpenAI client
client = openai.Client(api_key="") #replace with your key

# Thread-safe counters
lock = threading.Lock()
correct_counter = 0
processed_counter = 0

def ask_gpt4o_question(question: str, options: list, edited_image_path: str, original_image_path: str = None) -> str:
    system_prompt = (
        "You are evaluating an image editing model. Answer the question based only on the **edited image**, "
        "unless the question explicitly asks for a comparison with the original image."
    )

    user_prompt = (
        f"Answer the following multiple-choice question strictly by selecting one option from the list:\n\n"
        f"Question: {question}\n"
        f"Options: {', '.join(options)}\n\n"
        f"Just reply with exactly one of the options above."
    )

    def encode_image(path):
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "image/jpeg"
        with open(path, "rb") as img_file:
            image_bytes = img_file.read()
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            return {"url": f"data:{mime_type};base64,{encoded}", "detail": "auto"}

    content = []

    if original_image_path:
        try:
            content.append({"type": "text", "text": "Original image:"})
            content.append({"type": "image_url", "image_url": encode_image(original_image_path)})
        except Exception as e:
            print(f"Failed to encode original image {original_image_path}: {e}")

    if edited_image_path:
        try:
            content.append({"type": "text", "text": "Edited image:"})
            content.append({"type": "image_url", "image_url": encode_image(edited_image_path)})
        except Exception as e:
            print(f"Failed to encode edited image {edited_image_path}: {e}")

    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def process_entry(index, entry):
    global correct_counter, processed_counter
    question = entry.get("evaluation_question", "")
    options = entry.get("multiple_choice_options", [])
    expected = entry.get("expected_answer", "")
    edited_image_path = entry.get("edited_image_path", None)
    original_image_path = entry.get("image", None)

    if not question or not options or not edited_image_path:
        entry["GPT-4o_answer_to_multiple_choice"] = "INVALID"
        return entry

    gpt_answer = ask_gpt4o_question(question, options, edited_image_path, original_image_path)
    entry["GPT-4o_answer_to_multiple_choice"] = gpt_answer

    with lock:
        processed_counter += 1
        if gpt_answer == expected:
            correct_counter += 1

        if processed_counter % 100 == 0:
            accuracy = correct_counter / processed_counter
            print(f" Processed: {processed_counter}, Correct: {correct_counter}, Accuracy: {accuracy:.2%}")

    return entry

def evaluate_gpt4o_on_json(input_path: str, output_path: str, max_workers: int = 8):
    global correct_counter, processed_counter

    with open(input_path, "r") as f:
        data = json.load(f)

    updated_data = [None] * len(data)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_entry, i, entry): i for i, entry in enumerate(data)}

        for future in tqdm(as_completed(futures), total=len(data), desc="Evaluating GPT-4o answers"):
            idx = futures[future]
            try:
                updated_data[idx] = future.result()
            except Exception as e:
                print(f" Error processing entry {idx}: {e}")
                data[idx]["GPT-4o_answer_to_multiple_choice"] = "ERROR"
                updated_data[idx] = data[idx]

    with open(output_path, "w") as f:
        json.dump(updated_data, f, indent=2)

    final_accuracy = correct_counter / processed_counter if processed_counter > 0 else 0
    print(f"\n Final Accuracy: {final_accuracy:.2%} ({correct_counter}/{processed_counter})")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluation_script/GPT-4o_VQA_evaluation.py <input_json>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_path = input_json.replace(".json", "_with_mc.json")

    evaluate_gpt4o_on_json(input_json, output_path)
