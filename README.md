# [GIE-Bench](https://arxiv.org/pdf/2505.11493)

<font size=7><div align='center' > [[📖 Paper](https://arxiv.org/pdf/2505.11493)]  </div></font>

*GIE‑Bench* ( **G**rounded Evaluation for Text-Guided **I**mage **E**diting) is a curated dataset for assessing text‑guided image‑editing models along two complementary axes:

| Axis                          | Metric(s)                                | What it measures                                                |
| ----------------------------- | ---------------------------------------- | -------------------------------------------------------------- |
| **Functional Correctness**    | Multiple‑choice QA via GPT‑4o            | Did the edit satisfy the instruction?                          |
| **Content Preservation**      | CLIP‑Sim, SSIM, MSE, PSNR (masked)       | How well are unedited regions preserved?                       |


---

## 📂 Repository Layout
```
GIE‑Bench/
├── images2000urls/             # URL list + download helper
│   └── download_images_from_urls.py
├── evaluation_script/         # Automated evaluation
│   ├── GPT‑4o_VQA_evaluation.py
│   ├── masked_clip_ssim_evaluation.py
│   ├── masked_mse_evaluation.py
│   └── masked_psnr_evaluation.py
├── gie_bench_json.zip          # Zipped benchmark file 
└── README.md                   # You’re here
```

---


## 📥 Downloading the Benchmark

1. **Raw images**

   ```bash
   python images2000urls/download_images_from_urls.py 
   ```

2. **Benchmark JSON**

   ```bash
   unzip gie_bench_json.zip      # produces gie_bench.json
   ```

---

## 🚀 Running Your Model on GIE‑Bench

1. **Inference**

   - Load `gie_bench.json`.
   - For each entry, generate an edited image for input image `image`, following edit instruction `edit_instruction`.
   - Save the edited image **locally** and write the file path back to the same entry under the key `edited_image_path`.

   ```python
   entry["edited_image_path"] = f"outputs/{entry_id}.png"
   ```

2. **Save the modified benchmark**

   ```python
   with open("results/my_model_output.json", "w") as f:
       json.dump(data, f, indent=2)
   ```

---

## 🧪 Evaluation

### 1. Functional Correctness (GPT‑4o)

```bash
python evaluation_script/GPT-4o_VQA_evaluation.py #for all evaluation code, you will need to modify outout file path to yours
```

### 2. Content Preservation

```bash
# CLIP + SSIM (masked)
python evaluation_script/masked_clip_ssim_evaluation.py path/to/your_model_output.json

# MSE (masked)
python evaluation_script/masked_mse_evaluation.py path/to/your_model_output.json

# PSNR (masked)
python evaluation_script/masked_psnr_evaluation.py path/to/your_model_output.json

# CLIP (unmasked)
python evaluation_script/clip_whole_image_evaluation.py path/to/your_model_output.json
```

Each script appends score fields to a new JSON, preserving your original file.

---

## Citation
```
@article{qian2025gie,
  title={GIE-Bench: Towards Grounded Evaluation for Text-Guided Image Editing},
  author={Qian, Yusu and Lu, Jiasen and Fu, Tsu-Jui and Wang, Xinze and Chen, Chen and Yang, Yinfei and Hu, Wenze and Gan, Zhe},
  journal={arXiv preprint arXiv:2505.11493},
  year={2025}
}
```

## 📄 License

This project is distributed under the [LICENSE](LICENSE). All data is released under the [CC-by-NC-ND](LICENSE_DATA).

---

*Happy editing and benchmarking!* 🎨
