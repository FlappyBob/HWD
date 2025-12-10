import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel
from hwd.datasets.shtg import IAMWords, IAMLines
from torchvision import transforms as T
from tqdm import tqdm
import os


# ====================================
# 0. Load model (GPU only)
# ====================================
MODEL_PATH = "./emuru_result/emuru_t5_small_2e-5_ech5"

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).cuda().eval()

print("Model loaded on CUDA.")


# ====================================
# 1. Preprocessing for style images
# ====================================
transforms = T.Compose([
    lambda img: img.resize((int(64 * (img.width / img.height)), 64), Image.LANCZOS),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

def preprocess(pil_img):
    return transforms(pil_img.convert("RGB")).unsqueeze(0).cuda()



# ====================================
# helper: safe image saving
# ====================================
def safe_save_image(img, save_path):
    """
    Save PIL image safely:
    - Fix bad tile table using copy()
    - Fix zero-size images by resizing to 64x64
    - Convert to RGB on fallback
    """
    try:
        img2 = img.copy()

        w, h = img2.size
        if w <= 1 or h <= 1:
            print(f"[WARN] Bad image size {w}x{h}, resizing -> 64x64")
            img2 = img2.resize((64, 64), Image.BILINEAR)

        img2.save(save_path)
        return True

    except Exception as e:
        print(f"[ERROR] PIL save failed: {e}")
        print("Retrying with RGB conversion...")

        try:
            img3 = img.convert("RGB").copy()
            img3.save(save_path)
            return True
        except Exception as e2:
            print(f"[FATAL] Could not save even after RGB conversion: {e2}")
            return False



# ====================================
# FUNCTION: run one dataset (words or lines)
# ====================================
def run_dataset(dataset, out_dir_name, label="IAMWords"):

    OUT_DIR = Path(out_dir_name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FAILED_LOG = OUT_DIR / "failed_samples.txt"

    print(f"\n===================================")
    print(f"Loading {label} dataset...")

    print(f"{label} size: {len(dataset)} samples")

    # save transcriptions.json
    dataset.save_transcriptions(OUT_DIR)
    print(f"Saving images to: {OUT_DIR}")

    # iterate
    for sample in tqdm(dataset, desc=f"Generating {label}"):

        dst_path = OUT_DIR / sample["dst_path"]

        # skip if exists (support restart)
        if dst_path.exists():
            continue

        gen_text = sample["gen_text"]
        style_pil = sample["style_imgs"][0]
        style_imgs_text = sample["style_imgs_text"][0]

        try:
            style_tensor = preprocess(style_pil)

            with torch.inference_mode():
                out_img = model.generate(
                    style_text=style_imgs_text,
                    gen_text=gen_text,
                    style_img=style_tensor,
                    max_new_tokens=256
                )

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            ok = safe_save_image(out_img, dst_path)
            if not ok:
                with FAILED_LOG.open("a") as f:
                    f.write(f"FAILED_SAVE: {sample['dst_path']}\n")

        except Exception as e:
            print(f"[ERROR] Model failed at {sample['dst_path']}: {e}")
            with FAILED_LOG.open("a") as f:
                f.write(f"FAILED_MODEL: {sample['dst_path']}\n")
            continue

    print(f"\n✔ DONE: {label} → {OUT_DIR}\n")
    print(f"Failed samples logged at {FAILED_LOG}\n")



# ====================================
# 2. Run IAMWords
# ====================================
dataset_words = IAMWords(num_style_samples=1, load_gen_sample=False)
run_dataset(dataset_words, "small_emuru_words", label="IAMWords")


# ====================================
# 3. Run IAMLines
# ====================================
dataset_lines = IAMLines(num_style_samples=1, load_gen_sample=False)
run_dataset(dataset_lines, "small_emuru_lines", label="IAMLines")


print("\n🎉 ALL DONE! Words + Lines generation complete.\n")
