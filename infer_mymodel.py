import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel
from torchvision import transforms as T
from tqdm import tqdm

from hwd.datasets.shtg import (
    IAMWords,
    IAMLines,
    CVLLines,
    RimesLines,
    KaraokeLines
)

# ====================================
# 0. Load model (GPU only)
# ====================================
MODEL_PATH = "./emuru_result/head_t5_large_2e-5_ech3"

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
# 2. Core generation function
# ====================================
def run_generation(dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset.save_transcriptions(out_dir)
    print(f"Saving images to: {out_dir}")

    for sample in tqdm(dataset, desc=f"Generating {out_dir.name}"):

        gen_text = sample["gen_text"]
        style_pil = sample["style_imgs"][0]
        style_tensor = preprocess(style_pil)

        dst_path = out_dir / sample["dst_path"]
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            out_img = model.generate(
                gen_text=gen_text,
                style_img=style_tensor,
                max_new_tokens=256
            )

        try:
            out_img.save(dst_path)
        except Exception as e:
            print(f"[WARN] Failed saving {dst_path}: {e}")

    print(f"\n✔ DONE generating {out_dir.name} → {out_dir}\n")


# ====================================
# 3. Helper for standard Lines datasets
# ====================================
def generate_lines_dataset(dataset_cls, name):
    print("\n==============================================")
    print(f"Generating {name} ...")
    dataset = dataset_cls(num_style_samples=1, load_gen_sample=False)
    print(f"{name} size = {len(dataset)} samples")
    run_generation(dataset, f"{name}_my_model")

# ====================================
# 6. Generate KaraokeLines (handwritten)
# ====================================
print("\n==============================================")
print("Generating KaraokeLines (handwritten) ...")
dataset_karaoke_hand = KaraokeLines(
    flavor="handwritten",
    num_style_samples=1,
    load_gen_sample=False
)
print(f"KaraokeLines (handwritten) size = {len(dataset_karaoke_hand)} samples")
run_generation(
    dataset_karaoke_hand,
    "KaraokeLines_handwritten_my_model"
)


# ====================================
# 7. Generate KaraokeLines (typewritten)
# ====================================
print("\n==============================================")
print("Generating KaraokeLines (typewritten) ...")
dataset_karaoke_type = KaraokeLines(
    flavor="typewritten",
    num_style_samples=1,
    load_gen_sample=False
)
print(f"KaraokeLines (typewritten) size = {len(dataset_karaoke_type)} samples")
run_generation(
    dataset_karaoke_type,
    "KaraokeLines_typewritten_my_model"
)

generate_lines_dataset(RimesLines, "RimesLines")


# ====================================
# 4. Generate IAMWords
# ====================================
print("\n==============================================")
print("Generating IAMWords ...")
dataset_words = IAMWords(num_style_samples=1, load_gen_sample=False)
print(f"IAMWords size = {len(dataset_words)} samples")
run_generation(dataset_words, "IAMWords_my_model")


# ====================================
# 5. Generate Lines datasets
# ====================================
generate_lines_dataset(IAMLines, "IAMLines")
generate_lines_dataset(CVLLines, "CVLLines")

print("\n🎉 All generation complete!")