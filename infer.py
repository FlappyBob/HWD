import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel
from hwd.datasets.shtg import IAMWords, IAMLines
from torchvision import transforms as T
from tqdm import tqdm


# ====================================
# 0. Load model (GPU only)
# ====================================
MODEL_PATH = "./emuru_result/head_t5_small_2e-5_ech5"

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
# FUNCTION: Generate all samples for a given dataset
# ====================================
def run_generation(dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save transcriptions.json
    dataset.save_transcriptions(out_dir)
    print(f"Saving images to: {out_dir}")

    for sample in tqdm(dataset, desc=f"Generating {out_dir.name}"):

        gen_text = sample["gen_text"]
        style_pil = sample["style_imgs"][0]     # 1 style image
        style_tensor = preprocess(style_pil)

        dst_path = out_dir / sample["dst_path"]
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            out_img = model.generate(
                gen_text=gen_text,
                style_img=style_tensor,
                max_new_tokens=256
            )

        out_img.save(dst_path)

    print(f"\n✔ DONE generating {out_dir.name} → {out_dir}\n")



# ====================================
# 2. Generate IAMWords
# ====================================
print("\n==============================================")
print("Generating IAMWords ...")
dataset_words = IAMWords(num_style_samples=1, load_gen_sample=False)
print(f"IAMWords size = {len(dataset_words)} samples")

run_generation(dataset_words, "IAMWords_my_model")



# ====================================
# 3. Generate IAMLines
# ====================================
print("\n==============================================")
print("Generating IAMLines ...")
dataset_lines = IAMLines(num_style_samples=1, load_gen_sample=False)
print(f"IAMLines size = {len(dataset_lines)} samples")

run_generation(dataset_lines, "IAMLines_my_model")
print("\n🎉 All generation complete!")