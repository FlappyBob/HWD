import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel
from torchvision import transforms as T
from torch.utils.data import Dataset
import json
from torchvision.transforms.functional import to_pil_image

from hwd.datasets.shtg import IAMWords


# ====================================
# Failed sample dump
# ====================================
FAILED_DIR = Path("failed_samples")
FAILED_DIR.mkdir(exist_ok=True)

def dump_failed_sample(sample, index, error_msg):
    sample_dir = FAILED_DIR / f"sample_{index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # gen_text
    gen_text = sample.get("gen_text", "")
    (sample_dir / "gen_text.txt").write_text(
        gen_text if gen_text is not None else "<None>",
        encoding="utf-8"
    )

    # style_text
    style_texts = sample.get("style_imgs_text", [])
    style_text = style_texts[0] if style_texts else ""
    (sample_dir / "style_text.txt").write_text(
        style_text if style_text is not None else "",
        encoding="utf-8"
    )

    # style_img
    style_imgs = sample.get("style_imgs", [])
    if style_imgs:
        img_tensor = style_imgs[0].cpu().clamp(-1, 1)
        img_tensor = (img_tensor + 1) / 2
        to_pil_image(img_tensor).save(sample_dir / "style_img.png")

    # meta
    meta = {
        "index": index,
        "dst_path": sample.get("dst_path"),
        "error": error_msg
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ====================================
# Dataset wrapper（保持你原来的）
# ====================================
class SHTGWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transforms = T.Compose([
            self._to_height_64,
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def _to_height_64(self, img):
        w, h = img.size
        return img.resize((int(64 * w / h), 64), Image.LANCZOS)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["style_imgs"] = [
            self.transforms(img.convert("RGB"))
            for img in sample["style_imgs"]
        ]
        return sample


# ====================================
# Load model（你的 Emuru）
# ====================================
MODEL_PATH = "./emuru_result/emuru_t5_small_2e-5_ech5"

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).cuda().eval()


# ====================================
# Load dataset
# ====================================
dataset = IAMWords(num_style_samples=1, load_gen_sample=False)
dataset = SHTGWrapper(dataset)

output_dir = Path("output_samples")
output_dir.mkdir(parents=True, exist_ok=True)

dataset.dataset.save_transcriptions(output_dir)


# ====================================
# Inference loop（稳定版）
# ====================================
with torch.inference_mode():
    for idx, sample in enumerate(dataset):

        try:
            gen_text = sample["gen_text"] or ""
            style_texts = sample.get("style_imgs_text", [])
            style_text = style_texts[0] if style_texts else ""

            style_img = sample["style_imgs"][0].unsqueeze(0).cuda()

            out_img = model.generate(
                style_text=style_text,   # 必须是 str
                gen_text=gen_text,       # 必须是 str
                style_img=style_img,
                max_new_tokens=256
            )

            dst_path = output_dir / sample["dst_path"]
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            tmp = dst_path.with_suffix(".tmp.png")
            out_img.save(tmp)
            tmp.rename(dst_path)

        except Exception as e:
            print(f"❌ Failed at sample {idx}: {e}")
            dump_failed_sample(sample, idx, str(e))
