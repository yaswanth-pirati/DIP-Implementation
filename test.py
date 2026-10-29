import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor, ToPILImage
from model.base import CLCC

# -----------------------
# CONFIG
# -----------------------
INPUT_DIR = "./dataset/valid/input"
TARGET_DIR = "./dataset/valid/target"

COMPARE_DIR = "./output_compare"
RAW_OUTPUT_DIR = "./output"

BASE_CKPT = "./log/IACC/improved_research/ckpt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(COMPARE_DIR, exist_ok=True)
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)

# -----------------------
# SELECT MODEL
# -----------------------
if os.path.exists(os.path.join(BASE_CKPT, "best_psnr.pth")):
    MODEL_PATH = os.path.join(BASE_CKPT, "best_psnr.pth")
elif os.path.exists(os.path.join(BASE_CKPT, "best_ssim.pth")):
    MODEL_PATH = os.path.join(BASE_CKPT, "best_ssim.pth")
else:
    MODEL_PATH = os.path.join(BASE_CKPT, "last.pth")

print(f"Using model: {MODEL_PATH}")

# -----------------------
# LOAD MODEL
# -----------------------
model = CLCC(64, 3, 3).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

print("Model loaded successfully\n")

# -----------------------
# TRANSFORMS
# -----------------------
to_tensor = ToTensor()
to_pil = ToPILImage()

# -----------------------
# PROCESS IMAGES
# -----------------------
for name in os.listdir(INPUT_DIR):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    input_path = os.path.join(INPUT_DIR, name)
    target_path = os.path.join(TARGET_DIR, name)

    # load input
    inp_img = Image.open(input_path).convert("RGB")
    inp_tensor = to_tensor(inp_img).unsqueeze(0).to(DEVICE)

    # inference
    with torch.no_grad():
        outputs = model(inp_tensor)
        out = outputs[2].clamp(0, 1)

    out_img = to_pil(out.squeeze(0).cpu())

    # -----------------------
    # SAVE RAW OUTPUT
    # -----------------------
    out_img.save(os.path.join(RAW_OUTPUT_DIR, name))

    # -----------------------
    # CREATE COMPARISON IMAGE
    # -----------------------
    # -----------------------
    # FINAL PROFESSIONAL COMPARISON LAYOUT
    # -----------------------
    if os.path.exists(target_path):
        tgt_img = Image.open(target_path).convert("RGB")

        # resize
        inp_res = inp_img.resize((256, 256))
        out_res = out_img.resize((256, 256))
        tgt_res = tgt_img.resize((256, 256))

        # layout tuning
        spacing = 30
        top_pad = 30
        side_pad = 30
        bottom_pad = 30
        label_gap = 15
        label_height = 30

        total_width = side_pad * 2 + (256 * 3) + (spacing * 2)
        total_height = top_pad + 256 + label_gap + label_height + bottom_pad

        comparison = Image.new("RGB", (total_width, total_height), (255, 255, 255))

        # paste images (NOW centered with top padding)
        y_img = top_pad
        comparison.paste(inp_res, (side_pad, y_img))
        comparison.paste(out_res, (side_pad + 256 + spacing, y_img))
        comparison.paste(tgt_res, (side_pad + 2 * (256 + spacing), y_img))

        # draw labels
        draw = ImageDraw.Draw(comparison)

        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except:
            font = ImageFont.load_default()

        labels = ["Input", "Output", "Target"]

        centers = [
            side_pad + 128,
            side_pad + 256 + spacing + 128,
            side_pad + 2 * (256 + spacing) + 128,
        ]

        y_text = y_img + 256 + label_gap

        for text, cx in zip(labels, centers):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            draw.text((cx - w // 2, y_text), text, fill=(0, 0, 0), font=font)

        comparison.save(os.path.join(COMPARE_DIR, f"compare_{name}"))

    print(f"Processed: {name}")

print("\nDone!")
print("→ Comparison images:", COMPARE_DIR)
print("→ Raw outputs:", RAW_OUTPUT_DIR)