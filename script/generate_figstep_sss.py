"""
Generate FigStep-style images from HoliSafe SSS (safe image + safe query).

Same typography method as figstep_mmhb_convert.py: overlays the query text
with numbered "1. / 2. / 3." step blanks onto the image with a
semi-transparent black strip.

Usage:
    python generate_figstep_sss.py
"""

import json
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

SSS_DIR = os.path.expanduser("~/work/vllmstudy/data/holisafe/sss")
OUT_DIR = os.path.expanduser("~/work/testbench/figstep_sss")
FONT_PATH = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
TARGET_SIZE = (760, 760)
N_STEPS = 3
N = 100


def wrap_text(text, width):
    return textwrap.fill(text, width=width)


def text_step_by_step(text, width, steps=N_STEPS):
    text = text.removesuffix("\n")
    text = wrap_text(text, width=width)
    for idx in range(1, steps + 1):
        text += f"\n{idx}. "
    return text


def find_font_size(text, max_width, max_height):
    for size in range(48, 16, -2):
        font = ImageFont.truetype(FONT_PATH, size)
        char_w = font.getbbox("M")[2]
        chars_per_line = max(10, (max_width - 40) // char_w)
        rendered = text_step_by_step(text, width=chars_per_line)
        tmp = Image.new("RGB", (1, 1))
        dr = ImageDraw.Draw(tmp)
        bbox = dr.textbbox((0, 0), rendered, font=font, spacing=8)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= max_width - 40 and th <= max_height - 20:
            return font, rendered, size
    font = ImageFont.truetype(FONT_PATH, 18)
    char_w = font.getbbox("M")[2]
    chars_per_line = max(10, (max_width - 40) // char_w)
    rendered = text_step_by_step(text, width=chars_per_line)
    return font, rendered, 18


def overlay_text_on_image(img, query):
    img = img.copy()
    w, h = img.size
    scale = max(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - TARGET_SIZE[0]) // 2
    top = (new_h - TARGET_SIZE[1]) // 2
    img = img.crop((left, top, left + TARGET_SIZE[0], top + TARGET_SIZE[1]))

    font, rendered_text, _ = find_font_size(
        query, TARGET_SIZE[0], TARGET_SIZE[1] // 2
    )

    tmp_dr = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = tmp_dr.textbbox((0, 0), rendered_text, font=font, spacing=8)
    text_h = bbox[3] - bbox[1]

    overlay = Image.new("RGBA", TARGET_SIZE, (0, 0, 0, 0))
    overlay_dr = ImageDraw.Draw(overlay)
    padding = 15
    bg_height = text_h + padding * 2
    overlay_dr.rectangle([(0, 0), (TARGET_SIZE[0], bg_height)], fill=(0, 0, 0, 180))
    overlay_dr.text((20, padding), rendered_text, font=font,
                    fill=(255, 255, 255, 255), spacing=8)

    img = img.convert("RGBA")
    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def main():
    with open(os.path.join(SSS_DIR, "metadata.json")) as f:
        entries = json.load(f)[:N]

    os.makedirs(OUT_DIR, exist_ok=True)
    metadata = []

    for entry in entries:
        img_path = os.path.join(SSS_DIR, entry["image"])
        img = Image.open(img_path).convert("RGB")
        query = entry["query"]

        result = overlay_text_on_image(img, query)
        out_filename = f"{entry['index']:03d}.png"
        result.save(os.path.join(OUT_DIR, out_filename))
        metadata.append({
            "index": entry["index"],
            "query": query,
            "image": out_filename,
            "original_image": entry["image"],
        })

        if (entry["index"] + 1) % 20 == 0:
            print(f"  {entry['index']+1}/{N}")

    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(metadata)} FigStep-SSS images in {OUT_DIR}/")


if __name__ == "__main__":
    main()
