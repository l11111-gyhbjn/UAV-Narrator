import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavnarr import data, target

RED = (220, 40, 40)

def pick_frame(boxes, forced):
    if forced >= 0:
        return forced
    order = sorted(range(len(boxes)), key=lambda i: abs(i - (len(boxes) - 1) / 2))
    return next((i for i in order if boxes[i] is not None), order[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="case")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--width", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for v in a.videos:
        frames = data.sample_frames(v, 8)
        boxes = target.sampled_boxes(v, 8)
        k = pick_frame(boxes, a.frame)
        src = frames[k]
        W, H = src.size
        s = a.width / W
        img = src.resize((a.width, round(H * s)), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        lw = max(3, a.width // 220)
        b = boxes[k]
        if b is not None:
            x, y, w, h = b
            x0, y0 = max(0, x * s), max(0, y * s)
            x1, y1 = min(img.width - 1, (x + w) * s), min(img.height - 1, (y + h) * s)
            d.rectangle([x0, y0, x1, y1], outline=RED, width=lw)
            side = max(2 * max(w, h), 64)
            cx, cy = x + w / 2, y + h / 2
            crop = src.crop((round(cx - side / 2), round(cy - side / 2),
                             round(cx + side / 2), round(cy + side / 2)))
            ins = round(img.height * 0.45)
            crop = crop.resize((ins, ins), Image.LANCZOS)
            right = cx * s < img.width / 2
            bottom = cy * s < img.height / 2
            m = lw * 2
            px = img.width - ins - m if right else m
            py = img.height - ins - m if bottom else m
            img.paste(crop, (px, py))
            d.rectangle([px - lw, py - lw, px + ins + lw - 1, py + ins + lw - 1], outline=(255, 255, 255), width=lw)
            d.rectangle([px, py, px + ins - 1, py + ins - 1], outline=RED, width=lw)
        name = v if a.frame < 0 else f"{v}_f{k}"
        path = os.path.join(a.out, f"{name}.jpg")
        img.save(path, quality=92)
        print(f"[ok] {path}  sampled frame {k}/7  box {None if b is None else tuple(round(t) for t in b)}  frame {W}x{H}")

if __name__ == "__main__":
    main()
