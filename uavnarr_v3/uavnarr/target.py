import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw

from . import config, data

IMG = 336
PATCH = 14
GRID = 13
SLICES = 5

def load_boxes(video_name):
    vdir = os.path.join(config.DATA_ROOT, video_name)
    rows = []
    with open(os.path.join(vdir, "groundtruth_rect.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                vals = [float(v) for v in re.split(r"[,\s]+", line) if v != ""]
                rows.append(vals[:4] if len(vals) >= 4 else [0.0, 0.0, 0.0, 0.0])
    boxes = np.asarray(rows, dtype=np.float64)
    absent = (boxes[:, 2] <= 0) | (boxes[:, 3] <= 0) | ~np.isfinite(boxes).all(1)
    ap = os.path.join(vdir, "absent.txt")
    if os.path.exists(ap):
        with open(ap) as f:
            flags = [int(float(v)) for v in re.split(r"[,\s]+", f.read().strip()) if v != ""]
        if len(flags) == len(boxes):
            absent |= np.asarray(flags, dtype=bool)
    return boxes, absent

def frame_size(video_name):
    names = data.frame_names(video_name, 1)
    with Image.open(os.path.join(config.DATA_ROOT, video_name, "img", names[0])) as im:
        return im.size

def sampled_boxes(video_name, num_frames=8):
    boxes, absent = load_boxes(video_name)
    n_total = len(data.frame_names(video_name, 10 ** 9))
    if len(boxes) != n_total:
        raise ValueError(f"{video_name}: {len(boxes)} boxes for {n_total} frames")
    idx = data.frame_indices(n_total, num_frames)
    return [None if absent[i] else tuple(boxes[i]) for i in idx]

def fill_missing(boxes):
    out = list(boxes)
    last = None
    for i, b in enumerate(out):
        if b is None:
            out[i] = last
        else:
            last = b
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if boxes[i] is not None:
            nxt = boxes[i]
        if out[i] is None:
            out[i] = nxt
    return out

def to_336(box, W, H, mode):
    x, y, w, h = box
    if mode == "pad":
        S = max(W, H)
        s = IMG / S
        ox, oy = (S - W) / 2 * s, (S - H) / 2 * s
    elif mode == "crop":
        s = IMG / min(W, H)
        ox, oy = -(W * s - IMG) / 2, -(H * s - IMG) / 2
    else:
        raise ValueError(mode)
    return x * s + ox, y * s + oy, (x + w) * s + ox, (y + h) * s + oy

def _cell_bounds():
    b = []
    for j in range(GRID):
        lo, hi = max(0, (2 * j - 1) * PATCH), min(IMG, (2 * j + 1) * PATCH)
        b.append((lo, hi))
    return b

CELLS = _cell_bounds()
SLICE_FRAMES = [[f for f in (2 * i - 1, 2 * i) if 0 <= f < 8] for i in range(SLICES)]

def occupancy(boxes8, W, H, mode):
    occ = np.zeros((SLICES, GRID, GRID), dtype=np.float32)
    for i, frames in enumerate(SLICE_FRAMES):
        for f in frames:
            b = boxes8[f] if f < len(boxes8) else None
            if b is None:
                continue
            x1, y1, x2, y2 = to_336(b, W, H, mode)
            for r, (ylo, yhi) in enumerate(CELLS):
                oy = max(0.0, min(y2, yhi) - max(y1, ylo))
                if oy <= 0:
                    continue
                for c, (xlo, xhi) in enumerate(CELLS):
                    ox = max(0.0, min(x2, xhi) - max(x1, xlo))
                    if ox > 0:
                        occ[i, r, c] = max(occ[i, r, c], ox * oy / ((xhi - xlo) * (yhi - ylo)))
    return occ

def visible_fraction(boxes8, W, H, mode):
    n = v = 0
    for b in boxes8:
        if b is None:
            continue
        x1, y1, x2, y2 = to_336(b, W, H, mode)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        n += 1
        v += (0 <= cx < IMG) and (0 <= cy < IMG)
    return v / n if n else float("nan")

def zoom_crops(frames, boxes8, context=2.0, min_side=64):
    filled = fill_missing(boxes8)
    out = []
    for im, b in zip(frames, filled):
        W, H = im.size
        if b is None:
            side = min(W, H)
            cx, cy = W / 2, H / 2
        else:
            x, y, w, h = b
            side = max(min_side, context * max(w, h))
            cx, cy = x + w / 2, y + h / 2
        half = side / 2
        out.append(im.crop((int(round(cx - half)), int(round(cy - half)),
                            int(round(cx + half)), int(round(cy + half)))))
    return out

def draw_boxes(frames, boxes8, color=(255, 0, 0)):
    out, centres = [], []
    for im, b in zip(frames, boxes8):
        im = im.copy()
        d = ImageDraw.Draw(im)
        W, H = im.size
        lw = max(2, int(round(min(W, H) / 200)))
        if b is not None:
            x, y, w, h = b
            centres.append((x + w / 2, y + h / 2))
            d.rectangle([x, y, x + w, y + h], outline=color, width=lw)
        if len(centres) >= 2:
            d.line(centres, fill=color, width=lw)
        out.append(im)
    return out

def box_text(boxes8, W, H):
    parts = []
    for k, b in enumerate(boxes8, 1):
        if b is None:
            parts.append(f"frame {k}: absent")
        else:
            x, y, w, h = b
            parts.append(f"frame {k}: [{x / W:.2f}, {y / H:.2f}, {(x + w) / W:.2f}, {(y + h) / H:.2f}]")
    return ("Bounding box of the target in each sampled frame (normalised x1, y1, x2, y2): "
            + "; ".join(parts) + ".")

TRAJ_T = 32
TRAJ_DIM = 7

def trajectory_features(video_name, T=TRAJ_T):
    boxes, absent = load_boxes(video_name)
    W, H = frame_size(video_name)
    n = len(boxes)
    idx = np.linspace(0, n - 1, T).round().astype(int)
    feats = np.zeros((T, TRAJ_DIM), dtype=np.float32)
    known = [None if absent[i] else boxes[i] for i in range(n)]
    last = next((b for b in known if b is not None), None)
    prev_c = None
    for k, i in enumerate(idx):
        b = known[i]
        a = b is None
        if b is None:
            b = last
        else:
            last = b
        if b is None:
            feats[k, 4] = 1.0
            continue
        x, y, w, h = b
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        dcx, dcy = (0.0, 0.0) if prev_c is None else (cx - prev_c[0], cy - prev_c[1])
        prev_c = (cx, cy)
        feats[k] = [cx, cy, w / W, h / H, float(a), 10 * dcx, 10 * dcy]
    return feats
