import json
import math
import os
import zlib

import numpy as np
import torch

from . import data, target

class SampleBuilder:
    def __init__(self, cfg, processor):
        self.process_video = processor["video"].func
        self.cfg = cfg
        self.img_proc = processor["video"].keywords["processor"]
        self.nf = cfg.get("num_frames", 8)
        self.rag = None
        self.box_noise = None
        self.last_iou = None
        if cfg.get("rag_k", 0):
            with open(cfg["rag_file"]) as f:
                self.rag = json.load(f)["neighbors"]

    def preprocess(self, frames, mode):
        return self.process_video(frames, processor=self.img_proc,
                                  aspect_ratio="pad" if mode == "pad" else None, num_frames=self.nf)

    def needs_boxes(self):
        c = self.cfg
        return bool(c.get("tgc") or c.get("box_text") or c.get("box_draw"))

    def boxes_for(self, vname, box_src, W, H):
        b8 = target.sampled_boxes(box_src, self.nf)
        if box_src == vname:
            return b8
        sW, sH = target.frame_size(box_src)
        sx, sy = W / sW, H / sH
        return [None if b is None else (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in b8]

    def perturb(self, vname, b8, W, H):
        sigma, seed = self.box_noise
        rng = np.random.default_rng(zlib.crc32(f"{vname}|{seed}|{sigma}".encode()))
        out, ious = [], []
        for b in b8:
            if b is None:
                out.append(None)
                continue
            if rng.random() < sigma / 2:
                out.append(None)
                ious.append(0.0)
                continue
            x, y, w, h = b
            s = max(w, h)
            cx, cy = x + w / 2 + rng.normal(0, sigma) * s, y + h / 2 + rng.normal(0, sigma) * s
            w2, h2 = w * math.exp(rng.normal(0, sigma)), h * math.exp(rng.normal(0, sigma))
            x1, y1 = min(max(cx - w2 / 2, 0.0), W - 1.0), min(max(cy - h2 / 2, 0.0), H - 1.0)
            x2, y2 = min(max(cx + w2 / 2, x1 + 1.0), float(W)), min(max(cy + h2 / 2, y1 + 1.0), float(H))
            nb = (x1, y1, x2 - x1, y2 - y1)
            ix = max(0.0, min(x + w, x2) - max(x, x1))
            iy = max(0.0, min(y + h, y2) - max(y, y1))
            inter = ix * iy
            ious.append(inter / (w * h + nb[2] * nb[3] - inter))
            out.append(nb)
        self.last_iou = float(np.mean(ious)) if ious else None
        return out

    def rag_text(self, vname):
        nb = self.rag[vname][: self.cfg["rag_k"]]
        caps = [data.load_language(v) for v, _ in nb]
        return ("Captions of visually similar targets from the archive: "
                + " ".join(f"({i}) {c.rstrip('.')}." for i, c in enumerate(caps, 1)))

    def build(self, vname, attrs=None, box_src=None, traj_src=None, rag_src=None):
        c = self.cfg
        frames = data.sample_frames(vname, num_frames=self.nf)
        W, H = frames[0].size
        side, extra = {}, []
        b8 = self.boxes_for(vname, box_src or vname, W, H) if self.needs_boxes() else None
        if b8 is not None and self.box_noise and self.box_noise[0] > 0:
            b8 = self.perturb(vname, b8, W, H)
        shown = target.draw_boxes(frames, b8) if c.get("box_draw") else frames
        video = self.preprocess(shown, c.get("frame_mode", "crop"))
        if c.get("tgc"):
            side["occ"] = torch.from_numpy(target.occupancy(b8, W, H, c.get("frame_mode", "crop")))
            if c.get("zoom", True):
                side["zoom"] = self.preprocess(target.zoom_crops(frames, b8), "pad")
        if c.get("traj"):
            side["traj"] = torch.from_numpy(target.trajectory_features(traj_src or vname))
        if c.get("box_draw"):
            extra.append("The tracked target is marked by the red box.")
        if c.get("box_text"):
            extra.append(target.box_text(b8, W, H))
        if c.get("rag_k", 0):
            extra.append(self.rag_text(rag_src or vname))
        if c.get("attr_text"):
            extra.append(data.attr_sentence(attrs if attrs is not None else data.load_attributes(vname)))
        prompt = c["prompt"] + ("" if not extra else " " + " ".join(extra))
        return video, prompt, side

def derangement(vids, seed):
    import random
    rng, perm = random.Random(seed), list(vids)
    while True:
        rng.shuffle(perm)
        if all(a != b for a, b in zip(vids, perm)):
            return dict(zip(vids, perm))
