import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavnarr import data, target

DEFAULT_CLIP = "/root/autodl-tmp/models/clip-vit-large-patch14-336"

@torch.no_grad()
def embed(vids, model, proc, device, num_frames=8, batch=8):
    out = []
    t0 = time.time()
    for i, v in enumerate(vids, 1):
        frames = data.sample_frames(v, num_frames)
        crops = target.zoom_crops(frames, target.sampled_boxes(v, num_frames))
        px = proc(images=crops, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        e = model(pixel_values=px).pooler_output.float().mean(0)
        out.append(torch.nn.functional.normalize(e, dim=0).cpu().numpy())
        if i % 100 == 0:
            print(f"  {i}/{len(vids)}  {time.time() - t0:.0f}s")
    return np.stack(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.environ.get("UAVNARR_CLIP", DEFAULT_CLIP))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="data/rag_neighbors.json")
    a = ap.parse_args()
    from transformers import CLIPImageProcessor, CLIPVisionModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPVisionModel.from_pretrained(a.clip, torch_dtype=torch.float16).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(a.clip)

    splits = {s: data.load_split(s) for s in ("train", "val", "test")}
    emb = {s: embed(v, model, proc, device) for s, v in splits.items()}
    train = splits["train"]
    neighbors, report = {}, {}
    for s, vids in splits.items():
        sim = emb[s] @ emb["train"].T
        if s == "train":
            np.fill_diagonal(sim, -np.inf)
        same = 0
        for i, v in enumerate(vids):
            top = np.argsort(-sim[i])[: a.k]
            neighbors[v] = [[train[j], float(sim[i, j])] for j in top]
            same += data.category_from_vid(train[top[0]]) == data.category_from_vid(v)
        report[s] = round(100 * same / len(vids), 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"meta": {"encoder": a.clip, "view": "zoomed target, mean of 8 frames",
                            "k": a.k, "memory": "train split", "top1_same_category_%": report},
                   "neighbors": neighbors}, f, indent=1)
    print("[ok] neighbours ->", a.out)
    print("top-1 retrieved video has the same target category (%):", report)

if __name__ == "__main__":
    main()
