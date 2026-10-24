import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from uavnarr import config, data

def ranks(S):
    c = S[np.arange(len(S)), np.arange(len(S))][:, None]
    return (S > c).sum(1) + ((S == c).sum(1) - 1) / 2.0 + 1.0

def summary(r):
    return {"R@1": 100 * float(np.mean(r <= 1)), "R@5": 100 * float(np.mean(r <= 5)),
            "R@10": 100 * float(np.mean(r <= 10)), "MdR": float(np.median(r)), "MnR": float(np.mean(r))}

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def frames_cache(proc, clip_path, nf):
    tag = os.path.basename(os.path.normpath(clip_path))
    path = os.path.join(config.OUT_ROOT, "baselines", f"clip_frames_{tag}_{nf}.pt")
    if os.path.exists(path):
        return torch.load(path)
    out = {}
    for split in ("train", "val", "test"):
        for v in data.load_split(split):
            px = proc(images=data.sample_frames(v, nf), return_tensors="pt")["pixel_values"]
            if px.shape[0] < nf:
                px = torch.cat([px, px[-1:].repeat(nf - px.shape[0], 1, 1, 1)])
            out[v] = px.half()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    return out

class MeanP(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def video(self, px):
        B, T = px.shape[:2]
        f = self.clip.get_image_features(pixel_values=px.flatten(0, 1))
        f = F.normalize(f, dim=-1).view(B, T, -1).mean(1)
        return F.normalize(f, dim=-1)

    def text(self, tok):
        return F.normalize(self.clip.get_text_features(**tok), dim=-1)

def encode_split(model, tokenizer, frames, vids, dev, bs=16):
    model.eval()
    V, Tx = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        for i in range(0, len(vids), bs):
            chunk = vids[i:i + bs]
            px = torch.stack([frames[v] for v in chunk]).to(dev).float()
            V.append(model.video(px).float().cpu())
            tok = tokenizer([data.load_language(v) for v in chunk], padding=True, truncation=True,
                            max_length=77, return_tensors="pt").to(dev)
            Tx.append(model.text(tok).float().cpu())
    return torch.cat(Tx), torch.cat(V)

SCORES = {}

def evaluate(model, tokenizer, frames, split, dev):
    vids = sorted(data.load_split(split))
    t, v = encode_split(model, tokenizer, frames, vids, dev)
    S = (t @ v.T).numpy()
    SCORES[split] = (vids, S)
    return summary(ranks(S))

def load(clip_path, dev):
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(clip_path, torch_dtype=torch.float32)
    proc = CLIPProcessor.from_pretrained(clip_path)
    return MeanP(clip).to(dev), proc

def train_one(a, lr, seed, log=print):
    set_seed(seed)
    model, proc = load(a.clip, a.device)
    frames = frames_cache(proc.image_processor, a.clip, a.num_frames)
    tr = data.load_split("train")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.2, betas=(0.9, 0.98), eps=1e-6)
    steps_per_ep = math.ceil(len(tr) / a.batch)
    total, warm = steps_per_ep * a.epochs, int(0.1 * steps_per_ep * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    hist, best, best_state = [], -1.0, None
    zs = evaluate(model, proc.tokenizer, frames, "val", a.device)
    hist.append({"epoch": 0, "val": zs})
    for ep in range(1, a.epochs + 1):
        model.train()
        order = tr[:]
        random.Random(seed * 1000 + ep).shuffle(order)
        t0, tot = time.time(), 0.0
        for i in range(0, len(order), a.batch):
            chunk = order[i:i + a.batch]
            if len(chunk) < 2:
                continue
            px = torch.stack([frames[v] for v in chunk]).to(a.device).float()
            tok = proc.tokenizer([data.load_language(v) for v in chunk], padding=True, truncation=True,
                                 max_length=77, return_tensors="pt").to(a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
                v_, t_ = model.video(px).float(), model.text(tok).float()
            logits = model.clip.logit_scale.exp().clamp(max=100) * t_ @ v_.T
            y = torch.arange(len(chunk), device=a.device)
            loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(chunk)
        val = evaluate(model, proc.tokenizer, frames, "val", a.device)
        hist.append({"epoch": ep, "train_loss": tot / len(order), "val": val, "sec": time.time() - t0})
        log(f"  lr {lr:g} seed {seed} epoch {ep}: loss {tot / len(order):.4f} val R@1 {val['R@1']:.1f} "
            f"R@5 {val['R@5']:.1f} MdR {val['MdR']:.1f}")
        if val["R@1"] > best:
            best = val["R@1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
    model.load_state_dict(best_state)
    return model, proc, frames, hist, best_ep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zeroshot", "select", "train"])
    ap.add_argument("--clip", required=True, help="local HF CLIP folder")
    ap.add_argument("--lr", type=float, default=1e-7)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-7, 1e-6, 5e-6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", help="also save the query x video similarity matrices (.npz) of the final "
                                     "model, for rank fusion with the caption index (tools/hybrid_retrieval.py)")
    a = ap.parse_args()
    res = {"method": "CLIP4Clip-meanP" if a.mode != "zeroshot" else "CLIP zero-shot (mean-pooled frames)",
           "clip": a.clip, "num_frames": a.num_frames, "query": "reference caption",
           "gallery": "test videos"}
    if a.mode == "zeroshot":
        model, proc = load(a.clip, a.device)
        frames = frames_cache(proc.image_processor, a.clip, a.num_frames)
        res["val"] = evaluate(model, proc.tokenizer, frames, "val", a.device)
        res["test"] = evaluate(model, proc.tokenizer, frames, "test", a.device)
    elif a.mode == "select":
        res["grid"] = {}
        for lr in a.lrs:
            _, _, _, hist, ep = train_one(a, lr, 0)
            res["grid"][str(lr)] = {"best_epoch": ep, "best_val_R@1": max(h["val"]["R@1"] for h in hist[1:]),
                                    "history": hist}
        res["chosen_lr"] = float(max(res["grid"], key=lambda k: res["grid"][k]["best_val_R@1"]))
        print("chosen lr (validation R@1):", res["chosen_lr"])
    else:
        model, proc, frames, hist, ep = train_one(a, a.lr, a.seed)
        res.update(lr=a.lr, seed=a.seed, epochs=a.epochs, batch=a.batch, best_epoch=ep, history=hist,
                   val=evaluate(model, proc.tokenizer, frames, "val", a.device),
                   test=evaluate(model, proc.tokenizer, frames, "test", a.device))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    if a.scores and SCORES:
        os.makedirs(os.path.dirname(os.path.abspath(a.scores)), exist_ok=True)
        np.savez(a.scores, **{f"{sp}_ids": np.array(ids) for sp, (ids, _) in SCORES.items()},
                 **{f"{sp}_S": S for sp, (_, S) in SCORES.items()})
    if "test" in res:
        t = res["test"]
        print(f"[test] R@1 {t['R@1']:.1f} R@5 {t['R@5']:.1f} R@10 {t['R@10']:.1f} MdR {t['MdR']:.1f} -> {a.out}")

if __name__ == "__main__":
    main()
