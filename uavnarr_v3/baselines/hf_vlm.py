import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from uavnarr import config, data

MODELS_DIR = os.environ.get("UAVNARR_MODELS", "/root/autodl-tmp/models")
SPECS = {
    "qwen25vl": {"path": os.path.join(MODELS_DIR, "Qwen2.5-VL-7B-Instruct"),
                 "lora": r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"},
    "llava_ov": {"path": os.path.join(MODELS_DIR, "llava-onevision-qwen2-7b-ov-hf"),
                 "lora": r"^language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"},
}
QWEN_MAX_PIXELS = 256 * 28 * 28
END = "<|im_end|>"

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def load_model(name, path=None):
    from transformers import AutoProcessor
    if name == "qwen25vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    else:
        from transformers import LlavaOnevisionForConditionalGeneration as M
    path = path or SPECS[name]["path"]
    model = M.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    proc = AutoProcessor.from_pretrained(path)
    proc.tokenizer.padding_side = "left"
    return model, proc

def video_token_id(model):
    c = model.config
    for k in ("video_token_id", "video_token_index"):
        if getattr(c, k, None) is not None:
            return getattr(c, k)
    raise RuntimeError("video token id not found in the model config")

def load_video(name, vname, nf):
    frames = data.sample_frames(vname, num_frames=nf)
    if name == "qwen25vl":
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        W, H = frames[0].size
        h, w = smart_resize(H, W, factor=28, min_pixels=4 * 28 * 28, max_pixels=QWEN_MAX_PIXELS)
        return [f.resize((w, h), Image.BICUBIC) for f in frames]
    return np.stack([np.asarray(f) for f in frames])

def prompt_for(cfg, vname):
    text = cfg["prompt"]
    if cfg.get("box_text"):
        from uavnarr import target
        b8 = target.sampled_boxes(vname, cfg["num_frames"])
        W, H = target.frame_size(vname)
        text += " " + target.box_text(b8, W, H)
    return text

def chat_prefix(proc, text):
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True)

def encode(proc, cfg, vname, caption=None):
    video = load_video(cfg["model"], vname, cfg["num_frames"])
    prefix = chat_prefix(proc, prompt_for(cfg, vname))
    if caption is None:
        return proc(text=[prefix], videos=[video], return_tensors="pt")
    target_text = caption.strip() + END
    enc = proc(text=[prefix + target_text], videos=[video], return_tensors="pt")
    tgt = proc.tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    n = tgt.shape[0]
    ids = enc["input_ids"][0]
    if not torch.equal(ids[-n:], tgt):
        raise RuntimeError(f"{vname}: caption tokens differ between joint and separate tokenisation")
    labels = enc["input_ids"].clone()
    labels[:, :-n] = -100
    enc["labels"] = labels
    return enc

def to_device(enc, device):
    out = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(torch.bfloat16)
        out[k] = v
    return out

def train(a):
    set_seed(a.seed)
    run_dir = os.path.join(config.OUT_ROOT, "baselines", f"{a.model}{'_boxtext' if a.box_text else ''}_lora",
                           f"seed{a.seed}")
    if a.name:
        run_dir = os.path.join(config.OUT_ROOT, "baselines", a.name, f"seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    cfg = {"model": a.model, "model_path": a.model_path or SPECS[a.model]["path"], "prompt": a.prompt,
           "num_frames": a.num_frames, "box_text": a.box_text, "seed": a.seed, "lr": a.lr,
           "weight_decay": a.weight_decay, "epochs": a.epochs, "grad_accum": a.grad_accum,
           "lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "lora_dropout": 0.05}
    model, proc = load_model(a.model, cfg["model_path"])
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
                      target_modules=SPECS[a.model]["lora"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    n_mod = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    n_layers = model.config.get_text_config().num_hidden_layers
    if n_mod != 7 * n_layers:
        raise RuntimeError(f"LoRA attached to {n_mod} modules, expected {7 * n_layers}; check the regex "
                           f"against this transformers version")
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    cfg["param_counts"] = {"trainable": trainable, "total": total}
    print(f"[params] trainable {trainable / 1e6:.2f}M of {total / 1e9:.2f}B ({100 * trainable / total:.2f}%)")
    model = model.to(a.device)
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    tr, va = data.load_split("train"), data.load_split("val")
    caps = {v: data.load_language(v) for v in tr + va}
    log_path = os.path.join(run_dir, "train_log.jsonl")
    for stale in (log_path, os.path.join(run_dir, "done.txt")):
        if os.path.exists(stale):
            os.remove(stale)

    def step_loss(v):
        enc = to_device(encode(proc, cfg, v, caps[v]), a.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
            return model(**enc).loss, enc

    best, checked = math.inf, False
    for ep in range(1, a.epochs + 1):
        model.train()
        order = tr[:]
        random.Random(a.seed * 1000 + ep).shuffle(order)
        if a.max_steps:
            order = order[:a.max_steps]
        tot, n, skipped, t0 = 0.0, 0, [], time.time()
        opt.zero_grad(set_to_none=True)
        for i, v in enumerate(order, 1):
            try:
                loss, enc = step_loss(v)
            except Exception as e:
                if not checked:
                    raise
                skipped.append({"video": v, "error": repr(e)})
                continue
            if not torch.isfinite(loss):
                if not checked:
                    raise RuntimeError(f"{v}: non-finite loss on the first step")
                skipped.append({"video": v, "error": "non-finite loss"})
                opt.zero_grad(set_to_none=True)
                continue
            (loss / a.grad_accum).backward()
            if not checked:
                nv = int((enc["input_ids"] == video_token_id(model)).sum())
                nl = int((enc["labels"] != -100).sum())
                g = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
                print(f"[check] visual tokens {nv}, supervised caption tokens {nl}, |grad| LoRA {g:.3e}")
                assert nv > 0 and nl > 3 and g > 0, "video tokens, caption labels or LoRA gradients missing"
                cfg["visual_tokens_first_video"] = nv
                checked = True
            tot += loss.item()
            n += 1
            if i % a.grad_accum == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if i % 50 == 0:
                print(f"  epoch {ep} step {i}/{len(order)} loss {tot / max(n, 1):.4f} "
                      f"mem {peak_gb():.1f}GB {time.time() - t0:.0f}s", flush=True)
        sched.step()
        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for v in (va[:a.max_steps] if a.max_steps else va):
                try:
                    loss, _ = step_loss(v)
                    if torch.isfinite(loss):
                        vt += float(loss)
                        vn += 1
                except Exception as e:
                    skipped.append({"video": v, "error": "val: " + repr(e)})
        val = vt / max(vn, 1)
        log = {"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": val, "train_used": n, "val_used": vn,
               "skipped": skipped, "peak_mem_gb": peak_gb(),
               "sec_per_train_video": (time.time() - t0) / max(len(order), 1)}
        print(f"[epoch {ep}] train {log['train_loss']:.4f} val {val:.4f} skipped {len(skipped)} | "
              f"peak {log['peak_mem_gb']:.1f} GB | {log['sec_per_train_video']:.2f} s/video", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")
        if val < best:
            best = val
            model.save_pretrained(os.path.join(run_dir, "best"))
            with open(os.path.join(run_dir, "best", "epoch.txt"), "w") as f:
                f.write(str(ep))
            print(f"  * best val loss {val:.4f}")
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(run_dir, "done.txt"), "w") as f:
        f.write(f"best val loss {best:.4f}\n")

def infer(a):
    if a.zeroshot:
        cfg = {"model": a.model, "model_path": a.model_path or SPECS[a.model]["path"], "prompt": a.prompt,
               "num_frames": a.num_frames, "box_text": a.box_text}
        model, proc = load_model(a.model, cfg["model_path"])
    else:
        run = a.run if os.path.isabs(a.run) else os.path.join(config.OUT_ROOT, a.run)
        with open(os.path.join(run, "run_config.json")) as f:
            cfg = json.load(f)
        model, proc = load_model(cfg["model"], cfg["model_path"])
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(run, "best")).merge_and_unload()
    model = model.to(a.device).eval()
    vid_tok = video_token_id(model)
    vids = data.load_split(a.split)
    if a.limit:
        vids = vids[:a.limit]
    preds, meta, errors = {}, {}, []
    for k, v in enumerate(vids):
        try:
            t0 = time.perf_counter()
            enc = to_device(encode(proc, cfg, v), a.device)
            sync()
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None)
            sync()
            t2 = time.perf_counter()
            n_in = enc["input_ids"].shape[1]
            preds[v] = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()
            meta[v] = {"visual_tokens": int((enc["input_ids"] == vid_tok).sum()), "input_tokens": n_in,
                       "new_tokens": int(out.shape[1] - n_in), "prep_ms": 1e3 * (t1 - t0),
                       "generate_ms": 1e3 * (t2 - t1)}
        except Exception as e:
            errors.append({"video": v, "error": repr(e)})
        if k < 2 and v in preds:
            print(f"  {v}: {preds[v]}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    lat = [m["generate_ms"] for k_, m in list(meta.items())[3:]] or [m["generate_ms"] for m in meta.values()]
    summ = {"videos": len(preds), "visual_tokens_mean": float(np.mean([m["visual_tokens"] for m in meta.values()])),
            "generate_ms_p50": float(np.median(lat)), "generate_ms_mean": float(np.mean(lat)),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    import transformers
    with open(a.out.replace(".json", ".meta.json"), "w") as f:
        json.dump({"args": vars(a), "run_config": cfg, "summary": summ, "transformers": transformers.__version__,
                   "generation_config": model.generation_config.to_diff_dict(),
                   "per_video": meta, "errors": errors}, f, indent=2)
    print(f"[ok] {len(preds)}/{len(vids)} captions -> {a.out}; {len(errors)} errors | {json.dumps(summ)}")
    if errors:
        print("first errors:", errors[:3])
        raise SystemExit("some videos failed; see the .meta.json file (do not evaluate partial outputs)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer"])
    ap.add_argument("--model", choices=list(SPECS), required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--name", help="run name (default <model>_lora or <model>_boxtext_lora)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--box-text", action="store_true", help="ObjectMLLM-style box coordinates in the prompt")
    ap.add_argument("--prompt", default=config.STRUCTURED_PROMPT)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: first N videos per epoch")
    ap.add_argument("--run", help="infer: run dir relative to OUT_ROOT, e.g. baselines/qwen25vl_lora/seed0")
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.mode == "train":
        train(a)
    else:
        if not a.out or not (a.zeroshot or a.run):
            raise SystemExit("infer needs --out and either --zeroshot or --run")
        infer(a)

if __name__ == "__main__":
    main()
