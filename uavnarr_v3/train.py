import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from uavnarr import config, data
from uavnarr.model import (IGNORE_INDEX, attach_modules, count_parameters, encode_example,
                           import_videollama2, load_base, lora_config)
from uavnarr.modules import AttrCond, MSEnc, TargetTokenCompressor, TrajEncoder
from uavnarr.sample import SampleBuilder
from uavnarr import target

EXTRA = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="configuration name, e.g. lora / msenc / attrcond / full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--msenc", choices=["none", "temporal", "flat"], default="none")
    ap.add_argument("--spatial-tokens", type=int, default=169, help="13x13 for VideoLLaMA2-7B")
    ap.add_argument("--attrcond", action="store_true")
    ap.add_argument("--attr-encoding", choices=list(data.ENCODING_DIM), default="onehot23")
    ap.add_argument("--num-prefix", type=int, default=4)
    ap.add_argument("--attr-text", action="store_true", help="append the attributes to the prompt as text")
    ap.add_argument("--attr-constant", action="store_true",
                    help="control: feed the same constant vector to Attr-Cond for every video, i.e. 4 learned "
                         "prefix tokens that carry no attribute information (soft-prompt baseline)")
    ap.add_argument("--lora-vision", action="store_true", help="also put LoRA on CLIP q/k/v (old behaviour)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--module-lr", type=float, default=None,
                    help="learning rate of Attr-Cond (default: same as --lr). The MLP starts from a random "
                         "projection, so it usually needs a larger rate than the LoRA matrices")
    ap.add_argument("--msenc-lr", type=float, default=None,
                    help="learning rate of the 4 MS-Enc scalars (default: same as --lr). With 5e-5 they "
                         "move by <1e-3 in 3 epochs, i.e. MS-Enc stays at its initialisation")
    ap.add_argument("--module-wd", type=float, default=None,
                    help="weight decay of MS-Enc / Attr-Cond (default: same as --weight-decay, i.e. the "
                         "behaviour of the existing seed-0 runs)")
    ap.add_argument("--frame-mode", choices=["crop", "pad"], default="crop",
                    help="crop = VideoLLaMA2 default (center crop); pad = pad to square, keeps the whole frame")
    ap.add_argument("--tgc", action="store_true", help="target-guided token compression")
    ap.add_argument("--no-zoom", action="store_true", help="TGC without the zoomed target view")
    ap.add_argument("--tgc-select", choices=["box", "attn", "random", "uniform"], default="box",
                    help="comparison at a matched budget: box = TGC; attn = VisionZip-style [CLS] attention; "
                         "random; uniform = average pooling to --uniform-grid per slice (box-free modes need --no-zoom)")
    ap.add_argument("--uniform-grid", type=int, default=5)
    ap.add_argument("--tgc-ring", type=int, default=1)
    ap.add_argument("--tgc-max-fg", type=int, default=64)
    ap.add_argument("--traj", action="store_true", help="trajectory tokens from the tracked boxes")
    ap.add_argument("--box-text", action="store_true", help="baseline: box coordinates in the prompt (ObjectMLLM)")
    ap.add_argument("--box-draw", action="store_true", help="baseline: boxes drawn on the frames (BoxTuning)")
    ap.add_argument("--rag-k", type=int, default=0, help="retrieved training captions in the prompt")
    ap.add_argument("--rag-file", default="data/rag_neighbors.json")
    ap.add_argument("--tgc-lr", type=float, default=1e-3,
                    help="TGC holds only zero-initialised embedding vectors; they need a larger rate")
    ap.add_argument("--traj-lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=768, help="text tokens, excluding visual tokens")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many videos")
    ap.add_argument("--prompt", default=config.STRUCTURED_PROMPT)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()

def build(args):
    _, tok_mm, modal_map, video_token = import_videollama2()
    model, processor, tokenizer = load_base()
    ms_enc = MSEnc(mode=args.msenc, spatial_tokens=args.spatial_tokens) if args.msenc != "none" else None
    attr_cond = (AttrCond(num_attrs=data.ENCODING_DIM[args.attr_encoding], num_prefix_tokens=args.num_prefix)
                 if args.attrcond else None)
    tgc = (TargetTokenCompressor(ring=args.tgc_ring, max_fg=args.tgc_max_fg, use_zoom=not args.no_zoom,
                                 select=args.tgc_select, uniform_grid=args.uniform_grid)
           if args.tgc else None)
    traj = TrajEncoder(steps=target.TRAJ_T, in_dim=target.TRAJ_DIM) if args.traj else None
    attach_modules(model, ms_enc, attr_cond, tgc, traj)
    EXTRA.update(tgc=tgc, traj=traj)
    model.config.use_cache = False
    from peft import get_peft_model
    model = get_peft_model(model, lora_config(args.lora_r, args.lora_alpha, include_vision=args.lora_vision))
    for n, p in model.named_parameters():
        if p.requires_grad and "lora_" in n:
            p.data = p.data.float()
    for m in (ms_enc, attr_cond, tgc, traj):
        if m is not None:
            m.float()
            for n, p in m.named_parameters():
                p.requires_grad = True
                if not torch.isfinite(p).all() or p.abs().max() > 1e3:
                    raise RuntimeError(f"{type(m).__name__}.{n} is not properly initialised "
                                       f"(max |w| = {p.abs().max().item():.3e})")
    model = model.to(args.device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = {**vars(args), "zoom": not args.no_zoom}
    EXTRA["builder"] = SampleBuilder(cfg, processor)
    return model, processor, tokenizer, ms_enc, attr_cond, tok_mm, modal_map, video_token

def make_prompt(args, attrs):
    return args.prompt + (" " + data.attr_sentence(attrs) if args.attr_text else "")

def forward_one(model, processor, tokenizer, args, vname, tok_mm, modal_map, video_token):
    attrs = data.load_attributes(vname)
    caption = data.load_language(vname)
    video, prompt, side = EXTRA["builder"].build(vname, attrs=attrs)
    ids, labels = encode_example(tokenizer, prompt, caption, tok_mm, modal_map, video_token)
    if ids.shape[0] > args.max_length:
        raise RuntimeError(f"{vname}: {ids.shape[0]} text tokens exceed --max-length {args.max_length}")
    fdt = torch.float16 if args.device.startswith("cuda") else torch.float32
    video = video.to(dtype=fdt, device=args.device)
    if "zoom" in side:
        side["zoom"] = side["zoom"].to(dtype=fdt, device=args.device)
    inner = model.base_model.model
    inner._side = side
    vec = ([1.0] * data.ENCODING_DIM[args.attr_encoding] if args.attr_constant
           else data.encode_attributes(attrs, args.attr_encoding))
    inner._attr_vec = torch.tensor(vec, device=args.device)
    ids, labels = ids.unsqueeze(0).to(args.device), labels.unsqueeze(0).to(args.device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                images=[(video, "video")])
    return out, ids

def main():
    args = parse()
    set_seed(args.seed)
    run_dir = os.path.join(config.OUT_ROOT, "runs", args.name, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    model, processor, tokenizer, ms_enc, attr_cond, tok_mm, modal_map, video_token = build(args)

    counts = count_parameters(model)
    print("[params]", json.dumps({k: f"{v / 1e6:.2f}M" for k, v in counts.items()}))
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "zoom": not args.no_zoom, "param_counts": counts}, f, indent=2)

    params = [p for p in model.parameters() if p.requires_grad]
    mod_ids = set()
    groups = []
    wd_mod = args.weight_decay if args.module_wd is None else args.module_wd
    if ms_enc is not None:
        groups.append({"params": list(ms_enc.parameters()), "lr": args.msenc_lr or args.lr,
                       "weight_decay": wd_mod, "name": "ms_enc"})
        mod_ids |= {id(p) for p in ms_enc.parameters()}
    if attr_cond is not None:
        groups.append({"params": list(attr_cond.parameters()), "lr": args.module_lr or args.lr,
                       "weight_decay": wd_mod, "name": "attr_cond"})
        mod_ids |= {id(p) for p in attr_cond.parameters()}
    for nm, mod, lr in (("tgc", EXTRA.get("tgc"), args.tgc_lr), ("traj", EXTRA.get("traj"), args.traj_lr)):
        if mod is not None:
            groups.append({"params": list(mod.parameters()), "lr": lr, "weight_decay": 0.0, "name": nm})
            mod_ids |= {id(p) for p in mod.parameters()}
    groups.insert(0, {"params": [p for p in params if id(p) not in mod_ids], "lr": args.lr,
                      "weight_decay": args.weight_decay, "name": "lora"})
    assert sum(len(g["params"]) for g in groups) == len(params)
    print("[optim]", {g["name"]: (len(g["params"]), g["lr"]) for g in groups})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_videos, val_videos = data.load_split("train"), data.load_split("val")
    print(f"train {len(train_videos)}  val {len(val_videos)}  run_dir {run_dir}")

    best_val, checked = math.inf, False
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_videos[:]
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        if args.max_steps:
            order = order[:args.max_steps]
        tot, n, skipped, t0 = 0.0, 0, [], time.time()
        opt.zero_grad(set_to_none=True)
        for i, vname in enumerate(order, 1):
            try:
                out, ids = forward_one(model, processor, tokenizer, args, vname, tok_mm, modal_map, video_token)
            except Exception as e:
                skipped.append({"video": vname, "error": repr(e)})
                if not checked:
                    raise
                continue
            loss = out.loss
            if not torch.isfinite(loss):
                if not checked:
                    raise RuntimeError(f"{vname}: non-finite loss {loss.item()} on the first step; "
                                       f"run tools/diag_nan.py with the same flags")
                skipped.append({"video": vname, "error": "non-finite loss"})
                opt.zero_grad(set_to_none=True)
                continue
            (loss / args.grad_accum).backward()
            if not checked:
                extra = out.logits.shape[1] - ids.shape[1]
                print(f"[check] visual tokens inserted into the LLM input: {extra}")
                assert extra > 100, "visual features did not reach the language model"
                for nm, mod in (("ms_enc", ms_enc), ("attr_cond", attr_cond),
                                ("tgc", EXTRA.get("tgc")), ("traj", EXTRA.get("traj"))):
                    if mod is not None:
                        g = sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
                        print(f"[check] |grad| of {nm}: {g:.3e}")
                        assert g > 0, f"{nm} receives no gradient"
                checked = True
            tot += loss.item()
            n += 1
            if i % args.grad_accum == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if i % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9 if args.device.startswith("cuda") else 0.0
                print(f"  epoch {epoch} step {i}/{len(order)} loss {tot / max(n, 1):.4f} "
                      f"mem {mem:.1f}GB {time.time() - t0:.0f}s")
        sched.step()

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for vname in (val_videos[:args.max_steps] if args.max_steps else val_videos):
                try:
                    out, _ = forward_one(model, processor, tokenizer, args, vname, tok_mm, modal_map, video_token)
                    if torch.isfinite(out.loss):
                        vtot += float(out.loss)
                        vn += 1
                except Exception as e:
                    skipped.append({"video": vname, "error": "val: " + repr(e)})
        val = vtot / max(vn, 1)
        log = {"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": val,
               "train_used": n, "val_used": vn, "skipped": skipped}
        peak = torch.cuda.max_memory_allocated() / 1e9 if args.device.startswith("cuda") else 0.0
        log["peak_mem_gb"], log["sec_per_train_video"] = peak, (time.time() - t0) / max(len(order), 1)
        if ms_enc is not None:
            log["ms_enc_weights"] = [round(x, 4) for x in torch.softmax(ms_enc.alpha.detach().float(), 0).tolist()]
            log["ms_enc_gate"] = round(float(torch.sigmoid(ms_enc.scale.detach().float())), 4)
        if EXTRA.get("tgc") is not None:
            t_ = EXTRA["tgc"]
            log["tgc_norms"] = {k: round(float(getattr(t_, k).detach().norm()), 4)
                                for k in ("e_fg", "e_core", "e_bg", "e_zoom", "pos_h", "pos_w", "pos_uni")
                                if hasattr(t_, k)}
        print(f"[epoch {epoch}] train {log['train_loss']:.4f} val {val:.4f} skipped {len(skipped)} | "
              f"peak GPU memory {peak:.1f} GB | {log['sec_per_train_video']:.2f} s per video (incl. val)")
        with open(os.path.join(run_dir, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        save(model, tokenizer, ms_enc, attr_cond, os.path.join(run_dir, f"epoch{epoch}"))
        if val < best_val:
            best_val = val
            save(model, tokenizer, ms_enc, attr_cond, os.path.join(run_dir, "best"))
            print(f"  * best val loss {val:.4f}")

def save(model, tokenizer, ms_enc, attr_cond, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    if ms_enc is not None:
        torch.save(ms_enc.state_dict(), os.path.join(path, "ms_enc.pt"))
    if attr_cond is not None:
        torch.save(attr_cond.state_dict(), os.path.join(path, "attr_cond.pt"))
    for nm in ("tgc", "traj"):
        if EXTRA.get(nm) is not None:
            torch.save(EXTRA[nm].state_dict(), os.path.join(path, f"{nm}.pt"))

if __name__ == "__main__":
    main()
