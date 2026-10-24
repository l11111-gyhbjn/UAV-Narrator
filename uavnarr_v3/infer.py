import argparse
import json
import os

import torch

from uavnarr import config, data, target
from uavnarr.model import attach_modules, import_videollama2, load_base
from uavnarr.modules import AttrCond, MSEnc, TargetTokenCompressor, TrajEncoder
from uavnarr.sample import SampleBuilder, derangement

def load_trained(model, run, ckpt="best"):
    run = run if os.path.isabs(run) else os.path.join(config.OUT_ROOT, run)
    with open(os.path.join(run, "run_config.json")) as f:
        cfg = json.load(f)
    ck = os.path.join(run, ckpt)

    def load(mod, name):
        mod.load_state_dict(torch.load(os.path.join(ck, name), map_location="cpu"))
        return mod.float()

    ms_enc = attr_cond = tgc = traj = None
    if cfg.get("msenc", "none") != "none":
        ms_enc = load(MSEnc(mode=cfg["msenc"], spatial_tokens=cfg.get("spatial_tokens", 169)), "ms_enc.pt")
    if cfg.get("attrcond"):
        attr_cond = load(AttrCond(num_attrs=data.ENCODING_DIM[cfg["attr_encoding"]],
                                  num_prefix_tokens=cfg.get("num_prefix", 4)), "attr_cond.pt")
    if cfg.get("tgc"):
        tgc = load(TargetTokenCompressor(ring=cfg.get("tgc_ring", 1), max_fg=cfg.get("tgc_max_fg", 64),
                                         use_zoom=cfg.get("zoom", True), select=cfg.get("tgc_select", "box"),
                                         uniform_grid=cfg.get("uniform_grid", 5)), "tgc.pt")
    if cfg.get("traj"):
        traj = load(TrajEncoder(steps=target.TRAJ_T, in_dim=target.TRAJ_DIM), "traj.pt")
    attach_modules(model, ms_enc, attr_cond, tgc, traj)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ck).merge_and_unload()
    return model, cfg

ZEROSHOT_CFG = {"msenc": "none", "attrcond": False, "attr_text": False, "attr_encoding": "onehot23",
                "prompt": config.STRUCTURED_PROMPT, "frame_mode": "crop"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory (runs/<name>/seed<k>)")
    ap.add_argument("--ckpt", default="best", help="checkpoint sub-folder inside the run")
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--attr-mode", choices=["annotated", "shuffled", "zeros", "file"], default="annotated")
    ap.add_argument("--attr-file", help="JSON {video_id: attribute dict} for --attr-mode file")
    ap.add_argument("--target-mode", choices=["annotated", "shuffled"], default="annotated")
    ap.add_argument("--rag-mode", choices=["retrieved", "shuffled"], default="retrieved")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--box-noise", type=float, default=0.0,
                    help="simulated tracker error (sigma, relative to the target size), seeded by --shuffle-seed")
    ap.add_argument("--prompt", help="override the prompt (prompt study)")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    videollama2, *_ = import_videollama2()
    model, processor, tokenizer = load_base()
    if a.zeroshot:
        cfg = dict(ZEROSHOT_CFG)
        attach_modules(model)
    else:
        model, cfg = load_trained(model, a.run, a.ckpt)
    if a.prompt:
        cfg["prompt"] = a.prompt
    model = model.to("cuda").eval()
    builder = SampleBuilder(cfg, processor)
    if a.box_noise > 0:
        builder.box_noise = (a.box_noise, a.shuffle_seed)
    ious, ntok = [], []

    vids = data.load_split(a.split)
    attrs = {v: data.load_attributes(v) for v in vids}
    perm = derangement(vids, a.shuffle_seed)
    if a.attr_mode == "shuffled":
        attrs = {v: attrs[perm[v]] for v in vids}
    elif a.attr_mode == "file":
        with open(a.attr_file) as f:
            attrs = json.load(f)
    other = lambda v, on: perm[v] if on else None

    preds, errors = {}, []
    for v in vids:
        try:
            if getattr(model, "uav_attr_cond", None) is not None:
                dim = data.ENCODING_DIM[cfg["attr_encoding"]]
                if cfg.get("attr_constant"):
                    vec = [1.0] * dim
                elif a.attr_mode == "zeros":
                    vec = [0.0] * dim
                else:
                    vec = data.encode_attributes(attrs[v], cfg["attr_encoding"])
                model._attr_vec = torch.tensor(vec, device="cuda")
            tshuf = a.target_mode == "shuffled"
            video, prompt, side = builder.build(v, attrs=attrs[v], box_src=other(v, tshuf),
                                                traj_src=other(v, tshuf),
                                                rag_src=other(v, a.rag_mode == "shuffled"))
            if "zoom" in side:
                side["zoom"] = side["zoom"].to("cuda", torch.float16)
            model._side = side
            if builder.last_iou is not None:
                ious.append(builder.last_iou)
            model._last_ntok = None
            preds[v] = videollama2.mm_infer(video, prompt, model=model, tokenizer=tokenizer,
                                            do_sample=False, max_new_tokens=a.max_new_tokens).strip()
            if getattr(model, "_last_ntok", None):
                ntok.append(model._last_ntok)
        except Exception as e:
            errors.append({"video": v, "error": repr(e)})
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    with open(a.out.replace(".json", ".meta.json"), "w") as f:
        json.dump({"args": vars(a), "run_config": cfg, "errors": errors,
                   "box_iou_mean": float(sum(ious) / len(ious)) if ious else None,
                   "visual_tokens_mean": float(sum(ntok) / len(ntok)) if ntok else None}, f, indent=2)
    if ntok:
        print(f"visual tokens passed to the LLM: mean {sum(ntok) / len(ntok):.1f}")
    if ious:
        print(f"mean IoU of the perturbed boxes with the annotations: {sum(ious) / len(ious):.3f}")
    print(f"[ok] {len(preds)}/{len(vids)} captions -> {a.out}; {len(errors)} errors")
    if errors:
        print("first errors:", errors[:3])
        raise SystemExit("some videos failed; see the .meta.json file (do not evaluate partial outputs)")

if __name__ == "__main__":
    main()
