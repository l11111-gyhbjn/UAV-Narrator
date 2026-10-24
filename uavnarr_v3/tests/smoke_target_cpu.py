import functools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_cpu as s

import train
from uavnarr import data, target
from uavnarr.sample import SampleBuilder

class FakeVL2T(s.FakeVL2):
    def get_model(self):
        class M:
            def get_vision_tower(_):
                return lambda imgs: torch.rand(imgs.shape[0], 169, 16, dtype=imgs.dtype)
        return M()

def fake_process_video(frames, processor=None, aspect_ratio=None, num_frames=8):
    assert len(frames) <= num_frames and all(hasattr(f, "size") for f in frames)
    return torch.rand(num_frames, 3, 4, 4)

def run(argv):
    tok = s.make_tokenizer()
    processor = {"video": functools.partial(fake_process_video, processor="clip", aspect_ratio=None,
                                            num_frames=8)}
    train.import_videollama2 = lambda: (None, s.tokenizer_multimodal_token, {"<video>": s.VID}, "<video>")
    train.load_base = lambda: (FakeVL2T(len(tok)), processor, tok)
    sys.argv = ["train.py"] + argv
    train.main()

if __name__ == "__main__":
    FAKE = s.FAKE
    vids = [v for k in ("train", "val", "test") for v in data.load_split(k)]
    rag = {"neighbors": {v: [[u, 0.9] for u in data.load_split("train") if u != v][:3] for v in vids}}
    json.dump(rag, open(f"{FAKE}/rag.json", "w"))

    b8 = target.sampled_boxes(vids[0], 8)
    W, H = target.frame_size(vids[0])
    print("sampled boxes:", b8, "frame", (W, H))
    for mode in ("crop", "pad"):
        occ = target.occupancy(b8, W, H, mode)
        print(f"occupancy {mode}: nonzero cells per slice", (occ > 0).reshape(5, -1).sum(1).tolist(),
              "visible", target.visible_fraction(b8, W, H, mode))
    print("traj features", target.trajectory_features(vids[0]).shape)
    print("box text:", target.box_text(b8, W, H)[:90], "...")

    common = ["--device", "cpu", "--epochs", "1", "--max-steps", "3", "--grad-accum", "2", "--no-grad-ckpt",
              "--frame-mode", "pad"]
    for name, extra in [("tgc", ["--tgc"]), ("tgc_nozoom", ["--tgc", "--no-zoom"]), ("traj", ["--traj"]),
                        ("box_text", ["--box-text"]), ("box_draw", ["--box-draw"]),
                        ("rag", ["--rag-k", "3", "--rag-file", f"{FAKE}/rag.json"]),
                        ("tg_full", ["--tgc", "--traj", "--rag-k", "3", "--rag-file", f"{FAKE}/rag.json"])]:
        print(f"\n=== {name} ===")
        run(["--name", name] + extra + common)
        out = f"{FAKE}/out/runs/{name}/seed0/best"
        print("saved:", sorted(f for f in os.listdir(out) if f.endswith(".pt") or f.endswith(".json")))

    cfg = json.load(open(f"{FAKE}/out/runs/tg_full/seed0/run_config.json"))
    proc = {"video": functools.partial(fake_process_video, processor="clip", aspect_ratio=None, num_frames=8)}
    b = SampleBuilder(cfg, proc)
    v0, v1 = vids[0], vids[1]
    _, p_a, side_a = b.build(v0)
    _, p_b, side_b = b.build(v0, box_src=v1, traj_src=v1, rag_src=v1)
    print("\ncontrols: prompt changes with shuffled retrieval:", p_a != p_b,
          "| occupancy identical (same fake boxes):", torch.equal(side_a["occ"], side_b["occ"]))
    print("\n[ok] target-grounded smoke test passed")
