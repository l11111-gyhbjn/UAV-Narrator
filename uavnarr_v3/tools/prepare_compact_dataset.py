import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavnarr.data import frame_indices

ANNOT = ["attributes.txt", "groundtruth_rect.txt", "absent.txt", "scenario.txt", "language.txt"]

def find_video(src, name):
    for sub in ("", "Test", "Val", "Train"):
        d = os.path.join(src, sub, name)
        if os.path.isdir(os.path.join(d, "img")):
            return d
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang-root")
    ap.add_argument("--num-frames", type=int, nargs="+", default=[8, 16])
    a = ap.parse_args()

    split = json.load(open(a.split))
    videos = [v for s in ("train", "val", "test") for v in split[s]]
    total_bytes, problems = 0, []
    for i, v in enumerate(videos, 1):
        d = find_video(a.src, v)
        if d is None:
            problems.append(f"{v}: folder with img/ not found")
            continue
        all_frames = sorted(f for f in os.listdir(os.path.join(d, "img")) if f.endswith(".jpg"))
        keep = sorted({all_frames[j] for n in a.num_frames for j in frame_indices(len(all_frames), n)})
        od = os.path.join(a.out, v)
        os.makedirs(os.path.join(od, "img"), exist_ok=True)
        for f in keep:
            dst = os.path.join(od, "img", f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(d, "img", f), dst)
            total_bytes += os.path.getsize(dst)
        for f in ANNOT:
            p = os.path.join(d, f)
            if f == "language.txt" and not os.path.exists(p) and a.lang_root:
                p = os.path.join(a.lang_root, v, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(od, f))
            elif f in ("attributes.txt", "language.txt"):
                problems.append(f"{v}: {f} missing")
        json.dump({"all_frames": all_frames, "kept": keep, "num_frames": a.num_frames},
                  open(os.path.join(od, "frames.json"), "w"))
        if i % 100 == 0 or i == len(videos):
            print(f"  {i}/{len(videos)} videos, {total_bytes / 1e9:.2f} GB")
    shutil.copy2(a.split, os.path.join(a.out, os.path.basename(a.split)))
    print(f"[ok] {len(videos) - len(problems)} videos -> {a.out} ({total_bytes / 1e9:.2f} GB of frames)")
    if problems:
        print(f"[warn] {len(problems)} problems:")
        for p in problems[:20]:
            print("   ", p)

if __name__ == "__main__":
    main()
