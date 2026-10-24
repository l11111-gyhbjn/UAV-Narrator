import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavnarr import config, data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--open-images", action="store_true")
    a = ap.parse_args()
    print(f"DATA_ROOT={config.DATA_ROOT}\nSPLIT_PATH={config.SPLIT_PATH}")
    bad = []
    for split in ("train", "val", "test"):
        vids = data.load_split(split)
        levels = Counter()
        for v in vids:
            try:
                names = data.frame_names(v, a.num_frames)
                img_dir = os.path.join(config.DATA_ROOT, v, "img")
                miss = [n for n in names if not os.path.exists(os.path.join(img_dir, n))]
                if miss:
                    raise FileNotFoundError(f"{len(miss)} sampled frames missing")
                if a.open_images:
                    data.sample_frames(v, a.num_frames)
                d = data.load_attributes(v)
                levels.update(s for s in data.subsets_of(d))
                data.load_language(v)
            except Exception as e:
                bad.append(f"{split}/{v}: {e}")
        print(f"{split:>5}: {len(vids)} videos; attribute subsets: "
              + " ".join(f"{s}={levels.get(s, 0)}" for s in data.SUBSETS))
    if bad:
        print(f"\n[error] {len(bad)} videos have problems:")
        for b in bad[:30]:
            print("   ", b)
        sys.exit(1)
    print("\n[ok] all videos load correctly")

if __name__ == "__main__":
    main()
