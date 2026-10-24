import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavnarr import data, target
from uavnarr.modules import TargetTokenCompressor

def main():
    tgc = TargetTokenCompressor(hidden_dim=8)
    bad = []
    for split in ("train", "val", "test"):
        st = defaultdict(list)
        for v in data.load_split(split):
            try:
                b8 = target.sampled_boxes(v, 8)
                W, H = target.frame_size(v)
            except Exception as e:
                bad.append(f"{split}/{v}: {e!r}")
                continue
            st["out_crop"].append(1 - target.visible_fraction(b8, W, H, "crop"))
            st["out_pad"].append(1 - target.visible_fraction(b8, W, H, "pad"))
            known = [b for b in b8 if b is not None]
            if known:
                st["rel_size"].append(np.mean([np.sqrt(b[2] * b[3] / (W * H)) for b in known]))
            st["absent"].append(sum(b is None for b in b8) / len(b8))
            occ = torch.from_numpy(target.occupancy(b8, W, H, "pad"))
            st["kept"].append(int(tgc.keep_mask(occ).sum()))
            st["res"].append(f"{W}x{H}")
        n = len(st["kept"])
        res = {r: st["res"].count(r) for r in set(st["res"])}
        print(f"{split:>5}: {n} videos | target centre outside the view: crop "
              f"{100 * np.nanmean(st['out_crop']):.1f}% / pad {100 * np.nanmean(st['out_pad']):.1f}% of frames"
              f" | videos affected by cropping: {sum(x > 0 for x in st['out_crop'])}"
              f" | mean target size {100 * np.mean(st['rel_size']):.1f}% of frame side"
              f" | absent {100 * np.mean(st['absent']):.1f}% | TGC target tokens kept {np.mean(st['kept']):.0f}/845"
              f" | resolutions {res}")
    if bad:
        print(f"\n[error] {len(bad)} videos:", *bad[:20], sep="\n  ")
        sys.exit(1)
    print("\n[ok] boxes load for all videos")

if __name__ == "__main__":
    main()
