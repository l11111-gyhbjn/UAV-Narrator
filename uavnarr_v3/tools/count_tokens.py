import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavnarr import data, target
from uavnarr.modules import TargetTokenCompressor

def main():
    res = {}
    for zoom in (False, True):
        tgc = TargetTokenCompressor(hidden_dim=8, use_zoom=zoom)
        for split in ("train", "val", "test"):
            n = []
            for v in data.load_split(split):
                b8 = target.sampled_boxes(v, 8)
                W, H = target.frame_size(v)
                occ = torch.from_numpy(target.occupancy(b8, W, H, "pad"))
                x = torch.zeros(1, 845, 8)
                z = torch.zeros(1, 845, 8) if zoom else None
                n.append(tgc(x, occ, z).shape[1])
            res[(zoom, split)] = n
            print(f"{'tgc' if zoom else 'tgc_nozoom':10s} {split:5s}: mean {np.mean(n):6.1f}  p50 {np.median(n):5.0f}  "
                  f"max {max(n):4d}  of 845 ({100 * np.mean(n) / 845:.1f}%)")

if __name__ == "__main__":
    main()
