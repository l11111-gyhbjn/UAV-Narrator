import argparse
import glob

import numpy as np

from common import load_json, save_json

def per_video_matrix(results_dir, cfg, metrics):
    runs = [load_json(f) for f in sorted(glob.glob(f"{results_dir}/{cfg}/seed*.json"))]
    if not runs:
        raise SystemExit(f"[error] no results for {cfg}")
    ids = sorted(runs[0]["per_video"])
    mats = {}
    for m in metrics:
        arr = np.array([[r["per_video"][k][m] for k in ids] for r in runs])
        mats[m] = arr.mean(axis=0)
    return ids, mats, len(runs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--a", required=True, help="configuration A (e.g. full)")
    ap.add_argument("--b", required=True, help="configuration B (e.g. msenc)")
    ap.add_argument("--metrics", nargs="+", default=["CIDEr", "Cat", "B4", "METEOR", "R-L"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    ids_a, A, na = per_video_matrix(a.results, a.a, a.metrics)
    ids_b, B, nb = per_video_matrix(a.results, a.b, a.metrics)
    if ids_a != ids_b:
        raise SystemExit("[error] the two configurations were evaluated on different videos")
    n = len(ids_a)
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, n, size=(a.n_boot, n))

    out = {"A": a.a, "B": a.b, "n_videos": n, "seeds": {a.a: na, a.b: nb}, "metrics": {}}
    print(f"{a.a} ({na} seeds) vs {a.b} ({nb} seeds), {n} videos, {a.n_boot} resamples")
    for m in a.metrics:
        d = A[m] - B[m]
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        out["metrics"][m] = {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p": float(min(p, 1.0))}
        flag = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        print(f"  {m:>7s}: diff {d.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p={min(p, 1.0):.3f}  ({flag})")
        print(f"           LaTeX: [{lo:+.2f}, {hi:+.2f}]")
    if a.out:
        save_json(out, a.out)

if __name__ == "__main__":
    main()
