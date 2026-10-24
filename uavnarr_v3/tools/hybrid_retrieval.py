import argparse
import json
import os
import statistics as st
import sys

import numpy as np

def rank_matrix(S):
    R = np.empty_like(S, dtype=float)
    for q in range(len(S)):
        s = S[q]
        R[q] = (s[None, :] > s[:, None]).sum(1) + ((s[None, :] == s[:, None]).sum(1) - 1) / 2.0 + 1.0
    return R

def rrf(*mats, k=60.0):
    return sum(1.0 / (k + rank_matrix(S)) for S in mats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--exp", default="../uavnarr_exp")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--captions", nargs="+", required=True, help="caption configurations (preds/<name>/seed<s>.json)")
    ap.add_argument("--emb", nargs="+", required=True, help="name=path.npz, {s} is replaced by the seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=float, default=60.0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.exp))
    from common import as_ref_lists, load_json
    from eval_retrieval import bm25_scores, ranks, summary

    refs = as_ref_lists(load_json(a.refs))
    ids = sorted(refs)
    queries = [refs[k][0] for k in ids]
    embs = dict(e.split("=", 1) for e in a.emb)

    def emb_S(name, s):
        z = np.load(embs[name].replace("{s}", str(s)))
        assert list(z["test_ids"]) == ids, f"{name}: video order differs from the references"
        return z["test_S"]

    def cap_S(name, s):
        p = load_json(os.path.join(a.preds, name, f"seed{s}.json"))
        return bm25_scores(queries, [p[k] for k in ids])

    hits = {}
    for s in a.seeds:
        E = {n: emb_S(n, s) for n in embs}
        C = {n: cap_S(n, s) for n in a.captions}
        for n, S in E.items():
            hits.setdefault(f"{n}", []).append(ranks(S))
        for n, S in C.items():
            hits.setdefault(f"BM25[{n}]", []).append(ranks(S))
            for e, SE in E.items():
                hits.setdefault(f"RRF(BM25[{n}], {e})", []).append(ranks(rrf(S, SE, k=a.k)))

    out = {"protocol": "query = reference caption, gallery = 102 test videos", "rrf_k": a.k, "methods": {}}
    print(f"{'method':40s} {'R@1':>10s} {'R@5':>10s} {'R@10':>10s} {'MdR':>6s}")
    for n, rs in hits.items():
        per_seed = [summary(r) for r in rs]
        agg = {m: (st.mean(x[m] for x in per_seed), st.stdev(x[m] for x in per_seed) if len(per_seed) > 1 else 0.0)
               for m in per_seed[0]}
        out["methods"][n] = {"mean_std": agg, "per_seed": per_seed}
        print(f"{n:40s} " + " ".join(f"{agg[m][0]:5.1f}±{agg[m][1]:3.1f}" for m in ("R@1", "R@5", "R@10"))
              + f" {agg['MdR'][0]:6.1f}")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(a.n_boot, len(ids)))
    out["bootstrap"] = {}
    print(f"\npaired bootstrap over the {len(ids)} queries (difference in percentage points, 95% CI)")
    for n in a.captions:
        for e in embs:
            f = f"RRF(BM25[{n}], {e})"
            for b in (f"BM25[{n}]", e):
                res = {}
                for K in (1, 5, 10):
                    d = (np.mean([r <= K for r in hits[f]], 0) - np.mean([r <= K for r in hits[b]], 0)) * 100
                    bs = d[idx].mean(1)
                    res[f"R@{K}"] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                    float(np.percentile(bs, 97.5))]}
                out["bootstrap"][f"{f} vs {b}"] = res
                print(f"  {f} vs {b}: " + "  ".join(
                    f"{k}: {v['diff']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]" for k, v in res.items()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[ok] -> {a.out}")

if __name__ == "__main__":
    main()
