import argparse
import json
import re

import numpy as np

from common import as_ref_lists, load_json, save_json

def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def ranks(S):
    c = S[np.arange(len(S)), np.arange(len(S))][:, None]
    return (S > c).sum(1) + ((S == c).sum(1) - 1) / 2.0 + 1.0

def summary(r):
    return {"R@1": 100 * float(np.mean(r <= 1)), "R@5": 100 * float(np.mean(r <= 5)),
            "R@10": 100 * float(np.mean(r <= 10)), "MdR": float(np.median(r)),
            "MnR": float(np.mean(r))}

def bm25_scores(queries, gallery):
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([tok(g) or ["_empty_"] for g in gallery])
    return np.stack([bm.get_scores(tok(q)) for q in queries])

def dense_scores(queries, gallery, model_path):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_path)
    q = m.encode(queries, normalize_embeddings=True)
    g = m.encode(gallery, normalize_embeddings=True)
    return q @ g.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--dense-model", help="local sentence-transformers model folder")
    ap.add_argument("--out")
    a = ap.parse_args()
    refs = as_ref_lists(load_json(a.refs))
    ids = sorted(refs)
    queries = [refs[k][0] for k in ids]
    names = a.names or a.preds
    out = {"n_queries": len(ids), "query": "reference caption", "methods": {}}
    head = f"{'captions from':16s} | BM25  R@1   R@5  R@10  MdR"
    print(head + ("   | Dense R@1   R@5  R@10  MdR" if a.dense_model else ""))
    for name, path in zip(names, a.preds):
        p = load_json(path)
        missing = [k for k in ids if k not in p]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} test videos missing, e.g. {missing[:3]}")
        gallery = [p[k] for k in ids]
        res = {"BM25": summary(ranks(bm25_scores(queries, gallery)))}
        line = f"{name:16s} | " + " ".join(f"{res['BM25'][m]:5.1f}" for m in ("R@1", "R@5", "R@10", "MdR"))
        if a.dense_model:
            res["Dense"] = summary(ranks(dense_scores(queries, gallery, a.dense_model)))
            line += "   | " + " ".join(f"{res['Dense'][m]:5.1f}" for m in ("R@1", "R@5", "R@10", "MdR"))
        out["methods"][name] = {"pred_file": path, **res}
        print(line)
    if a.out:
        save_json(out, a.out)

if __name__ == "__main__":
    main()
