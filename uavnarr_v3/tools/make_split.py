import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

def category(name):
    return re.sub(r"_\d+$", "", name).lower()

def usable(vdir, lang_root, name):
    has_lang = os.path.exists(os.path.join(vdir, "language.txt")) or (
        lang_root and os.path.exists(os.path.join(lang_root, name, "language.txt")))
    return os.path.isdir(os.path.join(vdir, "img")) and os.path.exists(os.path.join(vdir, "attributes.txt")) and has_lang

def list_split(src, sub, lang_root):
    d = os.path.join(src, sub)
    names = sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))
    ok = [n for n in names if usable(os.path.join(d, n), lang_root, n)]
    if len(ok) < len(names):
        print(f"[warn] {sub}: {len(names) - len(ok)} of {len(names)} folders lack img/, attributes.txt or language.txt")
    return ok

def stratified(names, n, seed):
    if n <= 0 or n >= len(names):
        return sorted(names)
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for x in names:
        by_cat[category(x)].append(x)
    for v in by_cat.values():
        rng.shuffle(v)
    quota = {c: len(v) * n / len(names) for c, v in by_cat.items()}
    take = {c: int(q) for c, q in quota.items()}
    order = sorted(by_cat, key=lambda c: (quota[c] - int(quota[c]), rng.random()), reverse=True)
    for c in order[:n - sum(take.values())]:
        take[c] += 1
    assert sum(take.values()) == n
    return sorted(x for c, v in by_cat.items() for x in v[:take[c]])

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    o = sp.add_parser("official")
    o.add_argument("--src", required=True, help="WebUAV-3M root containing Train/ Val/ Test/")
    o.add_argument("--lang-root", help="separate Language/ folder, if language.txt is not in the video folders")
    o.add_argument("--train-n", type=int, default=1500, help="0 = all training videos")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--out", required=True)
    t = sp.add_parser("from-test", help="re-split the WebUAV-3M Test videos (replaces the lost paper split)")
    t.add_argument("--src", required=True, help="folder that directly contains the Test video folders")
    t.add_argument("--lang-root")
    t.add_argument("--sizes", type=int, nargs=3, default=[559, 119, 102], metavar=("TRAIN", "VAL", "TEST"))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    k = sp.add_parser("keep")
    k.add_argument("--old", required=True)
    k.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.mode == "keep":
        split = json.load(open(a.old))
    elif a.mode == "from-test":
        d = a.src
        names = sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))
        vids = [n for n in names if usable(os.path.join(d, n), a.lang_root, n)]
        if len(vids) < len(names):
            print(f"[warn] {len(names) - len(vids)} folders lack img/, attributes.txt or language.txt")
        n_tr, n_va, n_te = a.sizes
        if n_tr + n_va + n_te != len(vids):
            raise SystemExit(f"[error] sizes sum to {n_tr + n_va + n_te}, but {len(vids)} usable videos were found")
        test = stratified(vids, n_te, a.seed)
        rest = sorted(set(vids) - set(test))
        val = stratified(rest, n_va, a.seed + 1)
        train = sorted(set(rest) - set(val))
        split = {"train": train, "val": val, "test": test,
                 "meta": {"source": "WebUAV-3M Test videos, re-split stratified by category",
                          "sizes": a.sizes, "seed": a.seed}}
    else:
        tr = list_split(a.src, "Train", a.lang_root)
        va = list_split(a.src, "Val", a.lang_root)
        te = list_split(a.src, "Test", a.lang_root)
        dup = (set(tr) & set(va)) | (set(tr) & set(te)) | (set(va) & set(te))
        if dup:
            raise SystemExit(f"[error] video names repeat across splits, e.g. {sorted(dup)[:5]}")
        split = {"train": stratified(tr, a.train_n, a.seed), "val": va, "test": te,
                 "meta": {"source": "WebUAV-3M official splits", "train_pool": len(tr),
                          "train_n": a.train_n, "seed": a.seed}}
    for s in ("train", "val", "test"):
        cats = Counter(category(x) for x in split[s])
        print(f"{s:>5}: {len(split[s]):5d} videos, {len(cats)} categories")
    overlap = set(split["train"]) & set(split["test"])
    if overlap:
        raise SystemExit(f"[error] {len(overlap)} videos are in both train and test")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(split, open(a.out, "w"), indent=1)
    print(f"[ok] {a.out}")

if __name__ == "__main__":
    main()
