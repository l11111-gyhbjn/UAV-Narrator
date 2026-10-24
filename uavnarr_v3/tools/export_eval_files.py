import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavnarr import data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--convert")
    ap.add_argument("--to")
    a = ap.parse_args()
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        vids = data.load_split(a.split)
        refs = {v: data.load_language(v) for v in vids}
        attrs = {v: data.load_attributes(v) for v in vids}
        json.dump(refs, open(os.path.join(a.out_dir, f"{a.split}_refs.json"), "w"), indent=2, ensure_ascii=False)
        json.dump(attrs, open(os.path.join(a.out_dir, f"{a.split}_attrs.json"), "w"), indent=2)
        print(f"[ok] {len(vids)} videos -> {a.out_dir}/{a.split}_refs.json, {a.split}_attrs.json")
    if a.convert:
        old = json.load(open(a.convert, encoding="utf-8"))
        preds = old["predictions"]
        os.makedirs(os.path.dirname(os.path.abspath(a.to)), exist_ok=True)
        json.dump(preds, open(a.to, "w"), indent=2, ensure_ascii=False)
        print(f"[ok] {len(preds)} captions -> {a.to}; old file reported errors: {len(old.get('errors', []))}")

if __name__ == "__main__":
    main()
