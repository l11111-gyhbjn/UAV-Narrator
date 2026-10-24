import json
import re
from pathlib import Path

BINARY = ["LR", "PO", "FO", "OV", "FM", "CM", "VC", "ROT", "DEF", "BC", "SV", "ARV", "IV", "MB"]
LEVELED = ["COM", "SIZ", "LEN"]
LEVEL_NAMES = {"COM": ["easy", "medium", "hard"],
               "SIZ": ["small", "medium", "large"],
               "LEN": ["short", "medium", "long"]}
SUBSETS = BINARY + [f"{k}_{lv}" for k in LEVELED for lv in LEVEL_NAMES[k]]
ATTRS = SUBSETS
FULL_NAMES = {
    "LR": "low resolution", "PO": "partial occlusion", "FO": "full occlusion", "OV": "out-of-view",
    "FM": "fast motion", "CM": "camera motion", "VC": "viewpoint change", "ROT": "rotation",
    "DEF": "deformation", "BC": "background clutter", "SV": "scale variation",
    "ARV": "aspect ratio variation", "IV": "illumination variation", "MB": "motion blur",
    "COM": "scene complexity", "SIZ": "target size", "LEN": "sequence length",
}

def subsets_of(d):
    out = [k for k in BINARY if int(d.get(k, 0)) == 1]
    for k in LEVELED:
        if k in d:
            out.append(f"{k}_{LEVEL_NAMES[k][int(d[k])]}")
    return out

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def as_ref_lists(refs):
    return {k: (v if isinstance(v, list) else [v]) for k, v in refs.items()}

def category_from_vid(vid):
    base = re.sub(r"_\d+$", "", vid)
    return base.replace("_", " ").strip().lower()

def _variants(term):
    term = term.lower().strip()
    out = {term}
    if not term.endswith("s"):
        out.add(term + "s")
    out.add(term.replace("-", " "))
    return out

def mentions_category(caption, category, synonyms=None):
    terms = set()
    for t in (synonyms or {}).get(category, [category]):
        terms |= _variants(t)
    text = " " + re.sub(r"[^a-z0-9\- ]", " ", caption.lower()) + " "
    text = re.sub(r"\s+", " ", text)
    return any(f" {t} " in text for t in terms)

def check_ids(preds, refs, name="predictions"):
    missing = sorted(set(refs) - set(preds))
    extra = sorted(set(preds) - set(refs))
    if missing:
        raise SystemExit(f"[error] {name} is missing {len(missing)} test videos, e.g. {missing[:5]}")
    if extra:
        print(f"[warn] {name} has {len(extra)} videos not in the references; they are ignored.")
