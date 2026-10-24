import json
import os
import re

from PIL import Image

from . import config

BINARY = ["LR", "PO", "FO", "OV", "FM", "CM", "VC", "ROT", "DEF", "BC", "SV", "ARV", "IV", "MB"]
LEVELED = ["COM", "SIZ", "LEN"]
LEVEL_NAMES = {"COM": ["easy", "medium", "hard"],
               "SIZ": ["small", "medium", "large"],
               "LEN": ["short", "medium", "long"]}
FULL_NAMES = {
    "LR": "low resolution", "PO": "partial occlusion", "FO": "full occlusion", "OV": "out-of-view",
    "FM": "fast motion", "CM": "camera motion", "VC": "viewpoint change", "ROT": "rotation",
    "DEF": "deformation", "BC": "background clutter", "SV": "scale variation",
    "ARV": "aspect ratio variation", "IV": "illumination variation", "MB": "motion blur",
    "COM": "scene complexity", "SIZ": "target size", "LEN": "sequence length",
}
OLD_NAMES = ["LR", "LI", "DEF", "MB", "SV", "ARC", "BC", "LO", "FM", "OV",
             "PO", "OCC", "VC", "LT", "OM", "SOB", "ROT"]
SUBSETS = BINARY + [f"{k}_{lv}" for k in LEVELED for lv in LEVEL_NAMES[k]]
ENCODING_DIM = {"onehot23": 23, "ordinal17": 17, "binary14": 14}

def load_split(split="test"):
    with open(config.SPLIT_PATH) as f:
        return json.load(f)[split]

def frame_indices(n_total, num_frames):
    if n_total <= num_frames:
        return list(range(n_total))
    step = n_total / num_frames
    return [int(i * step) for i in range(num_frames)]

def frame_names(video_name, num_frames=8):
    vdir = os.path.join(config.DATA_ROOT, video_name)
    manifest = os.path.join(vdir, "frames.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            all_frames = json.load(f)["all_frames"]
    else:
        all_frames = sorted(f for f in os.listdir(os.path.join(vdir, "img")) if f.endswith(".jpg"))
    return [all_frames[i] for i in frame_indices(len(all_frames), num_frames)]

def sample_frames(video_name, num_frames=8):
    img_dir = os.path.join(config.DATA_ROOT, video_name, "img")
    names = frame_names(video_name, num_frames)
    missing = [n for n in names if not os.path.exists(os.path.join(img_dir, n))]
    if missing:
        raise FileNotFoundError(f"{video_name}: frames {missing[:3]} not found; the compact copy was "
                                f"prepared without {num_frames}-frame sampling (see --num-frames)")
    return [Image.open(os.path.join(img_dir, n)).convert("RGB") for n in names]

def load_language(video_name):
    for path in (os.path.join(config.LANG_ROOT, video_name, "language.txt"),
                 os.path.join(config.DATA_ROOT, video_name, "language.txt")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                raise ValueError(f"{path} is empty")
            return text
    raise FileNotFoundError(f"language.txt of {video_name} not found under LANG_ROOT or DATA_ROOT")

def read_attributes_raw(video_name):
    path = os.path.join(config.DATA_ROOT, video_name, "attributes.txt")
    with open(path) as f:
        vals = [int(float(v)) for v in re.split(r"[,\s]+", f.read().strip()) if v != ""]
    if len(vals) != 17:
        raise ValueError(f"{path}: expected 17 values, found {len(vals)}")
    return vals

def attributes_dict(raw):
    d = {k: int(raw[i] >= 1) for i, k in enumerate(BINARY)}
    for j, k in enumerate(LEVELED):
        lv = int(raw[14 + j])
        if lv not in (0, 1, 2):
            raise ValueError(f"{k} level must be 0/1/2, got {lv}")
        d[k] = lv
    return d

def load_attributes(video_name):
    return attributes_dict(read_attributes_raw(video_name))

def encode_attributes(d, encoding="onehot23"):
    vec = [float(d[k]) for k in BINARY]
    if encoding == "binary14":
        return vec
    if encoding == "ordinal17":
        return vec + [d[k] / 2.0 for k in LEVELED]
    if encoding == "onehot23":
        for k in LEVELED:
            oh = [0.0, 0.0, 0.0]
            oh[d[k]] = 1.0
            vec += oh
        return vec
    raise ValueError(encoding)

def subsets_of(d):
    return [k for k in BINARY if d[k] == 1] + [f"{k}_{LEVEL_NAMES[k][d[k]]}" for k in LEVELED]

def attr_sentence(d):
    active = [FULL_NAMES[k] for k in BINARY if d[k] == 1]
    levels = [f"{FULL_NAMES[k]}: {LEVEL_NAMES[k][d[k]]}" for k in LEVELED]
    head = ", ".join(active) if active else "none"
    return f"Scenario attributes of this aerial video: {head}; " + "; ".join(levels) + "."

def category_from_vid(vid):
    return re.sub(r"_\d+$", "", vid).replace("_", " ").strip().lower()
