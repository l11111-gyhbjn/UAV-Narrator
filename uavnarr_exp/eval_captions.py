import argparse
import os
import re
import statistics

from common import (SUBSETS, as_ref_lists, category_from_vid, check_ids, load_json,
                    mentions_category, save_json, subsets_of)

def tokenize(gts, res, mode):
    if mode == "ptb":
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tok = PTBTokenizer()
        return (tok.tokenize({k: [{"caption": c} for c in v] for k, v in gts.items()}),
                tok.tokenize({k: [{"caption": c} for c in v] for k, v in res.items()}))
    if mode == "raw":
        return gts, res
    norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower()))
    return ({k: [norm(c) for c in v] for k, v in gts.items()},
            {k: [norm(c) for c in v] for k, v in res.items()})

def coco_scores(gts, res, meteor):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    ids = sorted(gts)
    gts = {k: gts[k] for k in ids}
    res = {k: res[k] for k in ids}
    corpus, per_video = {}, {k: {} for k in ids}
    scorers = [(Bleu(4), ["B1", "B2", "B3", "B4"]), (Rouge(), ["R-L"]), (Cider(), ["CIDEr"])]
    if meteor:
        from pycocoevalcap.meteor.meteor import Meteor
        scorers.insert(1, (Meteor(), ["METEOR"]))
    for scorer, names in scorers:
        score, scores = scorer.compute_score(gts, res)
        if len(names) == 1:
            score, scores = [score], [scores]
        for name, s, ss in zip(names, score, scores):
            corpus[name] = 100.0 * float(s)
            for k, v in zip(ids, ss):
                per_video[k][name] = 100.0 * float(v)
    return corpus, per_video

def bertscore(preds, refs, ids, model=None):
    from bert_score import score
    kw = dict(model_type=model, num_layers=17) if model else dict(lang="en")
    _, _, F = score([preds[k] for k in ids], [refs[k][0] for k in ids],
                    rescale_with_baseline=False, verbose=False, **kw)
    return 100.0 * float(F.mean()), {k: 100.0 * float(f) for k, f in zip(ids, F)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--attrs", help="official attributes of the test videos (export_eval_files.py)")
    ap.add_argument("--synonyms", help="category synonym list (JSON)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", choices=["ptb", "raw", "simple"], default="ptb")
    ap.add_argument("--attr-mode", choices=["subset", "mean"], default="subset")
    ap.add_argument("--bertscore", action="store_true", help="also compute BERTScore-F1")
    ap.add_argument("--bert-model", default=os.environ.get("UAVNARR_BERT_MODEL"),
                    help="local roberta-large folder (default: env UAVNARR_BERT_MODEL, else download)")
    ap.add_argument("--no-meteor", action="store_true")
    a = ap.parse_args()

    refs = as_ref_lists(load_json(a.refs))
    preds_raw = load_json(a.pred)
    check_ids(preds_raw, refs)
    ids = sorted(refs)
    preds = {k: [preds_raw[k].strip()] for k in ids}
    syn = load_json(a.synonyms) if a.synonyms else None

    g, r = tokenize(refs, preds, a.tokenizer)
    corpus, per_video = coco_scores(g, r, meteor=not a.no_meteor)

    if a.bertscore:
        corpus["F_BERT"], fb = bertscore({k: preds[k][0] for k in ids}, refs, ids, a.bert_model)
        for k in ids:
            per_video[k]["F_BERT"] = fb[k]

    for k in ids:
        cat = category_from_vid(k)
        per_video[k]["Cat"] = 100.0 if mentions_category(preds[k][0], cat, syn) else 0.0
        per_video[k]["Len"] = float(len(preds[k][0].split()))
    corpus["Cat"] = statistics.mean(per_video[k]["Cat"] for k in ids)
    corpus["Len"] = statistics.mean(per_video[k]["Len"] for k in ids)
    corpus["Tokens_total"] = sum(per_video[k]["Len"] for k in ids)

    per_attr = {}
    if a.attrs:
        attrs = load_json(a.attrs)
        members = {s: [k for k in ids if s in subsets_of(attrs[k])] for s in SUBSETS}
        for s, sub in members.items():
            if not sub:
                continue
            if a.attr_mode == "subset":
                from pycocoevalcap.cider.cider import Cider
                sc, _ = Cider().compute_score({k: g[k] for k in sub}, {k: r[k] for k in sub})
                c = 100.0 * float(sc)
            else:
                c = statistics.mean(per_video[k]["CIDEr"] for k in sub)
            per_attr[s] = {"N": len(sub), "CIDEr": c}

    out = {"pred_file": a.pred, "n_videos": len(ids), "corpus": corpus,
           "per_attribute": per_attr, "per_video": per_video,
           "settings": {"tokenizer": a.tokenizer, "meteor": not a.no_meteor, "attr_mode": a.attr_mode}}
    save_json(out, a.out)
    print(f"[ok] {a.pred} ({a.tokenizer}): {({k: round(v, 2) for k, v in corpus.items()})}")

if __name__ == "__main__":
    main()
