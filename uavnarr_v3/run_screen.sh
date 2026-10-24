#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
EXP="${EXP:-../uavnarr_exp}"
export UAVNARR_BERT_MODEL="${UAVNARR_BERT_MODEL:-/root/autodl-tmp/models/roberta-large}"
OUT=$(python -c "from uavnarr import config; print(config.OUT_ROOT)")

declare -A ARGS=(
  [lora_pad]="--frame-mode pad"
  [box_text]="--frame-mode pad --box-text"
  [box_draw]="--frame-mode pad --box-draw"
  [tgc]="--frame-mode pad --tgc"
  [traj]="--frame-mode pad --traj"
  [rag]="--frame-mode pad --rag-k 3 --rag-file data/rag_neighbors.json"
  [tg_full]="--frame-mode pad --tgc --traj --rag-k 3 --rag-file data/rag_neighbors.json"
)
ORDER="lora_pad tgc rag traj box_text box_draw tg_full"

case "${1:-}" in
prep)
  python tools/check_boxes.py
  python tools/export_eval_files.py --out-dir data --split val
  [ -f data/rag_neighbors.json ] || python tools/build_retrieval.py --out data/rag_neighbors.json
  ;;
train)
  for c in $ORDER; do
    [ -f "$OUT/runs/$c/seed0/best/adapter_config.json" ] || python train.py --name "$c" --seed 0 ${ARGS[$c]}
  done
  ;;
val)
  for c in lora $ORDER; do
    [ -f "preds_val/$c/seed0.json" ] || python infer.py --run "runs/$c/seed0" --split val --out "preds_val/$c/seed0.json"
    [ -f "results_val/$c/seed0.json" ] || python $EXP/eval_captions.py --pred "preds_val/$c/seed0.json" \
        --refs data/val_refs.json --attrs data/val_attrs.json --bertscore --out "results_val/$c/seed0.json"
  done
  ;;
report)
  python - "$OUT" lora $ORDER <<'EOF'
import json, os, sys
out, cfgs = sys.argv[1], sys.argv[2:]
base = json.load(open("results_val/lora_pad/seed0.json"))["corpus"]
print(f"{'config':10s} {'B4':>6s} {'MET':>6s} {'R-L':>6s} {'CIDEr':>7s} {'Cat':>6s} {'len':>5s} | "
      f"{'val_loss':>8s} {'s/video':>7s}  keep?")
for c in cfgs:
    p = f"results_val/{c}/seed0.json"
    if not os.path.exists(p):
        print(f"{c:10s} (missing)"); continue
    r = json.load(open(p))["corpus"]
    logs = [json.loads(l) for l in open(os.path.join(out, "runs", c, "seed0", "train_log.jsonl"))]
    best = min(logs, key=lambda d: d["val_loss"])
    keep = "" if c in ("lora", "lora_pad") else ("yes" if r["CIDEr"] >= base["CIDEr"] and r["Cat"] >= base["Cat"] else "no")
    print(f"{c:10s} {r['B4']:6.2f} {r['METEOR']:6.2f} {r['R-L']:6.2f} {r['CIDEr']:7.2f} {r['Cat']:6.2f} "
          f"{r['Len']:5.1f} | {best['val_loss']:8.4f} {best['sec_per_train_video']:7.2f}   {keep}")
EOF
  ;;
*) echo "usage: bash run_screen.sh {prep|train|val|report}"; exit 1;;
esac
