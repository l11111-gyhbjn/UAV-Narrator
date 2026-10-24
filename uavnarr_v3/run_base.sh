#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
EXP="${EXP:-../uavnarr_exp}"
EVAL="python $EXP/eval_captions.py --refs data/test_refs.json --attrs data/test_attrs.json --bertscore"
OUT=$(python -c "from uavnarr import config; print(config.OUT_ROOT)")

case "${1:-}" in
prep)
  python tools/export_eval_files.py --out-dir data
  python tools/export_eval_files.py --out-dir data --split val
  [ -f data/rag_neighbors.json ] || python tools/build_retrieval.py --out data/rag_neighbors.json
  ;;
zeroshot)
  [ -f preds/zeroshot/seed0.json ] || python infer.py --zeroshot --out preds/zeroshot/seed0.json
  [ -f results/zeroshot/seed0.json ] || $EVAL --pred preds/zeroshot/seed0.json --out results/zeroshot/seed0.json
  ;;
lora)
  for s in 0 1 2; do
    [ -f "$OUT/runs/lora/seed$s/best/adapter_config.json" ] || python train.py --name lora --seed "$s"
    [ -f "preds/lora/seed$s.json" ] || python infer.py --run "runs/lora/seed$s" --out "preds/lora/seed$s.json"
    [ -f "results/lora/seed$s.json" ] || $EVAL --pred "preds/lora/seed$s.json" --out "results/lora/seed$s.json"
  done
  ;;
*) echo "usage: bash run_base.sh {prep|zeroshot|lora}"; exit 1;;
esac
