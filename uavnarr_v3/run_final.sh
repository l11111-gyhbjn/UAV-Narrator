#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
EXP="${EXP:-../uavnarr_exp}"
export UAVNARR_BERT_MODEL="${UAVNARR_BERT_MODEL:-/root/autodl-tmp/models/roberta-large}"
EVAL="python $EXP/eval_captions.py --refs data/test_refs.json --attrs data/test_attrs.json --bertscore"
OUT=$(python -c "from uavnarr import config; print(config.OUT_ROOT)")
RAG="--rag-k 3 --rag-file data/rag_neighbors.json"
declare -A ARGS=(
  [lora_pad]="--frame-mode pad"
  [box_text]="--frame-mode pad --box-text"
  [box_draw]="--frame-mode pad --box-draw"
  [tgc]="--frame-mode pad --tgc"
  [tgc_nozoom]="--frame-mode pad --tgc --no-zoom"
  [rag]="--frame-mode pad $RAG"
  [tgc_rag]="--frame-mode pad --tgc $RAG"
)
ORDER="tgc_rag tgc rag tgc_nozoom box_text box_draw lora_pad"
ALL="lora $ORDER"
ie () {
  local name=$1 seed=$2 run=$3; shift 3
  [ -f "preds/$name/seed$seed.json" ] || python infer.py --run "$run" --out "preds/$name/seed$seed.json" "$@"
  [ -f "results/$name/seed$seed.json" ] || $EVAL --pred "preds/$name/seed$seed.json" --out "results/$name/seed$seed.json"
}
case "${1:-}" in
train)
  for s in 0 1 2; do
    for c in $ORDER; do
      [ -f "$OUT/runs/$c/seed$s/best/adapter_config.json" ] || python train.py --name "$c" --seed "$s" ${ARGS[$c]}
    done
    rm -rf "$OUT"/runs/*/seed$s/epoch1 "$OUT"/runs/*/seed$s/epoch3
  done
  ;;
test)
  for s in 0 1 2; do
    for c in $ALL; do ie "$c" "$s" "runs/$c/seed$s"; done
    ie tgc_tshuf     "$s" "runs/tgc/seed$s"     --target-mode shuffled --shuffle-seed "$s"
    ie rag_rshuf     "$s" "runs/rag/seed$s"     --rag-mode shuffled    --shuffle-seed "$s"
    ie tgc_rag_tshuf "$s" "runs/tgc_rag/seed$s" --target-mode shuffled --shuffle-seed "$s"
    ie tgc_rag_rshuf "$s" "runs/tgc_rag/seed$s" --rag-mode shuffled    --shuffle-seed "$s"
  done
  ;;
report)
  python $EXP/aggregate_seeds.py --results results \
      --configs zeroshot $ALL tgc_tshuf rag_rshuf tgc_rag_tshuf tgc_rag_rshuf \
      --metrics B1 B2 B3 B4 METEOR R-L CIDEr F_BERT Cat --attr --out results/summary_final.json
  for pair in "tgc_rag lora" "tgc lora" "rag lora" "lora_pad lora" "box_text lora" "box_draw lora" \
              "tgc_rag box_text" "tgc_rag box_draw" "tgc box_text" "tgc box_draw" "tgc_rag tgc" "tgc_rag rag" \
              "tgc tgc_nozoom" "tgc_nozoom lora_pad" "tgc tgc_tshuf" "rag rag_rshuf" \
              "tgc_rag tgc_rag_tshuf" "tgc_rag tgc_rag_rshuf"; do
    set -- $pair
    python $EXP/paired_bootstrap.py --results results --a "$1" --b "$2" --out "results/bootF_${1}_vs_${2}.json"
  done
  for s in 0 1 2; do
    python $EXP/eval_retrieval.py --refs data/test_refs.json \
        --preds preds/zeroshot/seed0.json $(for c in $ALL; do echo "preds/$c/seed$s.json"; done) \
        --names zeroshot $ALL --out "results/retrievalF_seed$s.json" || echo "retrieval seed $s incomplete"
  done
  python - "$OUT" $ALL <<'EOF'
import json, os, sys, statistics
out, cfgs = sys.argv[1], sys.argv[2:]
print("\nbest validation loss / training time per video (3 seeds)")
for c in cfgs:
    vl, sp = [], []
    for s in (0, 1, 2):
        p = os.path.join(out, "runs", c, f"seed{s}", "train_log.jsonl")
        if os.path.exists(p):
            logs = [json.loads(l) for l in open(p)]
            b = min(logs, key=lambda d: d["val_loss"])
            vl.append(b["val_loss"]); sp.append(b["sec_per_train_video"])
    if vl:
        print(f"  {c:11s} val_loss {statistics.mean(vl):.4f}  s/video {statistics.mean(sp):.2f}  (n={len(vl)})")
EOF
  echo ">>> pack and send: tar -czf results_final.tar.gz results/"
  ;;
*) echo "usage: bash run_final.sh {train|test|report}"; exit 1;;
esac
