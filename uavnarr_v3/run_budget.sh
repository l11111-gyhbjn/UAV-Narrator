#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
EXP="${EXP:-../uavnarr_exp}"
export UAVNARR_BERT_MODEL="${UAVNARR_BERT_MODEL:-/root/autodl-tmp/models/roberta-large}"
EVAL="python $EXP/eval_captions.py --refs data/test_refs.json --attrs data/test_attrs.json --bertscore"
OUT=$(python -c "from uavnarr import config; print(config.OUT_ROOT)")
B="--frame-mode pad --tgc --no-zoom"
declare -A ARGS=(
  [budget_attn]="$B --tgc-select attn"
  [budget_random]="$B --tgc-select random"
  [budget_uni5]="$B --tgc-select uniform --uniform-grid 5"
  [budget_uni7]="$B --tgc-select uniform --uniform-grid 7"
)
ORDER="budget_attn budget_uni5 budget_random budget_uni7"

case "${1:-}" in
train)
  for s in 0 1 2; do
    for c in $ORDER; do
      [ -f "$OUT/runs/$c/seed$s/best/adapter_config.json" ] || python train.py --name "$c" --seed "$s" ${ARGS[$c]}
      rm -rf "$OUT/runs/$c/seed$s/epoch1" "$OUT/runs/$c/seed$s/epoch2" "$OUT/runs/$c/seed$s/epoch3"
    done
  done
  ;;
test)
  for s in 0 1 2; do
    for c in $ORDER; do
      [ -f "preds/$c/seed$s.json" ] || python infer.py --run "runs/$c/seed$s" --out "preds/$c/seed$s.json"
      [ -f "results/$c/seed$s.json" ] || $EVAL --pred "preds/$c/seed$s.json" --out "results/$c/seed$s.json"
    done
  done
  ;;
report)
  python $EXP/aggregate_seeds.py --results results --configs lora_pad $ORDER tgc_nozoom tgc \
      --metrics B4 METEOR R-L CIDEr F_BERT Cat --out results/summary_budget.json
  for b in budget_attn budget_random budget_uni5; do
    python $EXP/paired_bootstrap.py --results results --a tgc_nozoom --b "$b" --out "results/bootM_tgc_nozoom_vs_$b.json" \
        | grep -E "vs|CIDEr|Cat" || true
  done
  for b in budget_uni7 budget_attn; do
    python $EXP/paired_bootstrap.py --results results --a tgc --b "$b" --out "results/bootM_tgc_vs_$b.json" \
        | grep -E "vs|CIDEr|Cat" || true
  done
  for b in $ORDER; do
    python $EXP/paired_bootstrap.py --results results --a "$b" --b lora_pad --out "results/bootM_${b}_vs_lora_pad.json" \
        | grep -E "vs|CIDEr|Cat" || true
  done
  python - "$OUT" $ORDER tgc_nozoom tgc lora_pad <<'EOF'
import glob, json, os, sys, statistics as st
out, cfgs = sys.argv[1], sys.argv[2:]
print("\nvisual tokens (test, from preds/*.meta.json) / best val loss / training s per video")
for c in cfgs:
    nt = [json.load(open(p)).get("visual_tokens_mean") for p in sorted(glob.glob(f"preds/{c}/seed*.meta.json"))]
    nt = [x for x in nt if x]
    vl, sp = [], []
    for s in (0, 1, 2):
        p = os.path.join(out, "runs", c, f"seed{s}", "train_log.jsonl")
        if os.path.exists(p):
            logs = [json.loads(l) for l in open(p)]
            b = min(logs, key=lambda d: d["val_loss"]); vl.append(b["val_loss"]); sp.append(b["sec_per_train_video"])
    print(f"  {c:14s} tokens {st.mean(nt) if nt else float('nan'):6.1f}  "
          + (f"val_loss {st.mean(vl):.4f}  s/video {st.mean(sp):.2f}  (n={len(vl)})" if vl else ""))
EOF
  echo ">>> send: tar -czf results_budget.tar.gz results/summary_budget.json results/bootM_* results/budget_* preds/budget_*/*.meta.json"
  ;;
*) echo "usage: bash run_budget.sh {train|test|report}"; exit 1;;
esac
