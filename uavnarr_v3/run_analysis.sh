#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
EXP="${EXP:-../uavnarr_exp}"
export UAVNARR_BERT_MODEL="${UAVNARR_BERT_MODEL:-/root/autodl-tmp/models/roberta-large}"
EVAL="python $EXP/eval_captions.py --refs data/test_refs.json --attrs data/test_attrs.json --bertscore"
LEVELS="0.05 0.1 0.2 0.3"

ie () {
  local name=$1 seed=$2 run=$3; shift 3
  [ -f "preds/$name/seed$seed.json" ] || python infer.py --run "$run" --out "preds/$name/seed$seed.json" "$@"
  [ -f "results/$name/seed$seed.json" ] || $EVAL --pred "preds/$name/seed$seed.json" --out "results/$name/seed$seed.json"
}

case "${1:-}" in
tokens)
  mkdir -p results
  python tools/count_tokens.py | tee results/visual_tokens.txt
  ;;
noise)
  for s in 0 1 2; do
    for l in $LEVELS; do
      ie "tgc_noise$l" "$s" "runs/tgc/seed$s" --box-noise "$l" --shuffle-seed "$s"
    done
  done
  ;;
bench)
  for c in lora tgc; do
    [ -f "bench/${c}_1gpu.json" ] || python tools/benchmark_serving.py --run "runs/$c/seed0" --gpus 0 --out "bench/${c}_1gpu.json"
  done
  ;;
bench_pipe)
  for c in lora tgc; do
    [ -f "bench/${c}_1gpu_pipe.json" ] || python tools/benchmark_serving.py --run "runs/$c/seed0" --gpus 0 \
        --prefetch 8 --cpu-workers 8 --out "bench/${c}_1gpu_pipe.json"
  done
  ;;
report)
  CFGS="lora tgc"; for l in $LEVELS; do CFGS="$CFGS tgc_noise$l"; done
  CFGS="$CFGS tgc_tshuf"
  python $EXP/aggregate_seeds.py --results results --configs $CFGS --metrics B4 METEOR R-L CIDEr F_BERT Cat \
      --out results/summary_noise.json
  for l in $LEVELS; do
    python $EXP/paired_bootstrap.py --results results --a "tgc_noise$l" --b lora --out "results/bootN_${l}_vs_lora.json" \
        | grep -E "vs|CIDEr|Cat" || true
  done
  python - $LEVELS <<'EOF'
import glob, json, os, sys, statistics as st
print("\nmean IoU of the perturbed boxes with the annotated boxes (over test videos and seeds)")
for l in sys.argv[1:]:
    v = [json.load(open(p))["box_iou_mean"] for p in sorted(glob.glob(f"preds/tgc_noise{l}/seed*.meta.json"))]
    v = [x for x in v if x is not None]
    if v:
        print(f"  sigma {l:5s}: IoU {st.mean(v):.3f}  (n={len(v)})")
print("\nserving benchmark (one replica, one video at a time)")
for p in sorted(glob.glob("bench/*_1gpu*.json")):
    s = json.load(open(p))["summary"]
    L = s["latency_ms"]
    print(f"  {os.path.basename(p):16s} {s['videos']} videos  {s['videos_per_gpu_hour']:.0f} videos/GPU-h  "
          f"caption p50 {L['caption_ms']['p50']:.0f} ms  p95 {L['caption_ms']['p95']:.0f} ms  "
          f"load p50 {L['load_ms']['p50']:.0f} ms  peak {max(s['peak_mem_gb_per_replica'].values()):.1f} GB  "
          f"words {s['words_mean']:.1f}  | {s.get('mode', 'sequential')}"
          + (f"  wait p50 {L['wait_ms']['p50']:.0f} ms" if 'wait_ms' in L else ""))
EOF
  echo ">>> pack and send: tar -czf results_analysis.tar.gz results/ bench/ preds/tgc_noise*/*.meta.json"
  ;;
*) echo "usage: bash run_analysis.sh {tokens|noise|bench|bench_pipe|report}"; exit 1;;
esac
