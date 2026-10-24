#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VLM_PY=${VLM_PY:-/root/autodl-tmp/envs/vlm/bin/python}
EVAL_PY=${EVAL_PY:-/root/autodl-tmp/envs/vl2/bin/python}
EXP=${EXP:-../uavnarr_exp}
M=${UAVNARR_MODELS:-/root/autodl-tmp/models}
export UAVNARR_BERT_MODEL="${UAVNARR_BERT_MODEL:-/root/autodl-tmp/models/roberta-large}"
export TOKENIZERS_PARALLELISM=false
OUT=$($EVAL_PY -c "from uavnarr import config; print(config.OUT_ROOT)")
EVAL="$EVAL_PY $EXP/eval_captions.py --refs data/test_refs.json --attrs data/test_attrs.json --bertscore"
HF="$VLM_PY baselines/hf_vlm.py"
MODELS="qwen25vl llava_ov"
RB=results/retrieval_baselines

ev () {
  [ -f "results/$1/seed$2.json" ] || $EVAL --pred "preds/$1/seed$2.json" --out "results/$1/seed$2.json"
}
trained () { [ -f "$OUT/baselines/$1/seed$2/done.txt" ]; }

case "${1:-}" in
smoke)
  for m in $MODELS; do
    $HF train --model "$m" --name "smoke_$m" --epochs 1 --max-steps 4 --grad-accum 2
    $HF infer --model "$m" --run "baselines/smoke_$m/seed0" --limit 3 --out "$OUT/smoke/${m}_lora.json"
    $HF infer --model "$m" --zeroshot --limit 3 --out "$OUT/smoke/${m}_zs.json"
    rm -rf "$OUT/baselines/smoke_$m"
  done
  grep -h '"summary"' -A 6 "$OUT"/smoke/*.meta.json | grep -E "visual_tokens_mean|generate_ms_p50" || true
  echo "[ok] smoke test passed; captions in $OUT/smoke/"
  ;;
zeroshot)
  for m in $MODELS; do
    [ -f "preds/${m}_zs/seed0.json" ] || $HF infer --model "$m" --zeroshot --out "preds/${m}_zs/seed0.json"
    ev "${m}_zs" 0
  done
  ;;
train)
  for s in 0 1 2; do
    for m in $MODELS; do
      trained "${m}_lora" "$s" || $HF train --model "$m" --seed "$s"
    done
  done
  ;;
test)
  for s in 0 1 2; do
    for m in $MODELS; do
      [ -f "preds/${m}_lora/seed$s.json" ] || $HF infer --model "$m" --run "baselines/${m}_lora/seed$s" \
          --out "preds/${m}_lora/seed$s.json"
      ev "${m}_lora" "$s"
    done
  done
  ;;
boxtext)
  for s in 0 1 2; do
    trained qwen25vl_boxtext_lora "$s" || $HF train --model qwen25vl --box-text --seed "$s"
    [ -f "preds/qwen25vl_boxtext_lora/seed$s.json" ] || $HF infer --model qwen25vl \
        --run "baselines/qwen25vl_boxtext_lora/seed$s" --out "preds/qwen25vl_boxtext_lora/seed$s.json"
    ev qwen25vl_boxtext_lora "$s"
  done
  ;;
clip)
  B32="$M/clip-vit-base-patch32"; L14="$M/clip-vit-large-patch14-336"
  C4C="$VLM_PY baselines/clip4clip.py"
  [ -f $RB/clip_b32_zs.json ] || $C4C zeroshot --clip "$B32" --out $RB/clip_b32_zs.json
  [ -f $RB/clip_l14_336_zs.json ] || $C4C zeroshot --clip "$L14" --out $RB/clip_l14_336_zs.json
  [ -f $RB/clip4clip_lr.json ] || $C4C select --clip "$B32" --out $RB/clip4clip_lr.json
  LR=$($VLM_PY -c "import json; print(json.load(open('$RB/clip4clip_lr.json'))['chosen_lr'])")
  echo "CLIP4Clip learning rate chosen on validation: $LR"
  for s in 0 1 2; do
    [ -f $RB/clip4clip/seed$s.json ] || $C4C train --clip "$B32" --lr "$LR" --seed "$s" --out $RB/clip4clip/seed$s.json
  done
  ;;
report)
  OURS="zeroshot lora box_text box_draw tgc tgc_rag"
  BASE="qwen25vl_zs llava_ov_zs qwen25vl_lora llava_ov_lora qwen25vl_boxtext_lora"
  CFGS=""
  for c in $OURS $BASE; do if [ -d "results/$c" ]; then CFGS="$CFGS $c"; fi; done
  ($EVAL_PY $EXP/aggregate_seeds.py --results results --configs $CFGS \
      --metrics B1 B2 B3 B4 METEOR R-L CIDEr F_BERT Cat --out results/summary_baselines.json)
  for b in qwen25vl_lora llava_ov_lora qwen25vl_zs llava_ov_zs qwen25vl_boxtext_lora; do
    [ -d "results/$b" ] || continue
    for a in tgc lora; do
      $EVAL_PY $EXP/paired_bootstrap.py --results results --a "$a" --b "$b" --out "results/bootB_${a}_vs_${b}.json" \
          | grep -E "CIDEr|Cat|vs" || true
    done
  done
  for s in 0 1 2; do
    names=(); files=()
    for c in qwen25vl_zs llava_ov_zs; do
      if [ -f "preds/$c/seed0.json" ]; then names+=("$c"); files+=("preds/$c/seed0.json"); fi
    done
    for c in lora tgc tgc_rag qwen25vl_lora llava_ov_lora qwen25vl_boxtext_lora; do
      if [ -f "preds/$c/seed$s.json" ]; then names+=("$c"); files+=("preds/$c/seed$s.json"); fi
    done
    $EVAL_PY $EXP/eval_retrieval.py --refs data/test_refs.json --preds "${files[@]}" --names "${names[@]}" \
        --out "results/retrievalB_seed$s.json" > /dev/null
  done
  OUT="$OUT" $EVAL_PY - <<'EOF'
import glob, json, os, statistics as st
def ms(xs):
    return f"{st.mean(xs):5.1f}" + (f"±{st.stdev(xs):.1f}" if len(xs) > 1 else "     ")
print("\n=== text-to-video retrieval on the 102 test videos (query = reference caption) ===")
print(f"{'method':34s} {'R@1':>9s} {'R@5':>9s} {'R@10':>9s} {'MdR':>9s}  seeds")
rows = {}
for f in sorted(glob.glob("results/retrievalB_seed*.json")):
    for name, r in json.load(open(f))["methods"].items():
        rows.setdefault(f"BM25 over captions: {name}", []).append(r["BM25"])
for f in ("clip_b32_zs", "clip_l14_336_zs"):
    p = f"results/retrieval_baselines/{f}.json"
    if os.path.exists(p):
        rows[f"CLIP zero-shot ({f.split('_')[1]})"] = [json.load(open(p))["test"]]
c4c = [json.load(open(p))["test"] for p in sorted(glob.glob("results/retrieval_baselines/clip4clip/seed*.json"))]
if c4c:
    rows["CLIP4Clip-meanP (B/32, fine-tuned)"] = c4c
for name, rs in rows.items():
    if "_zs" in name and len(rs) > 1:
        rs = rs[:1]
    print(f"{name:34s} " + " ".join(ms([r[k] for r in rs]).rjust(9) for k in ("R@1", "R@5", "R@10", "MdR"))
          + f"  {len(rs)}")
print("\n=== visual tokens / generation latency of the baselines (from preds/*/seed0.meta.json) ===")
for p in sorted(glob.glob("preds/*/seed0.meta.json")):
    s = json.load(open(p)).get("summary")
    if s:
        print(f"  {p.split('/')[1]:24s} visual tokens {s['visual_tokens_mean']:7.1f}  "
              f"generate p50 {s['generate_ms_p50']:7.0f} ms  ({s['gpu']})")
print("\nbest validation loss / training time per video")
out = os.environ["OUT"]
for run in sorted(glob.glob(f"{out}/baselines/*_lora")):
    vl, sp, mem = [], [], []
    for s in (0, 1, 2):
        p = f"{run}/seed{s}/train_log.jsonl"
        if os.path.exists(p):
            logs = [json.loads(l) for l in open(p)]
            b = min(logs, key=lambda d: d["val_loss"]); vl.append(b["val_loss"]); sp.append(b["sec_per_train_video"])
            mem.append(max(d["peak_mem_gb"] for d in logs))
    if vl:
        cfg = json.load(open(f"{run}/seed0/run_config.json"))
        pc = cfg.get("param_counts", {})
        print(f"  {os.path.basename(run):24s} val_loss {st.mean(vl):.4f}  s/video {st.mean(sp):.2f}  "
              f"peak {max(mem):.1f} GB  trainable {pc.get('trainable', 0) / 1e6:.2f}M  (n={len(vl)})")
EOF
  echo ">>> pack and send: tar -czf results_baselines.tar.gz results/ preds/*/seed0.meta.json"
  ;;
*) echo "usage: bash baselines/run_baselines.sh {smoke|zeroshot|train|test|clip|report|boxtext}"; exit 1;;
esac
