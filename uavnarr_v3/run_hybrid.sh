#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VLM_PY=${VLM_PY:-/root/autodl-tmp/envs/vlm/bin/python}
EXP=${EXP:-../uavnarr_exp}
M=${UAVNARR_MODELS:-/root/autodl-tmp/models}
H=results/retrieval_hybrid
C4C="$VLM_PY baselines/clip4clip.py"
LR=$(python -c "import json; print(json.load(open('results/retrieval_baselines/clip4clip_lr.json'))['chosen_lr'])")
echo "CLIP4Clip lr (chosen on validation earlier): $LR"

[ -f $H/clip_l14_zs.npz ] || $C4C zeroshot --clip "$M/clip-vit-large-patch14-336" --out $H/clip_l14_zs.json --scores $H/clip_l14_zs.npz
for s in 0 1 2; do
  [ -f $H/clip4clip_seed$s.npz ] || $C4C train --clip "$M/clip-vit-base-patch32" --lr "$LR" --seed "$s" \
      --out $H/clip4clip_seed$s.json --scores $H/clip4clip_seed$s.npz
done
python tools/hybrid_retrieval.py --refs data/test_refs.json --exp "$EXP" \
    --captions lora qwen25vl_lora llava_ov_lora tgc \
    --emb "clip4clip=$H/clip4clip_seed{s}.npz" "clip_l14_zs=$H/clip_l14_zs.npz" \
    --out $H/summary.json | tee $H/summary.txt
echo ">>> send: tar -czf results_hybrid.tar.gz $H"
