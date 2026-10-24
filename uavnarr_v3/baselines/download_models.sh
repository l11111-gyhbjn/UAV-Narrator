#!/usr/bin/env bash
set -euo pipefail
PY=${VLM_PY:-/root/autodl-tmp/envs/vlm/bin/python}
M=${UAVNARR_MODELS:-/root/autodl-tmp/models}
export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope HF_HOME=/root/autodl-tmp/.cache/hf
mkdir -p "$M"

complete () {
  "$PY" - "$1" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    need = set(json.load(open(idx))["weight_map"].values())
else:
    need = {"model.safetensors"} if os.path.exists(os.path.join(d, "model.safetensors")) else {"pytorch_model.bin"}
miss = [f for f in need if not os.path.exists(os.path.join(d, f))]
ok = os.path.exists(os.path.join(d, "config.json")) and not miss
print(("[ok] " if ok else "[incomplete] ") + d + ("" if ok else f" missing {sorted(miss)[:3]}"))
sys.exit(0 if ok else 1)
EOF
}

get () {
  local dir="$M/$3"
  complete "$dir" 2>/dev/null && return 0
  echo ">>> $3 from ModelScope ($1)"
  "$PY" -c "from modelscope import snapshot_download; snapshot_download('$1', local_dir='$dir')" || true
  if ! complete "$dir"; then
    echo ">>> $3 from hf-mirror.com ($2)"
    HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} "$PY" -c \
      "from huggingface_hub import snapshot_download; snapshot_download('$2', local_dir='$dir')"
    complete "$dir"
  fi
}

get Qwen/Qwen2.5-VL-7B-Instruct            Qwen/Qwen2.5-VL-7B-Instruct            Qwen2.5-VL-7B-Instruct
get llava-hf/llava-onevision-qwen2-7b-ov-hf llava-hf/llava-onevision-qwen2-7b-ov-hf llava-onevision-qwen2-7b-ov-hf
get openai-mirror/clip-vit-base-patch32     openai/clip-vit-base-patch32           clip-vit-base-patch32
du -sh "$M"/Qwen2.5-VL-7B-Instruct "$M"/llava-onevision-qwen2-7b-ov-hf "$M"/clip-vit-base-patch32
df -h /root/autodl-tmp | tail -1
