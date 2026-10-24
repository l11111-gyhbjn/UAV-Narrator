#!/usr/bin/env bash
set -euo pipefail
VL2=${VL2:-/root/autodl-tmp/envs/vl2}
ENV=${VLM_ENV:-/root/autodl-tmp/envs/vlm}
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip TMPDIR=/root/autodl-tmp/.cache/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
[ -x "$ENV/bin/python" ] || "$VL2/bin/python" -m venv --system-site-packages "$ENV"
"$ENV/bin/python" -m pip install -q --upgrade pip
"$ENV/bin/python" -m pip install "transformers==4.51.3" "peft==0.15.2" "accelerate==1.6.0" "modelscope" \
    "huggingface_hub>=0.30,<1.0" "tokenizers>=0.21,<0.22"
"$ENV/bin/python" - <<'EOF'
import torch, transformers, peft, accelerate
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, CLIPModel
print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPU", torch.cuda.is_available() and torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "| peft", peft.__version__, "| accelerate", accelerate.__version__)
assert transformers.__version__ == "4.51.3"
print("[ok] vlm environment ready")
EOF
echo "vl2 unchanged:"; "$VL2/bin/python" -c "import transformers; print('  vl2 transformers', transformers.__version__)"
