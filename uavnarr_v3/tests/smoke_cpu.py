import os
import re
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = os.path.abspath(sys.argv[1])
os.environ.update(UAVNARR_DATA_ROOT=f"{FAKE}/ds", UAVNARR_LANG_ROOT=f"{FAKE}/ds/Language",
                  UAVNARR_SPLIT=f"{FAKE}/split.json", UAVNARR_OUT=f"{FAKE}/out")
sys.path.insert(0, ROOT)

from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PretrainedConfig, PreTrainedTokenizerFast

import train
from uavnarr import config

VID = -201
D = 4096
TEMPLATE = ("{%- if messages[0]['role'] == 'system' %}{%- set system_message = messages[0]['content'] %}"
            "{%- set loop_messages = messages[1:] %}{%- else %}{%- set loop_messages = messages %}{%- endif %}"
            "{{- bos_token }}{%- for message in loop_messages %}{%- if message['role'] == 'user' %}"
            "{%- if loop.first and system_message is defined %}{{- ' [INST] ' + system_message + '\\n\\n' + "
            "message['content'] + ' [/INST]' }}{%- else %}{{- ' [INST] ' + message['content'] + ' [/INST]' }}"
            "{%- endif %}{%- elif message['role'] == 'assistant' %}{{- ' ' + message['content'] + eos_token}}"
            "{%- endif %}{%- endfor %}")

def make_tokenizer():
    text = open(os.path.join(ROOT, "uavnarr", "config.py")).read()
    for root, _, files in os.walk(f"{FAKE}/ds/Language"):
        for f in files:
            text += " " + open(os.path.join(root, f)).read()
    text += " Scenario attributes of this aerial video none ; : [INST] [/INST]"
    words = sorted(set(re.findall(r"\w+|[^\w\s]", text)))
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|[^\w\s]"), "removed", invert=True)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tok.chat_template = TEMPLATE
    return tok

def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token="<video>", return_tensors=None):
    chunks = [tokenizer(c, add_special_tokens=False).input_ids for c in prompt.split(multimodal_token)]
    ids = []
    for i in range(1, 2 * len(chunks)):
        ids.extend(chunks[i // 2]) if i % 2 == 1 else ids.append(VID)
    return torch.tensor(ids, dtype=torch.long)

class Attn(nn.Module):
    def __init__(self, d, kv):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, kv)
        self.v_proj, self.o_proj = nn.Linear(d, kv), nn.Linear(kv, d)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(d, 32), nn.Linear(d, 32), nn.Linear(32, d)

class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn, self.mlp = Attn(d, 64), MLP(d)

    def forward(self, h):
        a = self.self_attn
        v = torch.tanh(a.v_proj(h)) * torch.sigmoid(a.k_proj(h))
        ctx = v.cumsum(1) / torch.arange(1, v.shape[1] + 1, device=v.device).view(1, -1, 1)
        h = h + a.o_proj(ctx) + 0 * a.q_proj(h).mean()
        return h + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))

class FakeVL2(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.config = PretrainedConfig(use_cache=True, hidden_size=D)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, D)
        self.model.layers = nn.ModuleList([Layer(D) for _ in range(2)])
        vt = nn.Module(); vt.vision_tower = nn.Module(); vt.vision_tower.vision_model = nn.Module()
        enc = nn.Module(); enc.layers = nn.ModuleList([nn.Module()])
        enc.layers[0].self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(enc.layers[0].self_attn, n, nn.Linear(16, 16))
        vt.vision_tower.vision_model.encoder = enc
        self.model.vision_tower = vt
        self.model.mm_projector = nn.Linear(16, D)
        self.lm_head = nn.Linear(D, vocab)
        for p in self.parameters():
            p.requires_grad = False

    def temporal_aggregator(self, frames_features):
        x = frames_features[:, :5, :169, :]
        return self.model.mm_projector(x).reshape(x.shape[0], 5 * 169, D)

    def forward(self, input_ids, attention_mask=None, labels=None, images=None, **kw):
        video = images[0][0]
        feats = self.temporal_aggregator(video.reshape(1, 8, 1, -1)[..., :16].expand(1, 8, 169, 16))[0]
        ids, lab = input_ids[0], labels[0]
        pos = (ids == VID).nonzero()
        if len(pos) == 0:
            h = torch.cat([self.model.embed_tokens(ids), feats[0:0]], 0)
            new_lab = lab
        else:
            p = int(pos[0])
            h = torch.cat([self.model.embed_tokens(ids[:p]), feats, self.model.embed_tokens(ids[p + 1:])], 0)
            new_lab = torch.cat([lab[:p], torch.full((feats.shape[0],), -100), lab[p + 1:]])
        h = h.unsqueeze(0)
        for layer in self.model.layers:
            h = layer(h)
        logits = self.lm_head(h)
        loss = F.cross_entropy(logits[0, :-1], new_lab[1:], ignore_index=-100)
        return types.SimpleNamespace(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, *a, **k):
        return {}

    def gradient_checkpointing_enable(self, **kw):
        pass

def run(argv, old_style=False):
    tok = make_tokenizer()
    processor = {"video": lambda frames: torch.rand(8, 3, 4, 4)}
    train.import_videollama2 = lambda: (None, tokenizer_multimodal_token, {"<video>": VID}, "<video>")
    train.load_base = lambda: (FakeVL2(len(tok)), processor, tok)
    if old_style:
        def old_encode(tokenizer, prompt, caption, *a, **k):
            ids = torch.tensor(tokenizer(prompt + " " + caption).input_ids)
            return ids, ids.clone()
        train.encode_example = old_encode
    sys.argv = ["train.py"] + argv
    train.main()

if __name__ == "__main__":
    common = ["--device", "cpu", "--epochs", "1", "--max-steps", "4", "--grad-accum", "2", "--no-grad-ckpt"]
    print("=== corrected pipeline: full model ===")
    run(["--name", "full", "--msenc", "temporal", "--attrcond"] + common)
    out = f"{FAKE}/out/runs/full/seed0/best"
    print("saved:", sorted(os.listdir(out)))
    print("\n=== original input format (no <video> token) must be rejected ===")
    try:
        run(["--name", "old", "--msenc", "temporal", "--attrcond"] + common, old_style=True)
        print("ERROR: old-style input was not caught")
        sys.exit(1)
    except AssertionError as e:
        print("caught as expected:", e)
