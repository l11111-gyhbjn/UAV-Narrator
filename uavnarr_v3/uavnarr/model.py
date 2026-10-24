import sys
import types

import torch

from . import config

IGNORE_INDEX = -100
LLM_LORA_REGEX = (r"^model\.layers\.\d+\."
                  r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$")
VISION_LORA_REGEX = r"^model\.vision_tower\..*\.self_attn\.(q_proj|k_proj|v_proj)$"

def import_videollama2():
    if config.VL2_PATH not in sys.path:
        sys.path.insert(0, config.VL2_PATH)
    import videollama2
    from videollama2.mm_utils import tokenizer_multimodal_token
    from videollama2.constants import MODAL_INDEX_MAP, DEFAULT_VIDEO_TOKEN
    return videollama2, tokenizer_multimodal_token, MODAL_INDEX_MAP, DEFAULT_VIDEO_TOKEN

def load_base(model_path=None):
    videollama2, *_ = import_videollama2()
    from videollama2.utils import disable_torch_init
    disable_torch_init()
    model, processor, tokenizer = videollama2.model_init(model_path or config.MODEL_PATH,
                                                         attn_implementation="eager")
    return model, processor, tokenizer

def attach_modules(model, ms_enc=None, attr_cond=None, tgc=None, traj=None):
    model.uav_ms_enc = ms_enc
    model.uav_attr_cond = attr_cond
    model.uav_tgc = tgc
    model.uav_traj = traj
    model._attr_vec = None
    model._side = {}
    orig = type(model).temporal_aggregator

    def patched(self, frames_features):
        vf = orig(self, frames_features)
        dtype = vf.dtype
        x = vf.float()
        if self.uav_ms_enc is not None:
            x = self.uav_ms_enc(x)
        if self.uav_tgc is not None:
            side = self._side
            if "occ" not in side:
                raise RuntimeError("TGC is enabled but model._side['occ'] was not set")
            zf = None
            if self.uav_tgc.use_zoom and side.get("zoom") is not None:
                with torch.no_grad():
                    zoom = side["zoom"].to(frames_features.device, frames_features.dtype)
                    zfeat = self.get_model().get_vision_tower()(zoom)
                    zf = orig(self, zfeat.unsqueeze(0)).float()
            cs = cls_attention_cells(self) if self.uav_tgc.select == "attn" else None
            x = self.uav_tgc(x, side["occ"].to(x.device), zf, cs)
        if self.uav_traj is not None:
            if "traj" not in self._side:
                raise RuntimeError("trajectory tokens are enabled but model._side['traj'] was not set")
            tt = self.uav_traj(self._side["traj"].to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1))
            x = torch.cat([tt.to(x.dtype), x], dim=1)
        if self.uav_attr_cond is not None:
            if self._attr_vec is None:
                raise RuntimeError("Attr-Cond is enabled but model._attr_vec was not set")
            av = self._attr_vec
            if av.dim() == 1:
                av = av.unsqueeze(0).expand(x.shape[0], -1)
            x = self.uav_attr_cond(x, av.to(x.device, torch.float32))
        self._last_ntok = int(x.shape[1])
        return x.to(dtype)

    model.temporal_aggregator = types.MethodType(patched, model)
    if tgc is not None and tgc.select == "attn":
        stash_vision_inputs(model)
    return model

def stash_vision_inputs(model):
    tower = model.get_model().get_vision_tower()

    @torch.no_grad()
    def forward(self, images):
        outs = self.vision_tower(images, output_hidden_states=True)
        self._h_in = outs.hidden_states[self.select_layer - 1]
        return self.feature_select(outs).to(images.dtype)

    tower.forward = types.MethodType(forward, tower)

@torch.no_grad()
def cls_attention_cells(model, frames=8, slices=5, grid=13):
    tower = model.get_model().get_vision_tower()
    h = tower._h_in
    layers = tower.vision_tower.vision_model.encoder.layers
    layer = layers[len(layers) + tower.select_layer if tower.select_layer < 0 else tower.select_layer - 1]
    att = layer.self_attn
    hn = layer.layer_norm1(h)
    T, L, C = hn.shape
    H = att.num_heads
    q = (att.q_proj(hn[:, :1]) * att.scale).view(T, 1, H, -1).transpose(1, 2)
    k = att.k_proj(hn).view(T, L, H, -1).transpose(1, 2)
    a = torch.softmax((q @ k.transpose(-1, -2)).float(), -1)[:, :, 0, 1:].mean(1)
    side = int(round((L - 1) ** 0.5))
    a = a.view(T, side, side)
    idx = [[p for p in (2 * j - 1, 2 * j) if 0 <= p < side] for j in range(grid)]
    sl = [[f for f in (2 * i - 1, 2 * i) if 0 <= f < T] for i in range(slices)]
    out = torch.zeros(slices, grid, grid, device=a.device)
    for i, fs in enumerate(sl):
        af = a[fs].mean(0)
        for r, pr in enumerate(idx):
            for c, pc in enumerate(idx):
                out[i, r, c] = af[pr][:, pc].mean()
    return out

def lora_config(r=16, alpha=32, dropout=0.05, include_vision=False):
    from peft import LoraConfig, TaskType
    target = LLM_LORA_REGEX if not include_vision else f"({LLM_LORA_REGEX})|({VISION_LORA_REGEX})"
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout,
                      bias="none", target_modules=target)

def build_messages(prompt, caption=None, video_token="<video>"):
    msgs = [{"role": "system", "content": config.MISTRAL_SYSTEM},
            {"role": "user", "content": video_token + "\n" + prompt}]
    if caption is not None:
        msgs.append({"role": "assistant", "content": caption})
    return msgs

def encode_example(tokenizer, prompt, caption, tokenizer_multimodal_token, modal_index_map,
                   video_token="<video>"):
    instr = tokenizer.apply_chat_template(build_messages(prompt, None, video_token),
                                          tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(build_messages(prompt, caption, video_token),
                                         tokenize=False, add_generation_prompt=False)
    if not full.startswith(instr):
        raise RuntimeError("chat template: the prompt is not a prefix of the full conversation")
    ids = tokenizer_multimodal_token(full, tokenizer, video_token, return_tensors="pt")
    n_instr = len(tokenizer_multimodal_token(instr, tokenizer, video_token, return_tensors="pt"))
    vid_idx = modal_index_map[video_token]
    if int((ids == vid_idx).sum()) != 1:
        raise RuntimeError("expected exactly one <video> token in the training sequence")
    labels = ids.clone()
    labels[:n_instr] = IGNORE_INDEX
    labels[ids == vid_idx] = IGNORE_INDEX
    if int((labels != IGNORE_INDEX).sum()) == 0:
        raise RuntimeError("no caption tokens left after masking (check max length / template)")
    return ids, labels

def count_parameters(model):
    groups = {"lora_llm": 0, "lora_vision": 0, "ms_enc": 0, "attr_cond": 0, "tgc": 0, "traj": 0,
              "other_trainable": 0}
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if not p.requires_grad:
            continue
        if "lora_" in n:
            groups["lora_vision" if "vision_tower" in n else "lora_llm"] += p.numel()
        elif "uav_ms_enc" in n:
            groups["ms_enc"] += p.numel()
        elif "uav_attr_cond" in n:
            groups["attr_cond"] += p.numel()
        elif "uav_tgc" in n:
            groups["tgc"] += p.numel()
        elif "uav_traj" in n:
            groups["traj"] += p.numel()
        else:
            groups["other_trainable"] += p.numel()
    groups["trainable_total"] = sum(v for k, v in groups.items())
    groups["all_parameters"] = total
    return groups
