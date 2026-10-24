# Comparison with published models (all run on UAV-Narr)

| Paper row | Model (released checkpoint) | Setting | Script |
|---|---|---|---|
| Qwen2.5-VL-7B [Bai et al. 2025] | Qwen/Qwen2.5-VL-7B-Instruct | zero-shot / LoRA, 3 seeds | `hf_vlm.py --model qwen25vl` |
| LLaVA-OneVision-7B [Li et al. 2024] | llava-hf/llava-onevision-qwen2-7b-ov-hf | zero-shot / LoRA, 3 seeds | `hf_vlm.py --model llava_ov` |
| VideoLLaMA2-7B [Cheng et al. 2024] | existing runs `zeroshot`, `lora` | zero-shot / LoRA, 3 seeds | (main code) |
| ObjectMLLM-style [Tang et al. ICCV 2025] | existing runs `box_text` (+ optional `qwen25vl_boxtext_lora`) | LoRA, 3 seeds | (main code) / `run_baselines.sh boxtext` |
| BoxTuning-style [arXiv 2604.11136] | existing runs `box_draw` | LoRA, 3 seeds | (main code) |
| CLIP4Clip-meanP [Luo et al. 2022] | openai/clip-vit-base-patch32 | fine-tuned retrieval, 3 seeds | `clip4clip.py` |
| CLIP zero-shot [Radford et al. 2021] | ViT-B/32, ViT-L/14-336 | zero-shot retrieval | `clip4clip.py zeroshot` |

Identical for every captioning model: the 559/119/102 split, the 8 sampled frames, the prompt
(`config.STRUCTURED_PROMPT`), LoRA r16/alpha32 on q/k/v/o/gate/up/down of the LLM (vision encoder frozen),
AdamW 5e-5, wd 0.01, 3 epochs, cosine per epoch, grad-accum 8, best epoch by validation loss, greedy decoding,
max 100 new tokens, the same metric code (`uavnarr_exp/eval_captions.py`, run in the vl2 env).
Each model keeps its released image preprocessing (Qwen2.5-VL: native aspect ratio, <=256x28x28 px per frame;
LLaVA-OneVision: 384x384 per frame).

```bash
cd /root/autodl-tmp/uav_narr_code/uavnarr_v3
bash baselines/setup_env.sh          # ~5 min, new venv /root/autodl-tmp/envs/vlm (vl2 untouched)
bash baselines/download_models.sh    # ~33 GB to /root/autodl-tmp/models
bash baselines/run_baselines.sh smoke
nohup bash -c "bash baselines/run_baselines.sh zeroshot && bash baselines/run_baselines.sh clip && \
  bash baselines/run_baselines.sh train && bash baselines/run_baselines.sh test && \
  bash baselines/run_baselines.sh report" > /root/autodl-tmp/baselines.log 2>&1 &
```
Every stage skips finished work, so it can be restarted after an interruption.
