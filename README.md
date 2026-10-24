# UAV-Narrator

Code for the paper **Target-Level Captioning for UAV Videos by Tracker-Guided Token Compression**
(DASFAA 2027, Industry Track).

UAV-Narrator generates target-level captions for UAV videos from the frames and the boxes of the
tracked target. It builds on a frozen VideoLLaMA 2-7B and adds three components:

- **TGC** (target-guided token compression): keeps the token cells covered by the tracked box at full
  resolution and pools the background (845 -> about 135 visual tokens per video);
- **ZTV** (zoomed target view): encodes a crop around the box with the same frozen encoder
  (16 tokens per slice, about 215 visual tokens per video in total);
- **PEA** (parameter-efficient adaptation): LoRA on the language model plus 60 zero-initialized
  embeddings; the vision encoder, the connector and the decoder stay frozen.

All experiments use the public WebUAV-3M benchmark. DJI production data and the deployed service are
not part of this repository.

## Repository layout

```
uavnarr_v3/                 training, inference and experiment scripts
  uavnarr/                  model code: TGC and ZTV (modules.py, target.py), data loading, sampling
  train.py, infer.py
  run_base.sh               prerequisites: evaluation files, zero-shot run, LoRA baseline
  run_screen.sh             design decisions on the validation split (seed 0)
  run_final.sh              main results and ablations, 3 seeds
  run_budget.sh             box-free token reduction at the same token budget
  run_analysis.sh           token counts, simulated tracker errors, serving benchmark
  run_hybrid.sh             hybrid retrieval (caption index + CLIP4Clip, reciprocal rank fusion)
  baselines/                Qwen2.5-VL, LLaVA-OneVision, CLIP and CLIP4Clip on the same data
  tools/                    data split, dataset checks, retrieval, token counting, serving, case frames
  tests/                    developer smoke tests with a mock VideoLLaMA 2 on a synthetic dataset
uavnarr_exp/                evaluation: caption metrics, seed aggregation, paired bootstrap, retrieval
data/uav_narr_split.json    the 559 / 119 / 102 split (video names of the WebUAV-3M test set)
```

## Setup

**Environment for UAV-Narrator** (the one used in the paper: PyTorch 2.2.0, CUDA 12.1,
Transformers 4.40.0):

1. Clone VideoLLaMA 2 and install its requirements:
   `git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2`.
2. Download the checkpoint `DAMO-NLP-SG/VideoLLaMA2-7B`.
3. Install the evaluation dependencies: `pip install pycocoevalcap bert-score numpy scipy`.
   METEOR and the PTB tokenizer of `pycocoevalcap` need Java.

**Environment for the baselines** (Transformers 4.51.3, built on top of the first one):
`bash uavnarr_v3/baselines/setup_env.sh`, then `bash uavnarr_v3/baselines/download_models.sh`.

**Paths.** All paths are set by environment variables; the defaults in `uavnarr/config.py` and in the
shell scripts are those of our server and must be changed:

| Variable | Meaning |
|---|---|
| `UAVNARR_DATA_ROOT` | WebUAV-3M sequences, one folder per video with `img/`, `groundtruth_rect.txt`, `attributes.txt` |
| `UAVNARR_LANG_ROOT` | folder with `<video>/language.txt` if the language specifications are stored separately |
| `UAVNARR_SPLIT` | split file, e.g. `data/uav_narr_split.json` |
| `UAVNARR_MODEL` | VideoLLaMA2-7B checkpoint |
| `UAVNARR_VL2` | clone of the VideoLLaMA 2 repository |
| `UAVNARR_OUT` | output folder for runs |
| `UAVNARR_BERT_MODEL` | local `roberta-large` for BERTScore (optional; downloaded if unset) |
| `UAVNARR_MODELS`, `VLM_PY`, `EVAL_PY` | baseline checkpoints and the Python of each environment |

## Data

Download WebUAV-3M from its official repository (https://github.com/983632847/WebUAV-3M). UAV-Narr uses
the 780 sequences of the WebUAV-3M test set, re-split by category into 559 training, 119 validation and
102 test videos (`data/uav_narr_split.json`). The references are the target-level language
specifications of WebUAV-3M and the boxes are its annotations.

Only 8 frames per video are read. To avoid copying the full test set, a compact copy with the sampled
frames and all annotation files can be built and checked with

```bash
cd uavnarr_v3
python tools/prepare_compact_dataset.py --src /path/to/WebUAV-3M/Test --split ../data/uav_narr_split.json \
       --out /path/to/uavnarr_compact --num-frames 8
export UAVNARR_DATA_ROOT=/path/to/uavnarr_compact UAVNARR_SPLIT=$PWD/../data/uav_narr_split.json
python tools/check_dataset.py --open-images
python tools/check_boxes.py
```

## Reproducing the paper

Run all commands from `uavnarr_v3/` with the UAV-Narrator environment unless noted. Every script skips
finished runs, so it can be restarted after an interruption. Training one configuration takes about
30 minutes per seed on one RTX 4080 SUPER (32 GB).

```bash
bash run_base.sh prep            # evaluation files and retrieval memory
bash run_base.sh zeroshot        # zero-shot VideoLLaMA 2
bash run_base.sh lora            # LoRA baseline, 3 seeds
bash run_screen.sh prep && bash run_screen.sh train && bash run_screen.sh val && bash run_screen.sh report
bash run_final.sh train && bash run_final.sh test && bash run_final.sh report
bash run_budget.sh train && bash run_budget.sh test && bash run_budget.sh report
bash run_analysis.sh tokens && bash run_analysis.sh noise
bash run_analysis.sh bench && bash run_analysis.sh bench_pipe && bash run_analysis.sh report
bash baselines/run_baselines.sh zeroshot && bash baselines/run_baselines.sh train
bash baselines/run_baselines.sh test && bash baselines/run_baselines.sh clip && bash baselines/run_baselines.sh report
bash run_hybrid.sh
```

| Paper | Produced by | Configurations |
|---|---|---|
| Table 1, dataset statistics | `data/uav_narr_split.json` | |
| Table 2, captioning | `run_base.sh`, `run_final.sh`, `baselines/run_baselines.sh` | `zeroshot`, `lora`, `box_text`, `box_draw`, `tgc` (Ours), `qwen25vl_*`, `llava_ov_*` |
| Table 3, retrieval | `baselines/run_baselines.sh clip`, `run_hybrid.sh` | CLIP, CLIP4Clip, BM25 caption indexes, RRF fusion |
| Table 4 and Fig. 3, ablation | `run_final.sh`, `run_budget.sh` | `lora_pad`, `rag`, `tgc_nozoom`, `tgc_rag`, `tgc`, `budget_attn`, `budget_random`, `budget_uni5`, `budget_uni7` |
| Table 5, tracker errors | `run_final.sh test`, `run_analysis.sh noise` | `tgc_tshuf` (boxes of another video), `tgc_noise{0.05,0.1,0.2,0.3}` |
| Table 6, case study | `tools/export_case_frames.py` and the predictions in `preds/` | |
| Table 7, serving cost | `run_analysis.sh bench`, `bench_pipe`; baselines from `baselines/run_baselines.sh` | |

The final model is `tgc`, i.e. TGC + ZTV with LoRA on padded frames (`train.py --frame-mode pad --tgc`).
Metrics are reported as mean and standard deviation over three seeds, and differences with 95% confidence
intervals from a paired bootstrap over test videos (`uavnarr_exp/paired_bootstrap.py`, 10,000 resamples).

## Citation

```bibtex
@inproceedings{tong2027uavnarrator,
  title     = {Target-Level Captioning for {UAV} Videos by Tracker-Guided Token Compression},
  author    = {Tong, Jingcheng and Fan, Miao and Gao, Bo and Ma, Chen and Li, Guohong},
  booktitle = {Database Systems for Advanced Applications (DASFAA)},
  year      = {2027}
}
```
