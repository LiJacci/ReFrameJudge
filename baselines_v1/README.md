# ReFrameJudge-v1 Baselines

This folder contains baselines for the ReFrameJudge-v1 combined datasets. These models target the project-level evaluator interface rather than the earlier FCDB-only exploration.

## Environment Setup

All baselines assume a Python virtualenv at `.venvR` in the project root. Create it once from the `ReFrameJudge/` directory:

```bash
cd /path/to/ReFrameJudge

# Choose one of the following to create the venv:
python  -m venv .venvR       # if `python` points to the desired interpreter (e.g., conda base)
python3 -m venv .venvR       # if only `python3` is available (may need `apt install python3-venv` first on Debian/Ubuntu)
```

Then install the baseline dependencies. **The `av` package must be installed from a pre-built binary wheel to avoid a FFmpeg source build that requires system `libavformat-dev` etc.:**

```bash
.venvR/bin/python -m pip install --upgrade pip setuptools wheel
.venvR/bin/python -m pip install --only-binary :all: av
.venvR/bin/python -m pip install -r requirements-baseline.txt
.venvR/bin/python -m pip install -U "transformers>=4.51.0"
```

Alternatively, you can install all packages explicitly (equivalent to above):

```bash
.venvR/bin/python -m pip install --upgrade pip setuptools wheel
.venvR/bin/python -m pip install --only-binary :all: av
.venvR/bin/python -m pip install -U \
    "transformers>=4.51.0" accelerate peft bitsandbytes qwen-vl-utils \
    torch torchvision Pillow numpy scipy scikit-learn tqdm openai
```

## CLIP Multi-task MLP

This is the first supervised ReFrameJudge evaluator baseline. It freezes a CLIP image encoder, builds pairwise features from source/edited embeddings, and trains a small MLP to predict:

- `overall_label`: 3-way `lose/tie/win`
- `improvement_score`
- `composition_gain`
- `content_preservation`
- `visual_naturalness`

Run the ReFrameJudge-v1 combined balanced1000 baseline:

```bash
.venvR/bin/python baselines_v1/reframejudge_multitask_mlp.py \
  --train-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl \
  --val-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --test-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --output-json baselines_v1/outputs/reframejudge_v1_clip_multitask_mlp.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_clip_multitask_mlp_predictions.jsonl \
  --checkpoint baselines_v1/outputs/reframejudge_v1_clip_multitask_mlp.pt \
  --cache data/cache/clip_embeddings_reframejudge_v1_balanced1000.npz \
  --hidden-dim 256 \
  --dropout 0.3 \
  --lr 3e-4 \
  --weight-decay 1e-3 \
  --regression-loss-weight 0.2 \
  --epochs 100 \
  --patience 15
```

If the default local CLIP directory is unavailable, pass a Hugging Face model id:

```bash
  --model-name openai/clip-vit-base-patch32
```

If you are running a quick environment check on a machine without all image folders, use `--missing-image-policy skip`. For actual reported numbers, keep the default `error` policy so missing data cannot silently change the benchmark.

## R2: Qwen2.5-VL Local Judge

R2 evaluates whether an open-source VLM can judge ReFrameJudge-v1 pairs better than frozen CLIP features. Run it in two stages:

1. `wo LoRA`: zero-shot or prompt-only Qwen2.5-VL.
2. `w LoRA`: the same model plus a LoRA adapter trained on ReFrameJudge-v1 train split.

Recommended starting model for a 24GB GPU:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

> Dependencies: follow the **Environment Setup** section at the top of this document. The baseline install (`requirements-baseline.txt` + the `av` pre-built wheel) covers everything needed by both CLIP MLP and Qwen-VL baselines.

### Network / Hugging Face Mirror Tips
```bash

export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY=http://127.0.0.1:10809
export NO_PROXY=localhost,127.0.0.1,.internal

# 顺便把这些写进 .bashrc，下次打开终端自动生效
echo '
# ReFrameJudge baseline proxy for HuggingFace / OpenAI
export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY=http://127.0.0.1:10809
export NO_PROXY=localhost,127.0.0.1,.internal
' >> ~/.bashrc

```

Run no-LoRA evaluation on the v1 test set:

```bash
.venvR/bin/python baselines_v1/qwen_vl_local_judge.py \
  --hf-cache-dir ./data/cache/huggingface \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --judge-mode blind_ab \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen25vl3b_nolora_blindab_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen25vl3b_nolora_blindab_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

For a quick smoke test:

```bash
  --max-samples 10 --continue-on-error
```

### With LoRA Training

The repository includes a lightweight Qwen2.5-VL LoRA trainer. It reads the ReFrameJudge-v1 train/val JSONL files directly, so exporting SFT data is optional.

Run a small smoke test first:

```bash
.venvR/bin/python baselines_v1/qwen_vl_lora_train.py \
  --train-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl \
  --val-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --project-root . \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --hf-cache-dir ./data/cache/huggingface \
  --output-dir baselines_v1/outputs/qwen25vl3b_reframejudge_lora_smoke \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --epochs 1 \
  --gradient-accumulation-steps 2 \
  --eval-steps 2 \
  --save-steps 0
```

Run the full LoRA training:

```bash
.venvR/bin/python baselines_v1/qwen_vl_lora_train.py \
  --train-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl \
  --val-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --project-root . \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --hf-cache-dir ./data/cache/huggingface \
  --output-dir baselines_v1/outputs/qwen25vl3b_reframejudge_lora \
  --load-in-4bit \
  --gradient-checkpointing \
  --epochs 3 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --eval-steps 50 \
  --save-steps 100
```

Evaluate the trained adapter in source/candidate mode:

```bash
.venvR/bin/python baselines_v1/qwen_vl_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --hf-cache-dir ./data/cache/huggingface \
  --adapter baselines_v1/outputs/qwen25vl3b_reframejudge_lora \
  --judge-mode source_candidate \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen25vl3b_lora_source_candidate_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen25vl3b_lora_source_candidate_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

Also evaluate the same adapter in blind A/B mode to measure position bias:

```bash
.venvR/bin/python baselines_v1/qwen_vl_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --hf-cache-dir ./data/cache/huggingface \
  --adapter baselines_v1/outputs/qwen25vl3b_reframejudge_lora \
  --judge-mode blind_ab \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen25vl3b_lora_blindab_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen25vl3b_lora_blindab_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

Optional: export LoRA SFT data for external trainers such as LLaMA-Factory or ms-swift:

```bash
.venvR/bin/python scripts/export_reframejudge_v1_qwen_sft.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl \
  --output-jsonl data/reframejudge_v1/sft/qwen_vl_train.jsonl \
  --project-root . \
  --absolute-paths

.venvR/bin/python scripts/export_reframejudge_v1_qwen_sft.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --output-jsonl data/reframejudge_v1/sft/qwen_vl_val.jsonl \
  --project-root . \
  --absolute-paths
```

Keep the no-LoRA and w-LoRA runs on the same test split, prompt, model size, image resolution, and decoding settings.
