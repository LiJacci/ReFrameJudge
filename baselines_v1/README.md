# ReFrameJudge-v1 Baselines

This folder contains baselines for the ReFrameJudge-v1 combined datasets. These models target the project-level evaluator interface rather than the earlier FCDB-only exploration.

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

Install extra dependencies in your server environment:

```bash
.venvR/bin/python -m pip install -U "transformers>=4.51.0" accelerate qwen-vl-utils peft bitsandbytes
```

Run no-LoRA evaluation on the v1 test set:

```bash
.venvR/bin/python baselines_v1/qwen_vl_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen25vl3b_nolora_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen25vl3b_nolora_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

For a quick smoke test:

```bash
  --max-samples 10 --continue-on-error
```

Export LoRA SFT data:

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

After training a LoRA adapter, evaluate it with the same script:

```bash
.venvR/bin/python baselines_v1/qwen_vl_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/qwen25vl3b_reframejudge_lora \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen25vl3b_lora_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen25vl3b_lora_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

Keep the no-LoRA and w-LoRA runs on the same test split, prompt, model size, image resolution, and decoding settings.
