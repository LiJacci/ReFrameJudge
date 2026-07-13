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
  --output-json outputs/reframejudge_v1_clip_multitask_mlp.json \
  --predictions-jsonl outputs/reframejudge_v1_clip_multitask_mlp_predictions.jsonl \
  --checkpoint outputs/reframejudge_v1_clip_multitask_mlp.pt \
  --cache data/cache/clip_embeddings_reframejudge_v1_balanced1000.npz \
  --epochs 80 \
  --patience 12
```

If the default local CLIP directory is unavailable, pass a Hugging Face model id:

```bash
  --model-name openai/clip-vit-base-patch32
```

If you are running a quick environment check on a machine without all image folders, use `--missing-image-policy skip`. For actual reported numbers, keep the default `error` policy so missing data cannot silently change the benchmark.
