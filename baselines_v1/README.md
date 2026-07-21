# ReFrameJudge-v1 Baselines

This folder contains baselines for the ReFrameJudge-v1 combined datasets. These models target the project-level evaluator interface rather than the earlier FCDB-only exploration.

## Environment Setup

All baselines assume a Python virtualenv at `.venvR` in the project root. Create it once from the `ReFrameJudge/` directory:

```bash
cd /path/to/ReFrameJudge
/usr/bin/python3.10 -m venv .venvR
```

Install dependencies:

```bash
.venvR/bin/python -m pip install --upgrade pip
.venvR/bin/python -m pip install -r requirements-baseline.txt
.venvR/bin/python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" accelerate bitsandbytes
```

```bash
conda create -n reframe python=3.10 -y
conda activate reframe
pip install --upgrade pip
pip install -r /workspace/ceph/ReFrameJudge/requirements-baseline.txt
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" accelerate bitsandbytes
```

If you need a proxy or Hugging Face mirror:

```bash
export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY=http://127.0.0.1:10809
export NO_PROXY=localhost,127.0.0.1,.internal
export HF_ENDPOINT=https://hf-mirror.com
```

## R1: CLIP Multi-task MLP

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

## R2: Qwen3.5 Local Judge

R2 now uses Qwen3.5 multimodal models only:

```text
Qwen/Qwen3.5-4B
Qwen/Qwen3.5-9B
```

Run every model in two modes:

1. `wo LoRA`: prompt-only zero-shot / few-shot-free judge.
2. `w LoRA`: LoRA adapter trained on ReFrameJudge-v1 train split.

For 24GB GPUs, start with 4B. If 4B works, run 9B.

### Qwen3.5-4B No-LoRA Blind A/B

```bash
python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-4B \
  --hf-cache-dir ./data/cache/huggingface \
  --judge-mode blind_ab \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_4b_nolora_blindab_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_4b_nolora_blindab_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

### Qwen3.5-4B No-LoRA source/candidate mode

```bash
python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-4B \
  --hf-cache-dir ./data/cache/huggingface \
  --judge-mode source_candidate \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_4b_nolora_source_candidate_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_4b_nolora_source_candidate_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

### Qwen3.5-9B No-LoRA Blind A/B

```bash
python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-9B \
  --hf-cache-dir ./data/cache/huggingface \
  --judge-mode blind_ab \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_9b_nolora_blindab_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_9b_nolora_blindab_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```

### Qwen3.5-9B No-LoRA source/candidate mode

```bash
python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-9B \
  --hf-cache-dir ./data/cache/huggingface \
  --judge-mode source_candidate \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_9b_nolora_source_candidate_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_9b_nolora_source_candidate_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0
```



### Qwen3.5-4B LoRA Training

Run a small smoke test first:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num_processes=4 baselines_v1/qwen35_lora_train.py \
  --train-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl \
  --val-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-4B \
  --hf-cache-dir ./data/cache/huggingface \
  --output-dir baselines_v1/outputs/qwen35_4b_reframejudge_lora_smoke \
  --load-in-4bit \
  --gradient-checkpointing \
  --mixed-precision bf16 \
  --local-files-only \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --epochs 1 \
  --gradient-accumulation-steps 2 \
  --eval-steps 2 \
  --save-steps 0
```

Full 4B LoRA training:

```bash
.venvR/bin/python baselines_v1/qwen35_lora_train.py \
  --train-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl \
  --val-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-4B \
  --hf-cache-dir ./data/cache/huggingface \
  --output-dir baselines_v1/outputs/qwen35_4b_reframejudge_lora_r16_e3 \
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
  --lora-target-modules all-linear \
  --eval-steps 50 \
  --save-steps 100 \
  --local-files-only
```

Evaluate 4B LoRA in source/candidate mode:

```bash
.venvR/bin/python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-4B \
  --hf-cache-dir ./data/cache/huggingface \
  --adapter baselines_v1/outputs/qwen35_4b_reframejudge_lora_r16_e3 \
  --judge-mode source_candidate \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_source_candidate_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_source_candidate_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0 \
  --local-files-only
```

Evaluate 4B LoRA in blind A/B mode:

```bash
.venvR/bin/python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-4B \
  --hf-cache-dir ./data/cache/huggingface \
  --adapter baselines_v1/outputs/qwen35_4b_reframejudge_lora_r16_e3 \
  --judge-mode blind_ab \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_blindab_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_blindab_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0 \
  --local-files-only
```

### Qwen3.5-9B LoRA Training

Use the same settings after 4B is verified:

```bash
.venvR/bin/python baselines_v1/qwen35_lora_train.py \
  --train-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl \
  --val-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-9B \
  --hf-cache-dir ./data/cache/huggingface \
  --output-dir baselines_v1/outputs/qwen35_9b_reframejudge_lora_r16_e3 \
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
  --lora-target-modules all-linear \
  --eval-steps 50 \
  --save-steps 100 \
  --local-files-only
```

Evaluate 9B LoRA in source/candidate mode:

```bash
.venvR/bin/python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-9B \
  --hf-cache-dir ./data/cache/huggingface \
  --adapter baselines_v1/outputs/qwen35_9b_reframejudge_lora_r16_e3 \
  --judge-mode source_candidate \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_9b_lora_r16_e3_source_candidate_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_9b_lora_r16_e3_source_candidate_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0 \
  --local-files-only
```

Evaluate 9B LoRA in blind A/B mode:

```bash
.venvR/bin/python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-9B \
  --hf-cache-dir ./data/cache/huggingface \
  --adapter baselines_v1/outputs/qwen35_9b_reframejudge_lora_r16_e3 \
  --judge-mode blind_ab \
  --shuffle-order \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_9b_lora_r16_e3_blindab_test.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_9b_lora_r16_e3_blindab_test_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0 \
  --local-files-only
```

### Score Calibration

If a LoRA model gives useful `improvement_score` but weak `overall_label`, run source/candidate evaluation on the validation split and calibrate thresholds:

```bash
.venvR/bin/python baselines_v1/qwen35_local_judge.py \
  --input-jsonl data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl \
  --project-root . \
  --model-name Qwen/Qwen3.5-4B \
  --hf-cache-dir ./data/cache/huggingface \
  --adapter baselines_v1/outputs/qwen35_4b_reframejudge_lora_r16_e3 \
  --judge-mode source_candidate \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_source_candidate_val.json \
  --predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_source_candidate_val_predictions.jsonl \
  --load-in-4bit \
  --max-new-tokens 512 \
  --temperature 0

.venvR/bin/python baselines_v1/calibrate_score_labels.py \
  --val-predictions baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_source_candidate_val_predictions.jsonl \
  --test-predictions baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_source_candidate_test_predictions.jsonl \
  --output-json baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_score_calibrated_test.json \
  --output-predictions-jsonl baselines_v1/outputs/reframejudge_v1_qwen35_4b_lora_r16_e3_score_calibrated_test_predictions.jsonl \
  --score-key improvement_score
```

### Source-Candidate Error Review

After running the current main R2 model, generate a summary JSON and visual HTML review:

```bash
python baselines_v1/analyze_qwen35_source_candidate.py \
  --predictions baselines_v1/outputs/reframejudge_v1_qwen35_9b_lora_r16_e3_source_candidate_test_predictions.jsonl \
  --summary-json baselines_v1/outputs/reframejudge_v1_qwen35_9b_lora_r16_e3_source_candidate_summary.json \
  --html-output baselines_v1/outputs/reframejudge_v1_qwen35_9b_lora_r16_e3_source_candidate_error_review.html \
  --project-root . \
  --max-cases 80 \
  --title "Qwen3.5-9B LoRA Source-Candidate Error Review"
```
