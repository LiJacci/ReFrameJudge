# Baselines

## CLIP + Logistic Regression

Install baseline dependencies in a local virtual environment:

```bash
python3 -m venv .venvR
.venvR/bin/python -m pip install --upgrade pip
.venvR/bin/python -m pip install torch torchvision transformers scikit-learn tqdm
```

Run a smoke test:

> The CLIP model files are expected in `data/cache/clip-vit-base-patch32`. If they are not present, download them manually (e.g. from [ModelScope](https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch32)) or pass `--model-name openai/clip-vit-base-patch32` to download from Hugging Face.

```bash
.venvR/bin/python baselines/clip_logreg.py \
  --train-jsonl data/pairs/annotations/fcdb_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_test.jsonl \
  --output-json outputs/clip_logreg_smoke.json \
  --predictions-jsonl outputs/clip_logreg_smoke_predictions.jsonl \
  --max-train 200 \
  --max-val 100 \
  --max-test 100
```

Run the full FCDB 5k baseline:

```bash
.venvR/bin/python baselines/clip_logreg.py \
  --train-jsonl data/pairs/annotations/fcdb_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_test.jsonl \
  --output-json outputs/clip_logreg_fcdb_5k.json \
  --predictions-jsonl outputs/clip_logreg_fcdb_5k_predictions.jsonl
```

Run the FCDB strong-preference v2 baseline:

```bash
.venvR/bin/python baselines/clip_logreg.py \
  --train-jsonl data/pairs/annotations/fcdb_strong_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_strong_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_strong_test.jsonl \
  --output-json outputs/clip_logreg_fcdb_strong_5k.json \
  --predictions-jsonl outputs/clip_logreg_fcdb_strong_5k_predictions.jsonl \
  --cache data/cache/clip_embeddings_fcdb_5k.npz
```

Run the FCDB 3-way v2 baseline:

```bash
.venvR/bin/python baselines/clip_logreg_3way.py \
  --train-jsonl data/pairs/annotations/fcdb_3way_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_3way_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_3way_test.jsonl \
  --output-json outputs/clip_logreg_3way_fcdb_5k.json \
  --predictions-jsonl outputs/clip_logreg_3way_fcdb_5k_predictions.jsonl \
  --cache data/cache/clip_embeddings_fcdb_5k.npz
```

Run the FCDB threshold-calibrated 3-way baseline:

```bash
.venvR/bin/python baselines/clip_threshold_3way.py \
  --strong-train-jsonl data/pairs/annotations/fcdb_strong_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_3way_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_3way_test.jsonl \
  --output-json outputs/clip_threshold_3way_fcdb_5k.json \
  --predictions-jsonl outputs/clip_threshold_3way_fcdb_5k_predictions.jsonl \
  --cache data/cache/clip_embeddings_fcdb_5k.npz
```

Build an error review page from validation/test predictions:

```bash
.venvR/bin/python scripts/build_error_review_html.py \
  --predictions outputs/clip_logreg_fcdb_5k_predictions.jsonl \
  --output outputs/clip_logreg_error_review.html \
  --project-root . \
  --buckets FP FN \
  --sample-size 100 \
  --title "CLIP LogReg Error Review"
```

Build a 3-way error review page:

```bash
.venvR/bin/python scripts/build_error_review_html.py \
  --predictions outputs/clip_logreg_3way_fcdb_5k_predictions.jsonl \
  --output outputs/clip_logreg_3way_error_review.html \
  --project-root . \
  --buckets ERROR \
  --sample-size 120 \
  --title "CLIP LogReg 3-way Error Review"
```

Build a threshold-calibrated 3-way error review page:

```bash
.venvR/bin/python scripts/build_error_review_html.py \
  --predictions outputs/clip_threshold_3way_fcdb_5k_predictions.jsonl \
  --output outputs/clip_threshold_3way_error_review.html \
  --project-root . \
  --buckets ERROR \
  --sample-size 120 \
  --title "CLIP Threshold 3-way Error Review"
```

## DINOv3/DINOv2 + Logistic Regression

DINOv3 is supported by Hugging Face Transformers 4.56+ but the official Meta weights may require accepting the model license before download. If DINOv3 access is unavailable, use DINOv2 as a fallback with `--model-name facebook/dinov2-base`.

Run DINOv3:

```bash
.venvR/bin/python baselines/vision_logreg.py \
  --train-jsonl data/pairs/annotations/fcdb_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_test.jsonl \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --cache data/cache/dinov3_vitb16_embeddings_fcdb_5k.npz \
  --output-json outputs/dinov3_logreg_fcdb_5k.json \
  --predictions-jsonl outputs/dinov3_logreg_fcdb_5k_predictions.jsonl
```

Run DINOv2 fallback:

```bash
.venvR/bin/python baselines/vision_logreg.py \
  --train-jsonl data/pairs/annotations/fcdb_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_test.jsonl \
  --model-name facebook/dinov2-base \
  --cache data/cache/dinov2_base_embeddings_fcdb_5k.npz \
  --output-json outputs/dinov2_logreg_fcdb_5k.json \
  --predictions-jsonl outputs/dinov2_logreg_fcdb_5k_predictions.jsonl
```

## CLIP + DINOv2 Feature Fusion

Run this after generating both CLIP and DINOv2 embedding caches:

```bash
.venvR/bin/python baselines/fusion_logreg.py \
  --train-jsonl data/pairs/annotations/fcdb_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_test.jsonl \
  --clip-cache data/cache/clip_embeddings_fcdb_5k.npz \
  --vision-cache data/cache/dinov2_base_embeddings_fcdb_5k.npz \
  --vision-name dinov2-base \
  --output-json outputs/clip_dinov2_fusion_logreg_fcdb_5k.json \
  --predictions-jsonl outputs/clip_dinov2_fusion_logreg_fcdb_5k_predictions.jsonl
```
