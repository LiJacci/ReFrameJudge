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
