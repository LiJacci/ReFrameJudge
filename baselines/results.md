# Baseline Results

## CLIP + Logistic Regression on FCDB 5k

Date: 2026-07-06

Command:

```bash
.venv/bin/python baselines/clip_logreg.py \
  --train-jsonl data/pairs/annotations/fcdb_train.jsonl \
  --val-jsonl data/pairs/annotations/fcdb_val.jsonl \
  --test-jsonl data/pairs/annotations/fcdb_test.jsonl \
  --output-json outputs/clip_logreg_fcdb_5k.json \
  --cache data/cache/clip_embeddings_fcdb_5k.npz \
  --batch-size 64
```

Setup:

```text
encoder: openai/clip-vit-base-patch32
device: mps
feature_dim: 2048
pair_feature: [src, edit, edit - src, src * edit]
classifier: LogisticRegression(C=0.1, solver=liblinear, class_weight=balanced)
train/val/test: 4000 / 500 / 500
```

Metrics:

| Split | Accuracy | Precision | Recall | F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.676 | 0.676 | 0.676 | 0.676 | 0.676 | 0.7688 |
| test | 0.672 | 0.672 | 0.672 | 0.672 | 0.672 | 0.7203 |

Confusion matrices use label order `[lose, win]`.

Validation:

```text
val:
[[169, 81],
 [ 81,169]]

test:
[[168, 82],
 [ 82,168]]
```

