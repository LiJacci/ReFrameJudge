# Development Notes

## Validate Annotations

Validate JSONL fields and labels:

```bash
python3 scripts/validate_dataset.py data/pairs/annotations/sample_annotations.jsonl --root .
```

Also check whether image paths exist:

```bash
python3 scripts/validate_dataset.py data/pairs/annotations/train.jsonl --root . --check-images
```

## Git Hygiene

Large files are intentionally ignored:

```text
raw datasets
generated images
model checkpoints
training outputs
downloaded archives
```

Keep only:

```text
code
docs
configs
small examples
metadata
annotation JSONL files
```

## Suggested Workflow

```text
1. Add or update dataset metadata.
2. Run the validation script.
3. Commit code, docs, and metadata.
4. Keep large local assets outside git.
```

## Test FCDB Conversion

Run the fixture conversion:

```bash
python3 scripts/prepare_fcdb_pairs.py \
  --annotation tests/fixtures/fcdb_ranking_sample.json \
  --output-jsonl /tmp/reframejudge_fcdb_sample.jsonl \
  --output-image-dir data/pairs/images \
  --metadata-only \
  --include-reverse
```

Validate the generated JSONL:

```bash
python3 scripts/validate_dataset.py /tmp/reframejudge_fcdb_sample.jsonl --root .
```

## Split FCDB Pairs

Split by source photo group to avoid putting crops from the same Flickr photo in both train and test:

```bash
python3 scripts/split_dataset.py \
  --input data/pairs/annotations/fcdb_pairs_5k.jsonl \
  --output-dir data/pairs/annotations \
  --prefix fcdb_ \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

Validate each split:

```bash
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_train.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_val.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_test.jsonl --root . --check-images
```

## Build FCDB v2 Splits

FCDB v2 keeps structured crop and vote metadata for each pair:

```text
photo_id
pair_index
source_crop / edited_crop
source_votes / edited_votes
vote_margin / abs_vote_margin
preference_strength
```

For the tracked FCDB 5k crop subset, reuse existing crop images by preserving the original pair ids and limiting to the first 5000 emitted source/edit directions:

```bash
python3 scripts/prepare_fcdb_pairs.py \
  --annotation data/external/fcdb/ranking_annotation.json \
  --image-dir data/external/fcdb/images \
  --output-jsonl data/pairs/annotations/fcdb_strong_pairs_5k.jsonl \
  --output-image-dir data/pairs/images \
  --include-reverse \
  --preference-filter strong \
  --max-source-records 5000 \
  --preserve-source-ids \
  --metadata-only

python3 scripts/split_dataset.py \
  --input data/pairs/annotations/fcdb_strong_pairs_5k.jsonl \
  --output-dir data/pairs/annotations \
  --prefix fcdb_strong_ \
  --seed 42
```

Create the 3-way variant by mapping weak 3:2 / 2:3 preferences to `tie`:

```bash
python3 scripts/prepare_fcdb_pairs.py \
  --annotation data/external/fcdb/ranking_annotation.json \
  --image-dir data/external/fcdb/images \
  --output-jsonl data/pairs/annotations/fcdb_3way_pairs_5k.jsonl \
  --output-image-dir data/pairs/images \
  --include-reverse \
  --preference-filter all \
  --label-weak-as-tie \
  --max-source-records 5000 \
  --preserve-source-ids \
  --metadata-only

python3 scripts/split_dataset.py \
  --input data/pairs/annotations/fcdb_3way_pairs_5k.jsonl \
  --output-dir data/pairs/annotations \
  --prefix fcdb_3way_ \
  --seed 42
```

Validate all v2 files:

```bash
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_strong_pairs_5k.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_strong_train.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_strong_val.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_strong_test.jsonl --root . --check-images

python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_3way_pairs_5k.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_3way_train.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_3way_val.jsonl --root . --check-images
python3 scripts/validate_dataset.py data/pairs/annotations/fcdb_3way_test.jsonl --root . --check-images
```

## Build Review HTML

Create a visual review page for manual quality control:

```bash
python3 scripts/build_review_html.py \
  --input data/pairs/annotations/fcdb_train.jsonl \
  --output ../outputs/fcdb_train_review_100.html \
  --project-root . \
  --sample-size 100 \
  --seed 7 \
  --title "ReFrameJudge FCDB Train Review"
```
