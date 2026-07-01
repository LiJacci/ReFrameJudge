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
