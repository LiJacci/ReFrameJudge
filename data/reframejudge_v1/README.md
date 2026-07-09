# ReFrameJudge-v1

This directory stores frozen ReFrameJudge-v1 annotation subsets and split files.

## AesRecon-500

`annotations/aesrecon_500.jsonl` is the first frozen ReFrameJudge-v1 subset.

- Source: AesRecon test good/poor pairs
- Pair type: `real_photo_aesthetic_pair`
- Label source: `dataset_direction+gpt_weak_composition_annotation`
- Preference label: `poor -> good = win`
- GPT role: weak composition annotation only, not winner selection
- Records: 500

Split files:

```text
splits/aesrecon_train.jsonl  400
splits/aesrecon_val.jsonl     50
splits/aesrecon_test.jsonl    50
```

Rebuild from the GPT output:

```bash
python3 scripts/build_reframejudge_v1_aesrecon.py
```

Quality report:

```text
reports/aesrecon_500_quality_report.md
```

## ReFrameGen Pilot Sources

`source_manifests/reframegen_pilot_aesrecon_sources_50.jsonl` contains 50 AesRecon poor images selected from the AesRecon test pairs that are not included in AesRecon-500.

- Source images: 50
- Overlap with AesRecon-500: 0
- Intended use: generate recomposed candidates for ReFrameGen pilot pairs
- Recommended candidates per source: 2
- Expected generated pairs: about 100

Rebuild the manifest:

```bash
python3 scripts/prepare_reframegen_sources.py --check-images
```

Source paths in the manifest are relative to the AesRecon dataset root.

## ReFrameGen Seedream Generation Plan

Positive composition prompts are stored in:

```text
prompt_banks/reframegen_positive_composition_prompts.json
```

Build a dry-run generation manifest with 5 randomly assigned positive composition prompts per source:

```bash
python3 scripts/generate_reframegen_seedream.py \
  --dry-run \
  --check-images
```

Dry-run output:

```text
generated_manifests/reframegen_pilot_seedream_250.jsonl
```

Run Seedream generation after configuring the dataset root and API credentials:

```bash
export AESRECON_DATASET_ROOT="/path/to/AesRecon_dataset"
export SEEDREAM_API_KEY="your_api_key"
export SEEDREAM_BASE_URL="https://ark.ap-southeast.bytepluses.com/api/v3"
export SEEDREAM_MODEL="doubao-seedream-4-0"

python3 scripts/generate_reframegen_seedream.py \
  --check-images \
  --continue-on-error
```

For China Volcengine Ark, set `SEEDREAM_BASE_URL` to the region endpoint, for example:

```text
https://ark.cn-beijing.volces.com/api/v3
```
