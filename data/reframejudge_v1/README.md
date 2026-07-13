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

## ReFrameGen Seedream Strong150

`annotations/reframegen_seedream_strong150.jsonl` is the first frozen generated-pair ReFrameJudge-v1 pilot subset.

- Source: 50 AesRecon poor images outside AesRecon-500
- Generated edits: 3 Seedream recompositions per source
- Pair type: `generated_recomposition_pair`
- Label source: VLM composition annotation
- Annotation policy: `composition_focused_ignore_watermarks`
- Records: 150
- Source-level split: all 3 edits from one source stay in the same split

Split files:

```text
splits/reframegen_seedream_strong150_train.jsonl  105
splits/reframegen_seedream_strong150_val.jsonl     15
splits/reframegen_seedream_strong150_test.jsonl    30
```

Rebuild from the VLM label output:

```bash
python3 scripts/build_reframejudge_v1_reframegen.py
```

Quality report:

```text
reports/reframegen_seedream_strong150_quality_report.md
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

## ReFrameGen Prompt Matching And Seedream Generation

Positive composition prompts are stored in:

```text
prompt_banks/reframegen_positive_composition_prompts.json
```

First match each source image to suitable composition prompts with a vision-language model:

```bash
export OPENAI_API_KEY="your_api_key"

python3 scripts/match_reframegen_prompts_vlm.py \
  --check-images
```

Matching output:

```text
generated_manifests/reframegen_pilot_seedream_matched_150.jsonl
```

This selects 3 source-aware positive composition prompts per source, for about 150 generated candidates.

Validate the matched generation manifest without spending Seedream credits:

```bash
python3 scripts/generate_reframegen_seedream.py \
  --dry-run \
  --check-images
```

Run Seedream generation after configuring the dataset root and Seedream credentials:

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

## Strong Recomposition Pilot

The prompt bank now asks for visible geometric recomposition rather than subtle retouching. Before regenerating all 150 candidates, run a 10-source / 20-candidate pilot:

```bash
export OPENAI_API_KEY="your_api_key"
export AESRECON_DATASET_ROOT="/path/to/AesRecon_dataset"

python3 scripts/match_reframegen_prompts_vlm.py \
  --check-images \
  --limit-sources 10 \
  --top-k 2 \
  --candidate-tag seedream_strong \
  --output-image-dir data/reframejudge_v1/generated/reframegen_pilot_seedream_strong20/images \
  --output-jsonl data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_matched_20.jsonl \
  --raw-jsonl outputs/reframegen_prompt_matching_strong20_raw.jsonl
```

Then validate the Seedream tasks:

```bash
python3 scripts/generate_reframegen_seedream.py \
  --generation-manifest data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_matched_20.jsonl \
  --output-jsonl data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_generated_20.jsonl \
  --raw-jsonl outputs/reframegen_seedream_strong20_raw.jsonl \
  --summary-json outputs/reframegen_seedream_strong20_summary.json \
  --dry-run \
  --check-images
```

Run actual generation by removing `--dry-run` after configuring `SEEDREAM_API_KEY`, `SEEDREAM_BASE_URL`, and `SEEDREAM_MODEL`.

`scripts/generate_reframegen_seedream.py` defaults to `--size source`, so it keeps the source aspect ratio. If the source image is below Seedream 5.0's minimum pixel requirement, the script scales the requested size up proportionally to at least `3,686,400` pixels using 64-pixel alignment. If the endpoint rejects exact pixel sizes, set `SEEDREAM_SIZE` to a provider-supported value.

## VLM Labeling for Generated Pairs

After generating the 50-source / 150-edit strong pilot, label each source/edit pair with a VLM:

```bash
export OPENAI_API_KEY="your_api_key"
export AESRECON_DATASET_ROOT="/path/to/AesRecon_dataset"

python3 scripts/label_reframegen_pairs_vlm.py \
  --check-images \
  --generation-manifest data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_generated_150.jsonl \
  --output-jsonl data/reframejudge_v1/annotations/reframegen_seedream_strong150_vlm_labels_ignore_watermark.jsonl \
  --raw-jsonl outputs/reframegen_seedream_strong150_vlm_label_ignore_watermark_raw.jsonl \
  --summary-json outputs/reframegen_seedream_strong150_vlm_label_ignore_watermark_summary.json
```

The labels include `overall_label`, a one-decimal `improvement_score`, composition/content/realism scores, and `change_strength`. The default prompt completely ignores watermarks and does not mention them in labels, scores, tags, or reasons.
