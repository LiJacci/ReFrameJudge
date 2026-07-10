# Data Sources

This document tracks external datasets that can be converted into ReFrameJudge pairs.

## FCDB / Flickr Image Cropping Dataset

Repository:

```text
https://github.com/yiling-chen/flickr-cropping-dataset
```

Useful files:

```text
ranking_annotation.json
cropping_training_set.json
cropping_testing_set.json
```

For ReFrameJudge, the primary file is `ranking_annotation.json`.

Reason:

```text
ranking annotation gives pairwise crop preferences.
This directly matches ReFrameJudge's win/tie/lose task.
```

The cropping annotations are useful as secondary data, but they are not the first choice for training a pairwise judge:

```text
cropping annotation: one good crop per image, useful for source -> good_crop pairs
ranking annotation: crop A vs crop B with votes, useful for win/lose supervision
```

Each image contains multiple crop pairs:

```json
{
  "url": "...jpg",
  "flickr_photo_id": 15251367120,
  "crops": [
    {
      "vote_for_0": 4,
      "vote_for_1": 1,
      "crop_0": [139, 234, 300, 400],
      "crop_1": [171, 281, 300, 400]
    }
  ]
}
```

The crop with more votes is considered more visually pleasing.

Conversion idea:

```text
lower-vote crop -> higher-vote crop = win
higher-vote crop -> lower-vote crop = lose, optional reverse sample
similar-vote crop pairs = tie, optional
```

Expected local layout:

```text
data/external/fcdb/
  ranking_annotation.json
  images/
    <downloaded Flickr images>
```

Download ranking annotation:

```bash
curl -L \
  https://cdn.jsdelivr.net/gh/yiling-chen/flickr-cropping-dataset@master/ranking_annotation.json \
  -o data/external/fcdb/ranking_annotation.json
```

Download images listed in the ranking annotation:

```bash
python3 scripts/download_fcdb_images.py \
  --annotation data/external/fcdb/ranking_annotation.json \
  --output-dir data/external/fcdb/images \
  --workers 16 \
  --timeout 8 \
  --retries 1
```

Build ReFrameJudge pairs:

```bash
python3 scripts/prepare_fcdb_pairs.py \
  --annotation data/external/fcdb/ranking_annotation.json \
  --image-dir data/external/fcdb/images \
  --output-jsonl data/pairs/annotations/fcdb_pairs.jsonl \
  --output-image-dir data/pairs/images \
  --include-reverse
```

For metadata-only dry run:

```bash
python3 scripts/prepare_fcdb_pairs.py \
  --annotation data/external/fcdb/ranking_annotation.json \
  --output-jsonl data/pairs/annotations/fcdb_pairs_preview.jsonl \
  --metadata-only
```
## AesRecon

AesRecon can provide real-photo good/poor pairs for an early ReFrameJudge-v1 composition subset. The local test JSON is expected to map:

```text
good_image.jpg -> poor_image.jpg
```

For this subset, the dataset direction is used as the preference label:

```text
poor image -> good image = win
```

GPT is used only to add weak composition annotations. It is not asked to decide the winner.

Annotate a small sample with an OpenAI vision model:

```bash
export OPENAI_API_KEY="your_api_key"

python3 scripts/label_aesrecon_pairs_gpt.py \
  --dataset-root ../../shared/ai-camera/AesRecon_dataset \
  --test-json ../../shared/ai-camera/AesRecon_dataset/jsons/test/test.json \
  --output-jsonl outputs/aesrecon_gpt_composition_labels_100.jsonl \
  --summary-json outputs/aesrecon_gpt_composition_labels_100_summary.json \
  --model gpt-5.4 \
  --max-samples 100 \
  --continue-on-error
```

The script treats the poor image as `source_image` and the good image as `edited_image`. Each output row has `overall_label=win` from the dataset direction, while GPT fills weak composition fields such as `composition_relevance`, `label_confidence`, `composition_score`, `composition_gain`, `positive_tags`, `negative_tags`, and `reason`. The prompt asks GPT to use `high` relevance/confidence and `composition_gain=5` sparingly, so these fields can help filter strong composition-specific pairs instead of turning every preferred AesRecon pair into a maximal composition win.

Freeze the 500-pair GPT annotation output as the first ReFrameJudge-v1 subset:

```bash
python3 scripts/build_reframejudge_v1_aesrecon.py
```

Outputs:

```text
data/reframejudge_v1/annotations/aesrecon_500.jsonl
data/reframejudge_v1/splits/aesrecon_train.jsonl
data/reframejudge_v1/splits/aesrecon_val.jsonl
data/reframejudge_v1/splits/aesrecon_test.jsonl
reports/aesrecon_500_quality_report.md
```

Prepare non-overlapping AesRecon poor images for the ReFrameGen generated-pair pilot:

```bash
python3 scripts/prepare_reframegen_sources.py --check-images
```

Outputs:

```text
data/reframejudge_v1/source_manifests/reframegen_pilot_aesrecon_sources_50.jsonl
reports/reframegen_pilot_sources_50.md
```

The manifest excludes all `source_record_id` values already used by `data/reframejudge_v1/annotations/aesrecon_500.jsonl`.

Match each source image to suitable positive composition prompts with a vision-language model:

```bash
export OPENAI_API_KEY="your_api_key"

python3 scripts/match_reframegen_prompts_vlm.py \
  --check-images
```

This creates `data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_matched_150.jsonl`, with 3 source-aware positive composition prompts for each source image.

Validate the matched Seedream editing tasks:

```bash
python3 scripts/generate_reframegen_seedream.py \
  --dry-run \
  --check-images
```

Remove `--dry-run` and configure `SEEDREAM_API_KEY`, `SEEDREAM_BASE_URL`, and `SEEDREAM_MODEL` to call the actual Seedream endpoint.

For the stronger recomposition pilot, first test 10 source images with 2 matched prompts each:

```bash
python3 scripts/match_reframegen_prompts_vlm.py \
  --check-images \
  --limit-sources 10 \
  --top-k 2 \
  --candidate-tag seedream_strong \
  --output-image-dir data/reframejudge_v1/generated/reframegen_pilot_seedream_strong20/images \
  --output-jsonl data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_matched_20.jsonl \
  --raw-jsonl outputs/reframegen_prompt_matching_strong20_raw.jsonl
```

Then generate from `reframegen_pilot_seedream_strong_matched_20.jsonl` with `scripts/generate_reframegen_seedream.py`.

The generation script defaults to `--size source`, which keeps the output pixel size close to the input image instead of forcing 2K upscaling. The prompts may still request a portrait/landscape reframing when compositionally useful.
