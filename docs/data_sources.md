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
