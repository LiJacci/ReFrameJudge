# ReFrameJudge Dataset Schema

Each sample is stored as one JSON object per line in a `.jsonl` file.

## Required Fields

```json
{
  "id": "rfj_000001",
  "source_image": "data/pairs/images/source/rfj_000001.jpg",
  "edited_image": "data/pairs/images/edit/rfj_000001.jpg",
  "edit_type": "zoom_in",
  "data_source": "manual_or_dataset_name",
  "overall_label": "win",
  "improvement_score": 2,
  "composition_gain": 5,
  "content_preservation": 5,
  "visual_naturalness": 4,
  "issue_tags": ["better_subject_prominence", "better_crop"],
  "notes": ""
}
```

## Field Definitions

`id`: Unique sample id.

`source_image`: Path to the original image, relative to the project root.

`edited_image`: Path to the recomposed/generated image, relative to the project root.

`edit_type`: One of:

```text
zoom_in
shift
crop
outpainting
view_change
subject_reposition
unknown
```

`data_source`: Source of the pair, such as `GAIC`, `CPC`, `FCDB`, `generated`, or `manual`.

`overall_label`: One of:

```text
win
tie
lose
```

`improvement_score`: Integer or float from -2 to 2.

```text
-2: much worse
-1: slightly worse
 0: about the same
 1: slightly better
 2: much better
```

`composition_gain`: 1-5 score measuring whether framing, balance, crop, subject placement, empty space, and visual focus improved.

`content_preservation`: 1-5 score measuring whether the main subject, identity, scene semantics, and important details are preserved.

`visual_naturalness`: 1-5 score measuring whether the edited image looks natural and free of obvious artifacts.

`issue_tags`: Multi-label diagnostic tags.

`notes`: Optional free-text comment.

## Recommended Split Files

```text
data/pairs/annotations/train.jsonl
data/pairs/annotations/val.jsonl
data/pairs/annotations/test.jsonl
```

