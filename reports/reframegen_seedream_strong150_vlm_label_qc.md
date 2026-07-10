# ReFrameGen Seedream Strong150 VLM Label QC

## Files

- Label file: `data/reframejudge_v1/annotations/reframegen_seedream_strong150_vlm_labels.jsonl`
- Summary file: `outputs/reframegen_seedream_strong150_vlm_label_summary.json`
- Generation manifest: `data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_generated_150.jsonl`

## Structural Validation

- Records: 150
- Unique ids: 150
- Duplicate ids: 0
- Missing manifest ids: 0
- Extra label ids: 0
- Schema/range errors: 0

All labels map cleanly to the generated-pair manifest. Required score ranges, boolean fields, and level fields are valid.

## Label Distribution

- tie: 122
- win: 23
- lose: 5

The distribution is conservative. The VLM does not appear to simply assume that the generated edit is better because the second image is labeled as the edited image.

## Training Usability

- usable_for_training=true: 51
- usable_for_training=false: 99

Cross-tab:

- win / usable=true: 23
- tie / usable=true: 27
- tie / usable=false: 95
- lose / usable=true: 1
- lose / usable=false: 4

The one `lose` but `usable_for_training=true` sample can be useful as a negative training pair, but if we define `usable_for_training` as "usable positive/tie supervision only", this rule should be revised.

## Score Distributions

Improvement score:

- 2: 13
- 1: 10
- 0: 122
- -1: 4
- -2: 1

Composition gain:

- 5: 13
- 4: 50
- 3: 84
- 2: 2
- 1: 1

Change strength:

- 3: 8
- 2: 99
- 1: 43
- 0: 0

There are many `tie` samples with `composition_gain=4`. This is internally explainable because the prompt allows composition improvement to be offset by content, realism, UI cleanup, or artifact issues. For model training, prefer using `overall_label` as the classification target and treat `composition_gain` as a diagnostic score, not as a direct replacement label.

## Quality Diagnostics

- composition_relevance=high: 55
- composition_relevance=medium: 67
- composition_relevance=low: 28
- label_confidence=high: 24
- label_confidence=medium: 126
- identity_preserved=true: 131
- realism_ok=true: 137
- artifact_issue=true: 109

The high artifact count is the main concern. Many negative tags mention watermarks or generated overlays. Some of these are visible artifacts, but some may be model over-diagnosis or tag wording drift. Do not filter only by exact tag strings without normalization.

## Top Negative Tags

- watermark_added: 63
- mostly_ui_cleanup: 22
- limited_recomposition: 14
- generated_look: 11
- added_watermark: 10
- identity_changed: 10
- minor_generation_artifacts: 9
- background_still_busy: 8
- change_too_subtle: 7
- watermark: 6

## Top Positive Tags

- subject_more_prominent: 33
- larger_subject_scale: 23
- subject_prominence: 22
- more_breathing_room: 17
- cleaner_edges: 16
- tighter_crop: 15
- cleaner_background: 14
- reduced_empty_space: 13
- ui_removed: 13
- subject_shift: 10

## Visual Review Notes

Sampled `win` labels are mostly reasonable: they often show stronger subject prominence, cleaner crop, better subject/background separation, or clearer background organization. Some wins are only moderate, and a few are partly driven by cleanup or beautification.

Sampled `tie` labels are generally conservative. Many have some composition improvement but are marked tie because the edit is too subtle, changes content, removes phone UI, or introduces generated artifacts.

The frequent watermark-related tags require caution. The local contact sheets used for QC are:

- `outputs/analysis/reframegen_labels_wins_sample.jpg`
- `outputs/analysis/reframegen_labels_loses_sample.jpg`
- `outputs/analysis/reframegen_labels_usable_ties_sample.jpg`
- `outputs/analysis/reframegen_labels_watermark_sample.jpg`
- `outputs/analysis/reframegen_labels_notusable_sample.jpg`

## Recommendation

Use this label file as a valid weak-label pilot, but do not treat all 150 samples as clean training data.

Recommended clean subset rule for the next step:

```text
usable_for_training == true
label_confidence in {medium, high}
composition_relevance in {medium, high}
identity_preserved == true
realism_ok == true
```

This yields 51 usable samples before any additional manual review. For stricter positive training, start with `overall_label=win` and `usable_for_training=true`, which yields 23 positive examples.

Before scaling generation, improve the pipeline by filtering screenshot/phone-UI sources and by normalizing diagnostic tags into canonical groups such as `watermark_or_overlay`, `ui_cleanup_dominant`, `identity_changed`, `low_change_strength`, `generated_artifact`, and `content_changed`.
