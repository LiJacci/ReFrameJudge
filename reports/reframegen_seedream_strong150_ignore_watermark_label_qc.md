# ReFrameGen Seedream Strong150 Ignore-Watermark Label QC

## Files

- New label file: `data/reframejudge_v1/annotations/reframegen_seedream_strong150_vlm_labels_ignore_watermark.jsonl`
- Previous label file: `data/reframejudge_v1/annotations/reframegen_seedream_strong150_vlm_labels.jsonl`
- Generation manifest: `data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_generated_150.jsonl`

## Structural Validation

- Records: 150
- Unique ids: 150
- Duplicate ids: 0
- Missing manifest ids: 0
- Extra label ids: 0
- Schema/range errors: 0
- Watermark/logo residual in tags or reasons: 0

The new file uses `annotation_policy=composition_focused_ignore_watermarks` for all 150 records. `improvement_score` is a one-decimal numeric score in `[-2.0, 2.0]`.

## New Label Distribution

- tie: 77
- win: 68
- lose: 5

## Previous Label Distribution

- tie: 122
- win: 23
- lose: 5

## Label Transitions

- lose -> lose: 1
- lose -> tie: 1
- lose -> win: 3
- tie -> lose: 4
- tie -> tie: 72
- tie -> win: 46
- win -> tie: 4
- win -> win: 19

Ignoring watermarks changed the label distribution substantially: `win` increased from 23 to 68, while `tie` decreased from 122 to 77. This confirms that the previous labels were strongly affected by watermark/artifact penalties.

## Score And Diagnostic Distributions

Improvement score:
- -1.0: 5
- 0.0: 77
- 1.0: 59
- 1.4: 3
- 1.6: 2
- 1.8: 4

Composition gain:
- 2: 5
- 3: 70
- 4: 68
- 5: 7

Composition relevance:
- high: 106
- medium: 39
- low: 5

Label confidence:
- medium: 115
- high: 35

Artifact issue:
- True: 29
- False: 121

## Top Tags

Negative tags:
- background_still_busy: 12
- minimal_recomposition: 6
- minor_generated_smoothing: 5
- composition change is subtle: 5
- subject_identity_changed: 5
- anatomy_distortion: 5
- identity_change: 4
- background_still_somewhat_busy: 4
- subject_still_centered: 4
- limited_recomposition: 4
- composition_change_subtle: 3
- slight_identity_softening: 3
- slightly tighter crop: 3
- subject_too_close_to_left_edge: 3
- subject_less_prominent: 3

Positive tags:
- larger_subject_scale: 23
- subject_prominence: 22
- cleaner_edges: 20
- more_breathing_room: 20
- cleaner_framing: 18
- reduced_empty_space: 17
- cleaner_background: 16
- better_subject_prominence: 14
- stronger_subject_prominence: 14
- subject_more_prominent: 13
- tighter_crop: 12
- subject_shift: 9
- improved_balance: 8
- stronger_visual_focus: 7
- off_center_subject: 6

## Derived Clean Counts

- Relevance/confidence/content/realism clean samples: 130
- Strict clean samples with `artifact_issue=false`: 106
- Clean win samples: 68
- Strict clean win samples: 68

Suggested clean rule for composition-only weak labels:

```text
composition_relevance in {high, medium}
label_confidence in {high, medium}
identity_preserved == true
realism_ok == true
```

For stricter training, add:

```text
artifact_issue == false
```

## Visual Review Notes

- The new labels are much closer to the intended composition-only task because watermarks no longer dominate labels, scores, tags, or reasons.
- Many old `tie` samples become `win` when the edit clearly improves crop, subject prominence, visual balance, or background separation.
- The new labels still preserve negative judgments for identity shifts, anatomy/perspective distortion, weak recomposition, and composition worsening.
- A remaining dataset issue is phone UI / black border cleanup. Some examples become `win` because removing capture UI also creates a better photographic frame. For a pure generated-recomposition benchmark, filter screenshot/phone-UI sources separately rather than relying on watermark/artifact penalties.

Local QC sheets:

- `outputs/analysis/reframegen_ignore_watermark_tie_to_win_sample.jpg`
- `outputs/analysis/reframegen_ignore_watermark_win_to_tie_sample.jpg`
- `outputs/analysis/reframegen_ignore_watermark_loses_sample.jpg`
- `outputs/analysis/reframegen_ignore_watermark_high_score_wins_sample.jpg`
