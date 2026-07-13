# ReFrameGen Seedream Strong150 Quality Report

## Overview

- Source label file: `data/reframejudge_v1/annotations/reframegen_seedream_strong150_vlm_labels_ignore_watermark.jsonl`
- Frozen annotation file: `data/reframejudge_v1/annotations/reframegen_seedream_strong150.jsonl`
- Total records: 150
- Source groups: 50
- Edits per source: 3
- Clean records: 130
- Strict clean records: 106

## Label Distribution

| Value | Count |
| --- | ---: |
| tie | 77 |
| win | 68 |
| lose | 5 |

## Improvement Score

| Value | Count |
| --- | ---: |
| -1.0 | 5 |
| 0.0 | 77 |
| 1.0 | 59 |
| 1.4 | 3 |
| 1.6 | 2 |
| 1.8 | 4 |

## Composition Gain

| Value | Count |
| --- | ---: |
| 2 | 5 |
| 3 | 70 |
| 4 | 68 |
| 5 | 7 |

## Composition Relevance

| Value | Count |
| --- | ---: |
| high | 106 |
| medium | 39 |
| low | 5 |

## Label Confidence

| Value | Count |
| --- | ---: |
| medium | 115 |
| high | 35 |

## Clean Flags

| Value | Count |
| --- | ---: |
| True | 130 |
| False | 20 |

## Strict Clean Flags

| Value | Count |
| --- | ---: |
| True | 106 |
| False | 44 |

## Split Summary

```json
{
  "train": {
    "records": 105,
    "sources": 35,
    "labels": {
      "tie": 55,
      "win": 48,
      "lose": 2
    },
    "clean_for_training": {
      "false": 17,
      "true": 88
    },
    "strict_clean_for_training": {
      "false": 34,
      "true": 71
    }
  },
  "val": {
    "records": 15,
    "sources": 5,
    "labels": {
      "tie": 10,
      "win": 5
    },
    "clean_for_training": {
      "true": 14,
      "false": 1
    },
    "strict_clean_for_training": {
      "true": 13,
      "false": 2
    }
  },
  "test": {
    "records": 30,
    "sources": 10,
    "labels": {
      "tie": 12,
      "win": 15,
      "lose": 3
    },
    "clean_for_training": {
      "true": 28,
      "false": 2
    },
    "strict_clean_for_training": {
      "true": 22,
      "false": 8
    }
  }
}
```

## Prompt By Label

| Prompt | Records | Win | Tie | Lose |
| --- | ---: | ---: | ---: | ---: |
| active_space_directional_room | 6 | 4 | 2 | 0 |
| bird_eye_top_down_layout | 1 | 0 | 1 | 0 |
| centered_symmetry | 3 | 1 | 1 | 1 |
| diagonal_dynamic_composition | 1 | 1 | 0 | 0 |
| edge_control_clean_crop | 28 | 14 | 14 | 0 |
| fill_the_frame_subject_prominence | 23 | 16 | 7 | 0 |
| foreground_depth_layering | 1 | 0 | 1 | 0 |
| frame_within_frame | 5 | 3 | 1 | 1 |
| horizon_thirds_placement | 4 | 3 | 1 | 0 |
| leading_lines_to_subject | 12 | 6 | 5 | 1 |
| low_angle_subject_emphasis | 3 | 3 | 0 | 0 |
| portrait_landscape_orientation_reframe | 4 | 1 | 2 | 1 |
| rule_of_thirds_subject_placement | 10 | 4 | 5 | 1 |
| simplify_background_clutter | 23 | 9 | 14 | 0 |
| subject_background_separation | 26 | 3 | 23 | 0 |

## Top Canonical Issue Tags

| Value | Count |
| --- | ---: |
| better_crop | 132 |
| more_natural_framing | 132 |
| better_subject_prominence | 82 |
| composition_not_improved | 77 |
| better_balance | 57 |
| texture_artifacts | 47 |
| bad_empty_space | 39 |
| better_visual_focus | 38 |
| composition_worse | 25 |
| subject_deformed | 21 |
| cleaner_background | 20 |
| better_leading_lines | 15 |
| identity_changed | 11 |
| better_rule_of_thirds | 10 |
| over_cropping | 8 |
| background_changed_too_much | 5 |
| better_symmetry | 3 |
| important_content_missing | 2 |

## Top VLM Positive Tags

| Value | Count |
| --- | ---: |
| larger_subject_scale | 23 |
| subject_prominence | 22 |
| cleaner_edges | 20 |
| more_breathing_room | 20 |
| cleaner_framing | 18 |
| reduced_empty_space | 17 |
| cleaner_background | 16 |
| better_subject_prominence | 14 |
| stronger_subject_prominence | 14 |
| subject_more_prominent | 13 |
| tighter_crop | 12 |
| subject_shift | 9 |
| improved_balance | 8 |
| stronger_visual_focus | 7 |
| off_center_subject | 6 |
| improved_subject_prominence | 6 |
| reduced_clutter | 6 |
| improved_visual_balance | 6 |
| stronger_leading_lines | 6 |
| reduced_empty_margins | 6 |

## Top VLM Negative Tags

| Value | Count |
| --- | ---: |
| background_still_busy | 12 |
| minimal_recomposition | 6 |
| minor_generated_smoothing | 5 |
| composition change is subtle | 5 |
| subject_identity_changed | 5 |
| anatomy_distortion | 5 |
| identity_change | 4 |
| background_still_somewhat_busy | 4 |
| subject_still_centered | 4 |
| limited_recomposition | 4 |
| composition_change_subtle | 3 |
| slight_identity_softening | 3 |
| slightly tighter crop | 3 |
| subject_too_close_to_left_edge | 3 |
| subject_less_prominent | 3 |
| slightly_tight_bottom_crop | 3 |
| awkward_crop | 3 |
| weaker_subject_prominence | 3 |
| excessive_empty_sky | 3 |
| subject_still_small_in_frame | 3 |

## Recommended Use

- Treat this as the first generated-pair ReFrameJudge-v1 pilot subset.
- Use `overall_label` for win/tie/lose classification.
- Use `improvement_score` as a continuous weak preference score in [-2.0, 2.0].
- Use `clean_for_training` or `strict_clean_for_training` for cleaner training/evaluation subsets.
- Splits are source-level: all three edits for one source stay in the same split.
- Watermarks are intentionally ignored by the annotation policy; phone UI / screenshot cleanup should be filtered separately if needed.
