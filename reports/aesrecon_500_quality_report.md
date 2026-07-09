# AesRecon-500 Quality Report

## Overview

- Source file: `outputs/aesrecon_gpt_composition_labels_500.jsonl`
- Frozen annotation file: `data/reframejudge_v1/annotations/aesrecon_500.jsonl`
- Total records: 500
- Main usable records (`high|medium` relevance and confidence): 456
- Strong composition records (`high/high`, score 2): 146
- Low composition relevance records: 44

## Label Distribution

| Value | Count |
| --- | ---: |
| win | 500 |

## Composition Relevance

| Value | Count |
| --- | ---: |
| medium | 308 |
| high | 148 |
| low | 44 |

## Label Confidence

| Value | Count |
| --- | ---: |
| medium | 328 |
| high | 172 |

## Composition Score

| Value | Count |
| --- | ---: |
| 1 | 296 |
| 2 | 164 |
| 0 | 40 |

## Composition Gain

| Value | Count |
| --- | ---: |
| 4 | 431 |
| 3 | 45 |
| 5 | 24 |

## Split Summary

```json
{
  "train": {
    "records": 400,
    "composition_relevance": {
      "medium": 247,
      "high": 118,
      "low": 35
    },
    "label_confidence": {
      "medium": 263,
      "high": 137
    },
    "composition_score": {
      "1": 237,
      "2": 131,
      "0": 32
    },
    "composition_gain": {
      "4": 345,
      "5": 19,
      "3": 36
    }
  },
  "val": {
    "records": 50,
    "composition_relevance": {
      "medium": 31,
      "high": 14,
      "low": 5
    },
    "label_confidence": {
      "medium": 34,
      "high": 16
    },
    "composition_score": {
      "1": 30,
      "2": 16,
      "0": 4
    },
    "composition_gain": {
      "4": 43,
      "5": 2,
      "3": 5
    }
  },
  "test": {
    "records": 50,
    "composition_relevance": {
      "medium": 30,
      "low": 4,
      "high": 16
    },
    "label_confidence": {
      "medium": 31,
      "high": 19
    },
    "composition_score": {
      "1": 29,
      "0": 4,
      "2": 17
    },
    "composition_gain": {
      "4": 43,
      "3": 4,
      "5": 3
    }
  }
}
```

## Top Canonical Issue Tags

| Value | Count |
| --- | ---: |
| better_crop | 488 |
| more_natural_framing | 488 |
| better_balance | 395 |
| better_subject_prominence | 288 |
| better_visual_focus | 178 |
| cleaner_background | 116 |
| better_leading_lines | 103 |
| bad_empty_space | 102 |
| over_cropping | 75 |
| composition_worse | 48 |
| composition_not_improved | 44 |
| better_symmetry | 13 |
| better_rule_of_thirds | 2 |

## Top GPT Positive Tags

| Value | Count |
| --- | ---: |
| better_subject_placement | 140 |
| cleaner_framing | 130 |
| improved_visual_balance | 120 |
| stronger_subject_prominence | 110 |
| better_subject_prominence | 104 |
| tighter_framing | 100 |
| clearer_visual_focus | 69 |
| cleaner_background | 69 |
| improved_balance | 65 |
| reduced_empty_space | 63 |
| better_framing | 51 |
| better_use_of_negative_space | 49 |
| tighter_crop | 40 |
| cleaner_crop | 38 |
| more_dynamic_framing | 37 |
| leading_lines | 37 |
| stronger_visual_focus | 31 |
| better_visual_balance | 29 |
| reduced_distractions | 28 |
| clearer_subject_prominence | 27 |

## Top GPT Negative Tags

| Value | Count |
| --- | ---: |
| busy_background | 135 |
| subject_still_centered | 42 |
| slightly_busy_background | 37 |
| tight_crop | 15 |
| centered_composition | 14 |
| tight_edge_crop | 12 |
| background_still_busy | 12 |
| subject_still_small_in_frame | 11 |
| text_overlay_distraction | 10 |
| foreground_obstruction | 10 |
| centered_subject | 10 |
| subject_still_near_center | 9 |
| text_overlay | 8 |
| background_still_somewhat_busy | 8 |
| tight_edge_cropping | 7 |
| large_empty_sky | 7 |
| background_remains_busy | 7 |
| slightly_tight_crop | 7 |
| centered_static_composition | 7 |
| tight_edge_spacing | 7 |

## Recommended Use

- Use all 500 records as the AesRecon real-photo composition preference subset.
- Treat `overall_label=win` as dataset-direction supervision from AesRecon.
- Treat `composition_relevance`, `label_confidence`, `composition_score`, `composition_gain`, tags, and `reason` as GPT weak composition annotations.
- For cleaner composition-specific training, prefer records where relevance and confidence are both `high` or `medium`.
- Keep low-relevance records as useful hard/weak positives: they prevent the evaluator from assuming every preferred image is a strong composition improvement.
