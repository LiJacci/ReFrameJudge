# ReFrameJudge-v1 Combined Dataset Report

## Outputs

- Full combined annotation: `data/reframejudge_v1/annotations/reframejudge_v1_combined_full.jsonl`
- Balanced1000 annotation: `data/reframejudge_v1/annotations/reframejudge_v1_combined_balanced1000.jsonl`
- Balanced1000 train split: `data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_train.jsonl`
- Balanced1000 val split: `data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_val.jsonl`
- Balanced1000 test split: `data/reframejudge_v1/splits/reframejudge_v1_combined_balanced1000_test.jsonl`

## Full Combined

- Total records: 5650

| Subset | Records | Win | Tie | Lose |
| --- | ---: | ---: | ---: | ---: |
| fcdb | 5000 | 1511 | 1978 | 1511 |
| aesrecon_500 | 500 | 500 | 0 | 0 |
| reframegen_seedream_strong150 | 150 | 68 | 77 | 5 |

## Balanced1000

- Total records: 1000
- Sampling policy: FCDB is stratified by `overall_label`; AesRecon is randomly sampled from existing splits; ReFrameGen uses all 150 records.
- Split policy: existing per-subset splits are preserved; ReFrameGen remains source-level split.

| Subset | Records | Win | Tie | Lose |
| --- | ---: | ---: | ---: | ---: |
| fcdb | 500 | 152 | 198 | 150 |
| aesrecon_500 | 350 | 350 | 0 | 0 |
| reframegen_seedream_strong150 | 150 | 68 | 77 | 5 |

## Balanced1000 Split Summary

```json
{
  "train": {
    "records": 800,
    "subsets": {
      "aesrecon_500": 295,
      "fcdb": 400,
      "reframegen_seedream_strong150": 105
    },
    "labels": {
      "win": 462,
      "lose": 120,
      "tie": 218
    },
    "labels_by_subset": {
      "fcdb": {
        "lose": 118,
        "tie": 163,
        "win": 119
      },
      "aesrecon_500": {
        "win": 295
      },
      "reframegen_seedream_strong150": {
        "tie": 55,
        "win": 48,
        "lose": 2
      }
    }
  },
  "val": {
    "records": 100,
    "subsets": {
      "aesrecon_500": 35,
      "fcdb": 50,
      "reframegen_seedream_strong150": 15
    },
    "labels": {
      "win": 56,
      "lose": 15,
      "tie": 29
    },
    "labels_by_subset": {
      "fcdb": {
        "win": 16,
        "lose": 15,
        "tie": 19
      },
      "aesrecon_500": {
        "win": 35
      },
      "reframegen_seedream_strong150": {
        "tie": 10,
        "win": 5
      }
    }
  },
  "test": {
    "records": 100,
    "subsets": {
      "aesrecon_500": 20,
      "fcdb": 50,
      "reframegen_seedream_strong150": 30
    },
    "labels": {
      "win": 52,
      "lose": 20,
      "tie": 28
    },
    "labels_by_subset": {
      "fcdb": {
        "win": 17,
        "lose": 17,
        "tie": 16
      },
      "aesrecon_500": {
        "win": 20
      },
      "reframegen_seedream_strong150": {
        "tie": 12,
        "win": 15,
        "lose": 3
      }
    }
  }
}
```

## Recommended Use

- Use `reframejudge_v1_combined_full.jsonl` as the complete data pool.
- Use `reframejudge_v1_combined_balanced1000.jsonl` and its splits for main pilot training/evaluation.
- Report metrics both overall and by `subset`, because FCDB crop pairs, AesRecon real-photo pairs, and ReFrameGen generated pairs test different behavior.
- Avoid duplicating ReFrameGen records in the annotation file; use sampler weighting during training if generated-pair emphasis is needed.
