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
