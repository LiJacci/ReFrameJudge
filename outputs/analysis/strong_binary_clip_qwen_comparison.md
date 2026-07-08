# Strong Binary CLIP vs Qwen Comparison

## Setup

Dataset: `data/pairs/annotations/fcdb_strong_test.jsonl`

Samples: 286 full strong-preference test pairs

Label distribution:

| label | count |
| --- | ---: |
| lose | 143 |
| win | 143 |

Vote-margin distribution:

| abs vote margin | count |
| ---: | ---: |
| 3 | 166 |
| 5 | 120 |

## Metric Format Check

The two result JSON files use different metric schemas, and this is expected.

`outputs/clip_logreg_fcdb_strong_5k.json` comes from `baselines/clip_logreg.py`, a binary probabilistic classifier. It reports:

- `accuracy`
- binary `precision`, `recall`, `f1` for the positive class `win`
- `macro_f1`
- `roc_auc`, because CLIP LogReg outputs `prob_win`
- confusion matrix in label order `[lose, win]`, from `label_mapping = {"lose": 0, "win": 1}`

`outputs/qwen_blind_ab_binary_fcdb_strong_test.json` comes from `baselines/qwen_vlm_judge.py`, a VLM label generator. It reports:

- `accuracy`
- `macro_precision`, `macro_recall`, `macro_f1`
- per-class precision/recall/F1
- confusion matrix in label order `[lose, win]`

Qwen does not report ROC-AUC because it does not output calibrated probabilities.

I recomputed the metrics from the prediction JSONL files, and both JSON files are internally consistent.

## Overall Comparison

| model | accuracy | macro F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| CLIP LogReg strong binary | 0.636 | 0.636 | 0.672 |
| Qwen blind A/B binary | 0.538 | 0.537 | n/a |

CLIP is clearly better on this binary strong-preference split.

## CLIP Strong Binary

Confusion matrix, rows are true labels and columns are predicted labels:

| true \\ pred | lose | win |
| --- | ---: | ---: |
| lose | 91 | 52 |
| win | 52 | 91 |

Metrics:

| metric | value |
| --- | ---: |
| accuracy | 0.636 |
| win precision | 0.636 |
| win recall | 0.636 |
| win F1 | 0.636 |
| macro F1 | 0.636 |
| ROC-AUC | 0.672 |

CLIP predictions are perfectly balanced:

| predicted label | count |
| --- | ---: |
| lose | 143 |
| win | 143 |

## Qwen Blind A/B Binary

Confusion matrix, rows are true labels and columns are predicted labels:

| true \\ pred | lose | win |
| --- | ---: | ---: |
| lose | 84 | 59 |
| win | 73 | 70 |

Metrics:

| class | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| lose | 0.535 | 0.587 | 0.560 | 143 |
| win | 0.543 | 0.490 | 0.515 | 143 |

Overall:

| metric | value |
| --- | ---: |
| accuracy | 0.538 |
| macro precision | 0.539 |
| macro recall | 0.538 |
| macro F1 | 0.537 |

Qwen prediction distribution after mapping back:

| predicted label | count |
| --- | ---: |
| lose | 157 |
| win | 129 |

Raw choices:

| choice | count |
| --- | ---: |
| A | 43 |
| B | 243 |

Qwen still has a very strong Candidate-B position bias: it chooses B in 85.0% of samples.

Because order is shuffled, this position bias partially maps to both `lose` and `win`, but it still harms real preference judgment.

## Breakdown By Vote Margin

Qwen binary by signed vote margin:

| vote margin | true label | n | pred lose | pred win | accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| -5 | lose | 60 | 42 | 18 | 0.700 |
| -3 | lose | 83 | 42 | 41 | 0.506 |
| 3 | win | 83 | 43 | 40 | 0.482 |
| 5 | win | 60 | 30 | 30 | 0.500 |

Qwen is not reliably better on stronger positive margins. The high `B` choice rate dominates the outcome.

## Interpretation

The binary setup removes the three-way `tie` failure, but Qwen still does not beat CLIP.

The main reason is not class imbalance; the test set is perfectly balanced. It is also not a metric mismatch; recomputation confirms the reported numbers.

The central failure remains order bias:

- Qwen chooses Candidate B 243 / 286 times.
- When Candidate A is the first image role, predicted `win` dominates.
- When Candidate A is the second image role, predicted `lose` dominates.

This means a large part of the prediction is explained by candidate position rather than photographic composition.

## Takeaway

For FCDB strong binary, the current best simple baseline remains CLIP LogReg.

Qwen should not be used as a single-pass A/B judge. If we keep exploring VLMs, the next protocol should be dual-order:

1. Query A/B.
2. Query B/A.
3. Map both answers back to `win/lose`.
4. Accept only consistent predictions.
5. Report coverage and accuracy on accepted examples.

This can turn position bias into a measurable confidence filter.
