# Qwen Blind A/B Judge Test500 Analysis

## Setup

Dataset: `data/pairs/annotations/fcdb_3way_test.jsonl`

Samples: 500 full test pairs

Model: Qwen VLM judge via OpenAI-compatible API

Prompt style: blind pairwise comparison with `Candidate A` and `Candidate B`; no source/edit wording. Pair order is shuffled and mapped back to `lose/tie/win`.

## Overall Result

| model | accuracy | macro F1 |
| --- | ---: | ---: |
| CLIP direct 3-way LogReg | 0.392 | 0.393 |
| CLIP threshold 3-way | 0.424 | 0.425 |
| CLIP score regression | 0.424 | 0.426 |
| Qwen source/edit prompt | 0.388 | 0.331 |
| Qwen blind A/B prompt | 0.366 | 0.302 |

The blind A/B prompt removes the previous edited-image bias, but it reveals a strong position bias toward Candidate B and almost never predicts `tie`.

## Confusion Matrix

Rows are true labels, columns are predicted labels.

| true \\ pred | lose | tie | win |
| --- | ---: | ---: | ---: |
| lose | 94 | 7 | 69 |
| tie | 83 | 3 | 74 |
| win | 79 | 5 | 86 |

Per-class metrics:

| class | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| lose | 0.367 | 0.553 | 0.441 | 170 |
| tie | 0.200 | 0.019 | 0.034 | 160 |
| win | 0.376 | 0.506 | 0.431 | 170 |

Predicted label distribution after mapping back:

| label | count |
| --- | ---: |
| lose | 256 |
| tie | 15 |
| win | 229 |

Raw Qwen choices:

| choice | count |
| --- | ---: |
| A | 88 |
| B | 397 |
| tie | 15 |

Qwen chooses Candidate B for 79.4% of samples.

## Order Bias

Because `--shuffle-order` randomizes which image appears as Candidate A/B, a pure Candidate-B bias becomes a noisy win/lose signal after mapping back.

When Candidate A is the original first image role:

- n = 233
- accuracy = 0.348
- raw choices: B = 184, A = 43, tie = 6

When Candidate A is the paired second image role:

- n = 267
- accuracy = 0.382
- raw choices: B = 213, A = 45, tie = 9

The raw choice distribution is similar regardless of which image is shown as Candidate A. This strongly suggests position bias rather than real pairwise composition preference.

## Breakdown By Vote Margin

| vote margin | true label | n | pred lose | pred tie | pred win | accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| -5 | lose | 71 | 49 | 4 | 18 | 0.690 |
| -3 | lose | 99 | 45 | 3 | 51 | 0.455 |
| -1 | tie | 80 | 41 | 2 | 37 | 0.025 |
| 1 | tie | 80 | 42 | 1 | 37 | 0.013 |
| 3 | win | 99 | 52 | 2 | 45 | 0.455 |
| 5 | win | 71 | 27 | 3 | 41 | 0.577 |

Compared with the source/edit prompt, blind A/B improves negative-margin examples because the second-image prior is gone. However, it destroys tie recall and remains highly unstable on moderate strong preferences.

## Interpretation

The blind prompt solves one bias but exposes another:

1. The source/edit prompt biased Qwen toward the image described as edited/reframed.
2. The blind A/B prompt biases Qwen toward Candidate B.
3. Qwen almost never uses `tie`, even though weak FCDB pairs are exactly where uncertainty should be common.

This suggests that a single forced A/B VLM call is not reliable enough as a direct evaluator for FCDB-style composition labels.

## Recommended Next Step

Do not treat a single Qwen A/B call as the final judge. The next VLM protocol should explicitly measure and reduce order bias:

1. Run each pair twice:
   - order 1: A = first image, B = second image
   - order 2: A = second image, B = first image
2. Map both answers back to `lose/tie/win`.
3. If both orders agree, keep the label.
4. If orders disagree, mark as `tie` or `uncertain`.
5. Report:
   - single-order metrics
   - dual-order agreement rate
   - agreement-only accuracy/macro-F1
   - coverage after filtering uncertain pairs

This turns order bias from an uncontrolled failure into a measurable uncertainty signal.

A second useful prompt change is to explicitly encourage `tie`:

```text
Use tie whenever the two candidates are close, when the improvement is subtle, or when the better choice depends on subjective taste. Do not force a winner.
```

The larger project lesson is that VLMs may be more useful as annotators for high-confidence pair filtering or rationale generation than as direct one-shot labels under the current FCDB setup.
