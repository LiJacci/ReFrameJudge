# CLIP 3-way Logistic Regression Analysis

## Setup

Dataset: `fcdb_3way_*`

Labels:

- `lose`: edited crop is worse than source crop
- `tie`: weak FCDB preference, originally 3:2 / 2:3 votes
- `win`: edited crop is better than source crop

Model: frozen CLIP image features + Logistic Regression

Feature: `[f_src, f_edit, f_edit - f_src, f_src * f_edit]`

## Result Summary

Validation:

- Accuracy: 0.380
- Macro F1: 0.374
- Weighted F1: 0.377
- OVR Macro ROC-AUC: 0.547

Test:

- Accuracy: 0.392
- Macro F1: 0.393
- Weighted F1: 0.394
- OVR Macro ROC-AUC: 0.538

The result is only modestly above random 3-way chance accuracy, which is about 0.333.

## Test Confusion Matrix

Rows are true labels, columns are predicted labels.

| true \\ pred | lose | tie | win |
| --- | ---: | ---: | ---: |
| lose | 65 | 67 | 38 |
| tie | 47 | 66 | 47 |
| win | 38 | 67 | 65 |

Per-class test metrics:

| class | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| lose | 0.433 | 0.382 | 0.406 | 170 |
| tie | 0.330 | 0.413 | 0.367 | 160 |
| win | 0.433 | 0.382 | 0.406 | 170 |

Predicted distribution on test:

- predicted `lose`: 150
- predicted `tie`: 200
- predicted `win`: 150

The classifier over-predicts `tie`, but the precision of `tie` is low.

## Interpretation

This direct 3-class baseline exposes a key issue: `tie` is not a clean visual class in FCDB. It means the human vote margin was weak, not necessarily that the two crops are visually identical or compositionally equivalent.

The model can learn some directional preference signal for `lose` versus `win`, but it struggles to separate weak-preference pairs from strong-preference pairs using global CLIP embeddings alone.

The symmetry of the confusion matrix is expected because the dataset includes reversed source/edit directions. For strong examples, reversing a pair flips `win` and `lose`; for weak examples mapped to `tie`, both directions become `tie`. This makes `tie` a margin/uncertainty decision rather than a normal class decision.

## Takeaway

The current CLIP 3-way Logistic Regression baseline should be treated as a diagnostic baseline, not as the main modeling path.

For ReFrameJudge, a better formulation is likely:

1. Learn a directional composition preference score: `s(source, edited)`.
2. Predict `win` if `s > tau`, `lose` if `s < -tau`, and `tie` if `abs(s) <= tau`.
3. Tune `tau` on validation data.

This matches the semantics of the task better than direct multiclass classification, because `tie` means no confident improvement rather than a third visual category.

## Recommended Next Step

Build Baseline 4: CLIP pairwise preference score + calibrated tie threshold.

Train binary preference on strong labels only, then use validation data to find a threshold that separates weak/tie cases from confident win/lose cases.
