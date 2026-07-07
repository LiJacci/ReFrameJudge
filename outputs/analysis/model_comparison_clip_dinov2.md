# CLIP vs DINOv2 vs Fusion Error Analysis

Date: 2026-07-07

## Metrics

| Model | Val Acc | Val AUC | Test Acc | Test AUC |
|---|---:|---:|---:|---:|
| CLIP + LogReg | 0.680 | 0.7695 | 0.672 | 0.7202 |
| DINOv2 + LogReg | 0.560 | 0.5841 | 0.616 | 0.6343 |
| CLIP + DINOv2 Fusion | 0.596 | 0.6447 | 0.624 | 0.6730 |

## Error Review Sample

Both error review pages contain 100 sampled mistakes.

| Model | FP | FN | Avg FP Prob | Avg FN Prob |
|---|---:|---:|---:|---:|
| CLIP | 49 | 51 | 0.7562 | 0.2355 |
| DINOv2 | 47 | 53 | 0.8844 | 0.1278 |
| Fusion | 44 | 56 | 0.8936 | 0.0838 |

The DINOv2 errors are more confident: it has more high-confidence wrong FP/FN examples than CLIP.

## Preference Strength

Error vote-margin distribution:

```text
CLIP:
margin=1: 54
margin=3: 32
margin=5: 14

DINOv2:
margin=1: 36
margin=3: 48
margin=5: 16

Fusion:
margin=1: 43
margin=3: 33
margin=5: 24
```

CLIP mistakes are more concentrated in weak-preference 3:2 / 2:3 pairs. DINOv2 makes more mistakes on stronger 4:1 / 1:4 pairs.
Fusion also makes many strong-preference mistakes, suggesting simple feature concatenation does not reliably recover CLIP's stronger signal.

## Error Overlap

Pairwise overlap by sample id:

```text
CLIP vs DINOv2: 14
CLIP vs Fusion: 17
DINOv2 vs Fusion: 25
all three: 6
```

Pairwise overlap by `photo_id + pair_index`:

```text
CLIP vs DINOv2: 18
CLIP vs Fusion: 25
DINOv2 vs Fusion: 32
all three: 9
```

Fusion errors overlap more with DINOv2 than with CLIP, which matches the qualitative observation that DINOv2's high-confidence visual-structure errors can pull the fused classifier away from CLIP's better semantic signal.

## Qualitative Patterns

CLIP:

```text
1. Better at semantic/object-level preference than DINOv2.
2. Still weak on subtle crop shifts and pure composition differences.
3. Often prefers tighter, more centered crops even when humans prefer context.
```

DINOv2:

```text
1. More sensitive to texture and visual structure.
2. Often very confident but wrong on scenes where the semantic subject and context matter.
3. Struggles with composition preference direction when crops differ only by framing.
```

## Takeaway

CLIP is currently the strongest baseline for FCDB pairwise crop preference. DINOv2 provides different visual-structure signals, but simple feature concatenation does not improve performance.

Recommended next steps:

```text
1. Build a strong-preference split using only 5:0 and 4:1 pairs.
2. Treat 3:2 pairs as weak/tie instead of hard win/lose.
3. Add explicit crop geometry features if crop boxes are preserved.
4. Try late fusion / calibrated probability averaging instead of raw feature concatenation.
```
