# CLIP vs DINOv2 Error Analysis

Date: 2026-07-07

## Metrics

| Model | Val Acc | Val AUC | Test Acc | Test AUC |
|---|---:|---:|---:|---:|
| CLIP + LogReg | 0.680 | 0.7695 | 0.672 | 0.7202 |
| DINOv2 + LogReg | 0.560 | 0.5841 | 0.616 | 0.6343 |

## Error Review Sample

Both error review pages contain 100 sampled mistakes.

| Model | FP | FN | Avg FP Prob | Avg FN Prob |
|---|---:|---:|---:|---:|
| CLIP | 49 | 51 | 0.7562 | 0.2355 |
| DINOv2 | 47 | 53 | 0.8844 | 0.1278 |

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
```

CLIP mistakes are more concentrated in weak-preference 3:2 / 2:3 pairs. DINOv2 makes more mistakes on stronger 4:1 / 1:4 pairs.

## Error Overlap

```text
overlap by sample id: 14 / 100
overlap by photo_id + pair_index: 18
```

The two models fail on largely different samples. This suggests an ensemble or feature fusion could improve performance.

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

CLIP is currently the stronger single baseline for FCDB pairwise crop preference. DINOv2 provides complementary visual-structure signals, but alone it is weaker.

Recommended next steps:

```text
1. Add a CLIP + DINOv2 feature-fusion baseline.
2. Build a strong-preference split using only 5:0 and 4:1 pairs.
3. Treat 3:2 pairs as weak/tie instead of hard win/lose.
4. Add explicit crop geometry features if crop boxes are preserved.
```

