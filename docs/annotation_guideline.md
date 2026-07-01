# ReFrameJudge Annotation Guideline

## Goal

Judge whether the edited image is a better recomposition of the source image.

Do not judge the edited image as a standalone pretty picture. Always compare it with the source image.

The key question is:

> Does the edited image improve composition while preserving the main content?

## Annotation Questions

### 1. Is the main content preserved?

Choose one:

```text
yes
partial
no
```

Consider:

```text
main subject identity
important objects
scene semantics
object shape
recognizable details
```

### 2. Is the composition better?

Choose one:

```text
better
same
worse
```

Consider:

```text
subject placement
visual balance
empty space
crop quality
framing
background distractions
leading lines
symmetry
rule of thirds
```

### 3. Is the edited image visually natural?

Choose one:

```text
good
acceptable
bad
```

Consider:

```text
artifacts
blur
distorted body or objects
lighting consistency
perspective
texture quality
AI-looking details
```

### 4. Final Label

Choose one:

```text
win
tie
lose
```

Use these rules:

```text
win:
  Composition is clearly improved.
  Main content is preserved.
  Visual quality is natural or acceptable.

tie:
  Composition is roughly unchanged.
  Or composition improves but content/quality losses offset the gain.
  Or the judgment is ambiguous.

lose:
  Composition is worse.
  Or the main content is not preserved.
  Or visual artifacts make the edit unusable.
```

## Score Rubric

### improvement_score

```text
-2: much worse
-1: slightly worse
 0: about the same
 1: slightly better
 2: much better
```

### composition_gain

```text
1: much worse composition
2: slightly worse composition
3: similar composition
4: better composition
5: much better composition
```

### content_preservation

```text
1: main content is lost or changed
2: major content damage
3: partial preservation
4: mostly preserved
5: fully preserved
```

### visual_naturalness

```text
1: severe artifacts
2: noticeable artifacts
3: acceptable but flawed
4: mostly natural
5: very natural
```

## Issue Tags

Negative tags:

```text
subject_cropped
subject_deformed
identity_changed
important_content_missing
background_changed_too_much
new_irrelevant_objects
composition_not_improved
composition_worse
bad_empty_space
over_cropping
unnatural_perspective
lighting_inconsistent
texture_artifacts
low_resolution_or_blur
```

Positive tags:

```text
better_subject_prominence
better_balance
better_crop
cleaner_background
better_rule_of_thirds
better_symmetry
better_leading_lines
better_visual_focus
more_natural_framing
```

