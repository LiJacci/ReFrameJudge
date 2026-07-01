# ReFrameJudge

ReFrameJudge is a pairwise evaluation project for generated image recomposition.

The core question is:

> Given a source image and a reframed/generated image, does the edited image improve photographic composition while preserving the main content?

## Task

Input:

```text
I_src: source image
I_edit: recomposed/generated image
```

Output:

```text
overall_label: win / tie / lose
improvement_score: -2 to 2
composition_gain: 1 to 5
content_preservation: 1 to 5
visual_naturalness: 1 to 5
issue_tags: list of positive or negative tags
```

## Directory Structure

```text
ReFrameJudge/
  data/
    raw/
    pairs/
      images/
        source/
        edit/
      annotations/
    metadata/
  scripts/
  baselines/
  docs/
```

## First Milestone

Build a small, high-quality MVP dataset:

```text
100-300 source/edit pairs
win / tie / lose labels
multi-dimensional scores
issue tags
dataset validation script
```

