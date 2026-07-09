# ReFrameGen Pilot Source Manifest Report

## Overview

- Output manifest: `data/reframejudge_v1/source_manifests/reframegen_pilot_aesrecon_sources_50.jsonl`
- Selected sources: 50
- Available non-overlapping AesRecon test pairs: 403
- Excluded frozen AesRecon records: 500
- Seed: 20260709

## Generation Plan

- Use each `source_image` as the input image for generated recomposition.
- Generate 2 candidates per source for the pilot, for about 100 generated pairs.
- Do not use `paired_good_image` as a generated target; it is kept only for traceability.

## Recommended Task Mix

| Value | Count |
| --- | ---: |
| crop_reframe | 50 |
| outpainting_recomposition | 50 |
| subject_reposition | 50 |

## First 10 Sources

| id | source_record_id | source_image |
| --- | --- | --- |
| reframegen_pilot_source_000001 | aesrecon_test_000011 | `images/poor_images/5072_poor.jpg` |
| reframegen_pilot_source_000002 | aesrecon_test_000037 | `images/poor_images/5098_poor.jpg` |
| reframegen_pilot_source_000003 | aesrecon_test_000048 | `images/poor_images/5109_poor.jpg` |
| reframegen_pilot_source_000004 | aesrecon_test_000098 | `images/poor_images/5159_poor.jpg` |
| reframegen_pilot_source_000005 | aesrecon_test_000146 | `images/poor_images/5207_poor.jpg` |
| reframegen_pilot_source_000006 | aesrecon_test_000152 | `images/poor_images/5213_poor.jpg` |
| reframegen_pilot_source_000007 | aesrecon_test_000175 | `images/poor_images/5236_poor.jpg` |
| reframegen_pilot_source_000008 | aesrecon_test_000179 | `images/poor_images/5240_poor.jpg` |
| reframegen_pilot_source_000009 | aesrecon_test_000194 | `images/poor_images/5255_poor.jpg` |
| reframegen_pilot_source_000010 | aesrecon_test_000201 | `images/poor_images/5262_poor.jpg` |
