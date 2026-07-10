# ReFrameGen Strong Composition Prompt Bank

This prompt bank is designed for visible image recomposition, not subtle retouching.

The previous prompt style was too conservative: it asked the image model to preserve the same subject, pose, scene, and naturalness while making only a mild composition improvement. In practice, Seedream often produced near-duplicate images.

The current prompt style uses explicit geometric operations:

```text
subject placement
canvas expansion
crop shift
subject scale
horizon placement
foreground/background layout
visible subject-background separation
high angle / low angle viewpoint
bird's-eye viewpoint
three-quarter side viewpoint
portrait / landscape reframing
```

## General Constraint

This constraint is prepended by `scripts/match_reframegen_prompts_vlm.py` when it writes generation prompts:

```text
Edit the input photo into a realistic recomposed photograph.
Keep the same main subject identity, clothing, and overall scene semantics.
The edited image should clearly change the composition compared with the input image.
You may adjust camera framing, crop, canvas size, subject scale, subject placement, and surrounding background layout.
You may extend or synthesize plausible background regions when needed for recomposition.
Do not change the main subject identity.
Do not remove the main subject.
Do not add new important subjects.
Do not turn the image into an illustration, painting, poster, or stylized artwork.
The result should look like a natural real photograph, but the composition should be visibly different from the input.
Keep the output pixel size and visual scale close to the original image; do not simply upscale the photo.
You may switch between portrait and landscape framing when the composition instruction calls for it.
```

## Strong Positive Prompts

### rule_of_thirds_subject_placement

```text
Recompose the photo using a clear rule-of-thirds layout.
Move the main subject away from the center and place them near the left or right vertical third line, or near a rule-of-thirds intersection.
Keep the subject fully visible.
Adjust the surrounding background, crop, or canvas so there is balanced space on the opposite side.
The final image should have visibly different subject placement from the input, not just a minor crop.
```

### active_space_directional_room

```text
Recompose the photo by creating clear active space in the direction the subject is facing, looking, walking, or moving.
Place the subject on one side of the frame.
Expand or synthesize realistic background space in front of the subject's gaze or movement direction, and reduce unnecessary space behind the subject.
The edited image should show an obvious new balance between the subject and open space.
```

### intentional_negative_space

```text
Recompose the photo into a stronger negative-space composition.
Make the main subject occupy a smaller portion of the frame, around 20-35% of the image height when appropriate.
Add or preserve a large clean area of realistic background around the subject.
Place the subject off-center so the empty space feels intentional.
The final image should look like a deliberate wide composition, not a close duplicate of the input.
```

### leading_lines_to_subject

```text
Recompose the photo to make visible leading lines guide attention toward the main subject.
Adjust the camera framing, crop, and subject placement so roads, railings, paths, walls, rivers, shadows, stairs, or architectural edges point toward the subject.
If needed, widen the frame or shift the crop to include more of these linear elements.
The edited image should clearly emphasize the leading-line structure, not merely preserve the original framing.
```

### frame_within_frame

```text
Recompose the photo using a clear frame-within-frame composition.
Place the main subject inside a visible natural frame, such as a doorway, window, arch, tree opening, railing structure, mirror, or car window.
Adjust the crop, camera distance, or surrounding background so the frame element clearly surrounds or contains the subject.
Do not add unrealistic objects, but you may extend plausible existing structures to complete the frame.
The edited image should make the framing structure visibly stronger than in the input.
```

### foreground_depth_layering

```text
Recompose the photo with a stronger foreground-middle-ground-background structure.
Adjust the framing so there is a visible foreground layer, the main subject in the middle ground, and a coherent background layer.
Use existing scene elements such as flowers, railing, rocks, branches, pavement, furniture, or architecture as foreground depth cues.
Keep the subject unobstructed.
The final composition should have noticeably more spatial depth than the input.
```

### centered_symmetry

```text
Recompose the photo into a clear centered symmetrical composition.
Place the main subject on the central vertical axis.
Align the background architecture, doorway, corridor, staircase, reflection, trees, or structural elements symmetrically around the subject.
Adjust the crop and camera position so both sides of the frame feel balanced.
The edited image should look intentionally symmetrical, not just slightly straighter.
```

### diagonal_dynamic_composition

```text
Recompose the photo with a clear diagonal visual structure.
Shift the camera framing or crop so the subject, road, railing, shoreline, shadow, staircase, or architectural edge forms a strong diagonal across the image.
Place the subject along or near this diagonal path.
The final image should show a more dynamic diagonal flow than the input composition.
```

### edge_control_clean_crop

```text
Recompose the photo with clear edge control and a cleaner crop.
Zoom out or expand the canvas if needed so the subject's head, hands, feet, joints, and important objects are not awkwardly cut off.
Remove or reduce distracting elements near the frame edges.
Give the subject visible breathing room on all important sides.
The edited image should clearly fix edge crowding or awkward cropping from the input.
```

### horizon_thirds_placement

```text
Recompose the photo with deliberate horizon placement.
Move the horizon, skyline, waterline, road boundary, or major horizontal boundary close to the upper or lower third of the image.
Avoid placing the horizon through the subject's head or exactly through the center of the frame.
Adjust crop or canvas expansion so the vertical balance is visibly different from the input.
```

### subject_background_separation

```text
Recompose the photo to create stronger subject-background separation.
Move the subject or adjust the camera framing so the subject no longer overlaps with distracting background objects.
Create cleaner space around the head and body.
Reduce visual tangents, mergers, and clutter directly behind the subject.
The edited image should make the subject stand out more clearly than in the input.
```

### fill_the_frame_subject_prominence

```text
Recompose the photo by making the main subject more dominant in the frame.
Increase the subject scale so the subject occupies about 45-65% of the image height when appropriate.
Reduce excessive empty margins and unnecessary background.
Keep the full important body parts visible unless a natural portrait crop is more appropriate.
The edited image should clearly shift from a loose composition to a stronger subject-focused composition.
```

### simplify_background_clutter

```text
Recompose the photo with a simpler, cleaner background.
Adjust the framing, crop, or subject placement to reduce background clutter and competing visual elements.
Move the subject away from busy background areas when possible.
Keep the same general setting, but make the background visually quieter and less distracting.
The edited image should show a clearer difference in background organization than the input.
```

### golden_ratio_visual_flow

```text
Recompose the photo with a golden-ratio-inspired visual flow.
Place the main subject noticeably off-center, around a golden-ratio focal area rather than the exact center.
Arrange surrounding space, lines, or background elements so the viewer's eye naturally moves toward the subject.
Adjust the crop or canvas so the image has a new organic visual flow.
The edited image should be visibly different from the input and not just a subtle cleanup.
```

### high_angle_environmental_view

```text
Recompose the photo from a higher camera angle, as if the camera is slightly above the subject and looking downward.
Use this viewpoint to show the subject's relationship to the surrounding environment, ground pattern, table surface, shoreline, road, stairs, or architectural layout.
Keep the main subject recognizable and naturally photographed.
The edited image should clearly feel less eye-level and more top-down than the input.
```

### low_angle_subject_emphasis

```text
Recompose the photo from a lower camera angle, as if the camera is below the subject's eye level and looking slightly upward.
Use the low viewpoint to make the subject feel more prominent, powerful, or sculptural while preserving identity and realistic perspective.
Include appropriate sky, ceiling, building, tree, or background structure above the subject when plausible.
The edited image should clearly feel less eye-level and more upward-looking than the input.
```

### bird_eye_top_down_layout

```text
Recompose the photo with a bird's-eye or near top-down viewpoint when plausible for the scene.
Show the subject and surrounding layout from above so paths, floors, tables, roads, water edges, shadows, or ground patterns become part of the composition.
Keep the result realistic and do not use this if a top-down camera position would be impossible for the scene.
The edited image should show a clearly different overhead spatial arrangement.
```

### three_quarter_side_view_depth

```text
Recompose the photo from a three-quarter side viewpoint instead of a flat straight-on view.
Shift the camera slightly left or right so the subject and background show more depth, side planes, diagonal lines, and foreground-background separation.
Keep the same subject and scene semantics.
The edited image should clearly reveal a more dimensional side angle than the input.
```

### portrait_landscape_orientation_reframe

```text
Recompose the photo by switching between portrait and landscape framing when it improves the composition.
If the input is vertical, create a natural horizontal composition with more side context; if the input is horizontal, create a natural vertical composition that emphasizes the subject and vertical structure.
Keep the output resolution and visual scale close to the original image rather than simply upscaling.
The edited image should clearly use a different frame orientation or aspect-ratio logic while preserving the same subject and scene.
```

## Pilot Recommendation

Do not regenerate all 150 candidates immediately. First run a 10-source / 20-candidate pilot:

```bash
python3 scripts/match_reframegen_prompts_vlm.py \
  --check-images \
  --limit-sources 10 \
  --top-k 2 \
  --candidate-tag seedream_strong \
  --output-image-dir data/reframejudge_v1/generated/reframegen_pilot_seedream_strong20/images \
  --output-jsonl data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_matched_20.jsonl \
  --raw-jsonl outputs/reframegen_prompt_matching_strong20_raw.jsonl
```

Then run Seedream from the strong matched manifest.

By default, `scripts/generate_reframegen_seedream.py` now uses `--size source`, which requests the input image pixel size instead of forcing a 2K output. Override `SEEDREAM_SIZE` only when a provider requires a fixed size.

## References

- Digital Photography School explains bird's-eye, high-angle, low-angle, and bug's-eye viewpoints as distinct photographic camera angles.
- B&H's viewpoint and perspective guide emphasizes changing elevation, left/right position, and distance to change foreground/background organization.
- Anton Gorlin's composition guide discusses frame shape, horizontal/vertical alignment, and matching orientation to dominant compositional lines.
- NYIP notes that low angle shots provide a different perspective and help images stand out.
