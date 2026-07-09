# ReFrameGen Composition Prompt Bank

This document collects photography-composition editing prompts for generating
ReFrameGen pilot candidates.

The goal is not generic beautification. The goal is to produce image edits with
clear, nameable composition changes, so that generated pairs can later be judged
by ReFrameJudge.

## General Editing Constraint

Use this constraint with every prompt unless the generation model requires a
shorter instruction.

```text
Edit the input photo as a realistic photograph.
Preserve the same main subject, identity, clothing, important objects, and scene identity.
Do not add new important objects.
Do not remove the main subject.
Do not change the person identity.
Do not turn the image into an illustration, painting, poster, or stylized artwork.
Keep the result natural and photographically realistic.
```

## Prompt 01: Rule Of Thirds Subject Placement

Principle:

```text
Rule of thirds
```

Use When:

```text
The subject is too centered, too static, or placed without visual tension.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same main subject, identity, clothing, pose, and scene.
Recompose the frame using rule-of-thirds placement:
position the main subject near a vertical third line or a rule-of-thirds intersection,
while keeping balanced surrounding space and clean frame edges.
Do not add new important objects or change the scene identity.
```

Expected Change:

```text
subject placement
visual balance
cleaner framing
```

Risk:

```text
May over-shift the subject or change the pose too much.
```

## Prompt 02: Active Space Directional Room

Principle:

```text
Active space / lead room
```

Use When:

```text
The subject is looking, walking, facing, or moving toward one side of the image.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and scene.
Recompose the image by adding active space in the direction the subject is looking,
facing, walking, or moving.
Avoid dead space behind the subject.
Keep the subject visually prominent and maintain a natural photographic look.
```

Expected Change:

```text
active space
directional balance
less cramped subject placement
```

Risk:

```text
May invent new background content in the added space.
```

## Prompt 03: Intentional Negative Space

Principle:

```text
Negative space / minimal composition
```

Use When:

```text
The background is busy or the subject lacks visual clarity.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same main subject and scene identity.
Recompose the image with intentional negative space around the subject.
Simplify the background and reduce visual clutter so the subject becomes clearer,
without changing who or what the subject is.
```

Expected Change:

```text
negative space
simpler background
stronger subject isolation
```

Risk:

```text
May remove meaningful context or make the scene too empty.
```

## Prompt 04: Leading Lines To Subject

Principle:

```text
Leading lines
```

Use When:

```text
The scene contains roads, railings, paths, walls, rivers, shadows, or architecture.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject, scene, and important objects.
Recompose the frame so existing roads, railings, paths, walls, rivers, shadows,
or architectural lines act as leading lines that guide the viewer's eye toward the main subject.
Keep the result natural and realistic.
```

Expected Change:

```text
leading lines
clearer visual flow
stronger focal direction
```

Risk:

```text
May distort architecture, roads, or background geometry.
```

## Prompt 05: Frame Within Frame

Principle:

```text
Frame within a frame
```

Use When:

```text
The image contains doors, windows, arches, trees, mirrors, railings, or car frames.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and scene.
Recompose the image using a frame-within-frame structure,
where existing elements such as a doorway, window, arch, tree branches, railing,
mirror, or car window naturally frame the main subject.
Avoid adding unrealistic objects.
```

Expected Change:

```text
natural framing
subject isolation
stronger visual focus
```

Risk:

```text
May add fake frame elements that were not present in the scene.
```

## Prompt 06: Foreground Depth Layering

Principle:

```text
Foreground, middle-ground, background layering
```

Use When:

```text
The photo feels flat or lacks depth.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same main subject and scene.
Recompose the image with clearer foreground, middle-ground, and background layering.
Use subtle foreground elements to create depth,
but keep the main subject unobstructed and visually dominant.
```

Expected Change:

```text
spatial depth
foreground layering
stronger subject separation
```

Risk:

```text
May introduce foreground occlusion or unwanted blur.
```

## Prompt 07: Centered Symmetry

Principle:

```text
Centered composition and symmetry
```

Use When:

```text
The scene has architecture, corridors, staircases, reflections, doors, or balanced structures.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and scene.
Recompose the frame with centered symmetry when suitable:
align the subject with a symmetrical background, corridor, doorway, staircase,
reflection, or architectural structure.
Keep both sides of the frame visually balanced.
```

Expected Change:

```text
symmetry
centered visual stability
architectural balance
```

Risk:

```text
Not suitable for scenes without naturally symmetrical structure.
```

## Prompt 08: Diagonal Dynamic Composition

Principle:

```text
Diagonals and dynamic visual structure
```

Use When:

```text
The image has paths, railings, shorelines, shadows, body lines, or architectural edges.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject, clothing, pose as much as possible, and scene identity.
Recompose the image with a stronger diagonal structure,
using the subject's body line, road, railing, shadow, shoreline,
or architectural edge to create a more dynamic visual flow.
```

Expected Change:

```text
diagonal structure
dynamic balance
visual movement
```

Risk:

```text
May alter the subject pose or rotate the scene unnaturally.
```

## Prompt 09: Edge Control Clean Crop

Principle:

```text
Edge control / clean cropping
```

Use When:

```text
The subject's head, hands, feet, joints, or important objects are awkwardly cut off.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and scene.
Recompose the frame with better edge control:
avoid awkwardly cutting off the head, hands, feet, joints, or important objects,
remove distracting elements near the frame edges,
and give the subject clean breathing room.
```

Expected Change:

```text
cleaner crop
better breathing room
reduced edge distractions
```

Risk:

```text
May zoom out too much or invent missing body parts.
```

## Prompt 10: Horizon Thirds Placement

Principle:

```text
Horizon placement
```

Use When:

```text
Outdoor photos with visible horizon, waterline, skyline, road boundary, or wall boundary.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and scene.
Recompose the image by placing the horizon or major horizontal boundary
near the upper or lower third of the frame,
instead of cutting through the subject or sitting awkwardly across the center.
Keep the subject clear and the scene natural.
```

Expected Change:

```text
horizon control
better vertical balance
less awkward background alignment
```

Risk:

```text
Not useful when the image has no clear horizon or horizontal boundary.
```

## Prompt 11: Subject Background Separation

Principle:

```text
Subject-background separation
```

Use When:

```text
The subject blends into the background or background objects intersect the subject.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same main subject and scene identity.
Recompose the image to improve subject-background separation:
reduce background clutter behind the subject,
avoid objects visually merging with the head or body,
and make the subject read clearly at first glance.
```

Expected Change:

```text
cleaner background
stronger subject isolation
clearer visual hierarchy
```

Risk:

```text
May remove scene context or over-blur the background.
```

## Prompt 12: Visual Balance With Secondary Element

Principle:

```text
Visual weight and balance
```

Use When:

```text
The main subject is too heavy on one side or the frame feels visually lopsided.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and scene.
Recompose the frame so the main subject is visually balanced
by a smaller secondary element, background structure, or area of negative space.
The secondary element should support the subject without competing for attention.
```

Expected Change:

```text
visual balance
controlled secondary element
more intentional spatial structure
```

Risk:

```text
May add or overemphasize secondary objects.
```

## Prompt 13: Fill The Frame Subject Prominence

Principle:

```text
Fill the frame
```

Use When:

```text
The subject is too small or the image has unnecessary empty margins.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject identity, clothing, expression, and scene.
Recompose by filling more of the frame with the main subject,
reducing unnecessary empty margins and background clutter,
while avoiding an overly tight or awkward crop.
```

Expected Change:

```text
larger subject
stronger prominence
less wasted space
```

Risk:

```text
May crop too tightly or lose environmental context.
```

## Prompt 14: Simplify Background Clutter

Principle:

```text
Simplicity and background control
```

Use When:

```text
The background has many competing objects, overlapping shapes, or distracting details.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and overall scene.
Recompose the image to simplify the background:
reduce distracting clutter, overlapping objects, and competing visual elements,
while keeping the main subject, setting, and natural photo style intact.
```

Expected Change:

```text
cleaner background
reduced clutter
clearer subject priority
```

Risk:

```text
May erase important contextual objects.
```

## Prompt 15: Golden Ratio Visual Flow

Principle:

```text
Golden ratio inspired visual flow
```

Use When:

```text
The image needs a less rigid off-center composition than rule of thirds.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same subject and scene.
Recompose the image with a golden-ratio-inspired visual flow:
place the main focal point slightly off-center
and arrange surrounding lines, space, or background elements
so the viewer's eye moves naturally toward the subject.
```

Expected Change:

```text
off-center focal point
curved or progressive visual flow
more organic balance
```

Risk:

```text
This instruction may be abstract for some image editing models.
```

## Prompt 16: Mild Bad Edge Composition

Principle:

```text
Hard negative / mild composition degradation
```

Use When:

```text
We need lose or weak-negative candidates for evaluator training.
```

Prompt:

```text
Edit the input photo as a realistic photograph.
Preserve the same main subject and scene,
but intentionally introduce a mild composition problem:
place the subject slightly too close to the frame edge,
leave unbalanced empty space,
or crop a secondary object awkwardly.
Do not create obvious artifacts or unrealistic content.
```

Expected Change:

```text
edge crowding
unbalanced empty space
awkward crop
```

Risk:

```text
If too strong, this creates trivial negatives rather than useful hard negatives.
```

## Suggested Pilot Selection

For the first ReFrameGen pilot, use 8 positive prompts and 2 negative or weak prompts.

Recommended positive prompts:

```text
rule_of_thirds_subject_placement
active_space_directional_room
intentional_negative_space
leading_lines_to_subject
frame_within_frame
foreground_depth_layering
edge_control_clean_crop
subject_background_separation
```

Recommended auxiliary prompts:

```text
visual_balance_with_secondary_element
mild_bad_edge_composition
```

## References

- PetaPixel, "28 Composition Techniques That Will Improve Your Photos": https://petapixel.com/photography-composition-techniques/
- Digital Photography School, "Rule of Thirds in Photography": https://digital-photography-school.com/rule-of-thirds/
- SLR Lounge, "Rule of Thirds Definition": https://www.slrlounge.com/glossary/rule-of-thirds-definition/
- Depositphotos, "10 Composition Rules in Photography": https://blog.depositphotos.com/back-to-basics-10-composition-rules-in-photography.html
- Digital Camera World, "I judged my local photo competition": https://www.digitalcameraworld.com/photography/awards-and-competitions/i-judged-my-local-photo-competition-here-are-my-top-tips-to-climb-up-the-ranks
