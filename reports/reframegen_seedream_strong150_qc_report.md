# ReFrameGen Seedream Strong150 QC

## Completeness

- Records: 150
- Unique ids: 150
- Sources: 50
- Per-source edits: 3 edits for each source
- Generation status: 150 ok
- Missing edited image files: 0

## Prompt Distribution

- edge_control_clean_crop: 28
- subject_background_separation: 26
- fill_the_frame_subject_prominence: 23
- simplify_background_clutter: 23
- leading_lines_to_subject: 12
- rule_of_thirds_subject_placement: 10
- active_space_directional_room: 6
- frame_within_frame: 5
- portrait_landscape_orientation_reframe: 4
- horizon_thirds_placement: 4
- low_angle_subject_emphasis: 3
- centered_symmetry: 3
- diagonal_dynamic_composition: 1
- foreground_depth_layering: 1
- bird_eye_top_down_layout: 1

## Applicability Score Distribution

- Score 3: 93
- Score 2: 57

## Visual QC Notes

Strong150 is visibly stronger than the earlier subtle-generation batch. Many examples show larger crop, scale, subject placement, or background simplification changes.

The current distribution is still dominated by local composition repairs: edge control, subject/background separation, fill frame, and simplify clutter. Viewpoint/orientation prompts are present but rare, so this batch is useful as a first generated-pair pilot but not yet balanced for viewpoint recomposition.

Some AesRecon sources are phone screenshots or contain black UI borders. Seedream often removes or normalizes those UI regions, which can confound pure composition evaluation because the model is changing both composition and screenshot artifacts.

The full local contact sheet is available at `outputs/analysis/reframegen_seedream_strong150_contact_sheet_full.jpg`.
