#!/usr/bin/env python3
"""Match ReFrameGen source images to suitable composition editing prompts.

This script uses a vision-language model to diagnose each source image and rank
positive composition prompts by applicability. The output is a generation
manifest consumed by scripts/generate_reframegen_seedream.py.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path

OPENAI_API_KEY_DEFAULT = ""
OPENAI_BASE_URL_DEFAULT = "https://www.aifast.club/v1"

MATCHING_PROMPT = """You are a photography composition editor.

You will receive one source photograph and a list of positive composition editing prompts.
Your task is to decide which prompts are actually suitable for this image.

Do not choose prompts randomly. Choose prompts that can plausibly solve visible composition problems in the source image.
The source image does not need to already contain the final composition. A prompt is suitable if its editing operation could reasonably improve a visible composition issue while preserving subject identity and scene semantics.

Think in terms of visible composition problems and possible solutions:
- Centered, static, or weak subject placement can be improved by rule_of_thirds_subject_placement, golden_ratio_visual_flow, three_quarter_side_view_depth, or low_angle_subject_emphasis.
- A subject with gaze, facing, walking, or motion direction can be improved by active_space_directional_room or portrait_landscape_orientation_reframe.
- Busy background, visual clutter, or competing objects can be improved by simplify_background_clutter, subject_background_separation, high_angle_environmental_view, low_angle_subject_emphasis, or three_quarter_side_view_depth.
- Subject-background merging, tangents, or objects intersecting the head/body can be improved by subject_background_separation, high_angle_environmental_view, low_angle_subject_emphasis, or three_quarter_side_view_depth.
- Unclear spatial layout or weak depth can be improved by foreground_depth_layering, high_angle_environmental_view, bird_eye_top_down_layout, or three_quarter_side_view_depth.
- A subject that is too small or lacks presence can be improved by fill_the_frame_subject_prominence, low_angle_subject_emphasis, or portrait_landscape_orientation_reframe.
- Edge crowding, awkward crop, or excess dead space can be improved by edge_control_clean_crop, portrait_landscape_orientation_reframe, high_angle_environmental_view, or rule_of_thirds_subject_placement.
- A horizon, skyline, waterline, road boundary, or major horizontal structure can be improved by horizon_thirds_placement, high_angle_environmental_view, or low_angle_subject_emphasis.
- Roads, railings, paths, walls, rivers, stairs, shadows, shorelines, or architectural lines can be improved by leading_lines_to_subject or diagonal_dynamic_composition.
- Doors, windows, arches, trees, mirrors, railings, or car frames can be improved by frame_within_frame.
- Natural symmetry such as corridors, stairs, doors, reflections, bridges, or architecture can be improved by centered_symmetry.

Viewpoint/orientation prompts are normal composition solutions, not special cases:
- high_angle_environmental_view is suitable when a higher viewpoint could organize clutter, reduce excessive foreground, clarify subject-environment relationship, reveal ground/table/stair/road/shoreline layout, or separate the subject from a busy background.
- low_angle_subject_emphasis is suitable when the subject lacks prominence, appears too small or weak, blends into clutter, or when sky/ceiling/architecture/trees/open upper background can support an upward-looking composition.
- bird_eye_top_down_layout is suitable when a high-oblique or overhead view could clarify meaningful floor/table/road/plaza/shoreline/stairs/seating/food/object/shadow/ground patterns. It does not require a perfect vertical top-down view.
- three_quarter_side_view_depth is suitable when the image feels flat, front-facing, centered, static, or crowded, or when a side viewpoint could create depth, diagonal structure, parallax, or cleaner subject-background separation.
- portrait_landscape_orientation_reframe is suitable when the current aspect ratio creates excessive empty space, cuts off useful context, crowds the subject, lacks directional room, or does not match dominant scene structure.

Do not choose only the safest local crop edits. Consider both local composition fixes and stronger viewpoint/orientation recompositions when they can solve visible problems.

Return only one JSON object:
{
  "diagnosis": {
    "main_subject": "",
    "subject_size": "too_small|moderate|too_large|unclear",
    "subject_position": "centered|edge_crowded|off_center_good|unclear",
    "background_clutter": "low|medium|high",
    "edge_problem": true,
    "horizon_visible": false,
    "leading_lines_visible": false,
    "frame_elements_visible": false,
    "symmetry_possible": false,
    "directional_subject": false,
    "foreground_depth_possible": false,
    "subject_background_merging": false
  },
  "matches": [
    {
      "prompt_id": "",
      "applicability_score": 3,
      "match_reason": ""
    }
  ]
}

Rules:
- Return exactly the top prompts requested by the user.
- applicability_score must be 0, 1, 2, or 3.
- Use 3 only when the prompt is very suitable.
- Use 2 when the prompt is suitable.
- Do not include prompts with score 0.
- match_reason must cite visible image evidence.
"""


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_data_url(path):
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_json(text):
    if text is None:
        raise ValueError("Response content is None (model returned empty content)")
    text = text.strip()
    if not text:
        raise ValueError("Response content is empty string")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    return json.loads(match.group(0))


def prompt_bank_brief(prompt_bank):
    return [
        {
            "id": record["id"],
            "principle": record["principle"],
            "expected_change": record.get("expected_change", []),
        }
        for record in prompt_bank
    ]


def build_messages(source_image_path, prompt_bank, top_k):
    text = (
        MATCHING_PROMPT
        + "\n\n"
        + f"Return exactly {top_k} matched prompts.\n\n"
        + "Available prompts:\n"
        + json.dumps(prompt_bank_brief(prompt_bank), indent=2)
    )
    return [
        {
            "role": "system",
            "content": "You are a strict image composition prompt matcher. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url(source_image_path)}},
            ],
        },
    ]


def call_model(client, model, messages, temperature, max_tokens, retries, retry_sleep):
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content is None or not str(content).strip():
                raise ValueError(
                    "Model returned empty content (content=None or empty), "
                    "possibly due to content filtering or max_tokens truncation"
                )
            return content, response.model_dump()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (2**attempt))
    raise last_error


def normalize_matches(raw, prompt_ids, top_k):
    diagnosis = raw.get("diagnosis", {})
    matches = []
    seen = set()
    for item in raw.get("matches", []):
        prompt_id = str(item.get("prompt_id", "")).strip()
        if prompt_id not in prompt_ids or prompt_id in seen:
            continue
        try:
            score = int(item.get("applicability_score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(3, score))
        if score <= 0:
            continue
        seen.add(prompt_id)
        matches.append(
            {
                "prompt_id": prompt_id,
                "applicability_score": score,
                "match_reason": str(item.get("match_reason", "")),
            }
        )
    matches.sort(key=lambda item: (-item["applicability_score"], item["prompt_id"]))
    return diagnosis, matches[:top_k]


def build_generation_prompt(prompt_record):
    general_constraint = (
        "Edit the input photo into a realistic recomposed photograph. Keep the same main "
        "subject identity, clothing, and overall scene semantics. The edited image should "
        "clearly change the composition compared with the input image. You may adjust camera "
        "framing, crop, canvas size, subject scale, subject placement, and surrounding "
        "background layout. You may extend or synthesize plausible background regions when "
        "needed for recomposition. Do not change the main subject identity. Do not remove "
        "the main subject. Do not add new important subjects. Do not turn the image into an "
        "illustration, painting, poster, or stylized artwork. The result should look like a "
        "natural real photograph, but the composition should be visibly different from the input. "
        "Keep the output pixel size and visual scale close to the original image; do not simply "
        "upscale the photo. You may switch between portrait and landscape framing when the "
        "composition instruction calls for it."
    )
    return (
        f"{general_constraint}\n\n"
        f"Composition principle: {prompt_record['principle']}.\n"
        f"Editing instruction: {prompt_record['prompt']}\n\n"
        "Prioritize a visible geometric composition change over subtle retouching. Keep the "
        "output as a natural real photo rather than a stylized reinterpretation."
    )


def make_generation_records(
    source,
    diagnosis,
    matches,
    prompt_by_id,
    model,
    output_image_dir,
    output_ext,
    candidate_tag,
):
    records = []
    for index, match in enumerate(matches, 1):
        prompt_record = prompt_by_id[match["prompt_id"]]
        candidate_id = f"{source['id']}_{candidate_tag}_{index:02d}"
        records.append(
            {
                "id": candidate_id,
                "source_manifest_id": source["id"],
                "source_dataset": source["source_dataset"],
                "source_record_id": source["source_record_id"],
                "source_image": source["source_image"],
                "edited_image": str(output_image_dir / f"{candidate_id}{output_ext}"),
                "generation_model": model,
                "generation_provider": "seedream",
                "generation_task": "composition_recomposition",
                "target_behavior": "positive_composition_edit",
                "prompt_id": prompt_record["id"],
                "composition_principle": prompt_record["principle"],
                "expected_change": prompt_record.get("expected_change", []),
                "applicability_score": match["applicability_score"],
                "match_reason": match["match_reason"],
                "source_composition_diagnosis": diagnosis,
                "generation_prompt": build_generation_prompt(prompt_record),
                "paired_good_image": source.get("paired_good_image", ""),
                "notes": (
                    "Source-aware matched prompt for ReFrameGen pilot. Final win/tie/lose labels "
                    "must be assigned later by VLM or human review."
                ),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/reframejudge_v1/source_manifests/reframegen_pilot_aesrecon_sources_50.jsonl"),
    )
    parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=Path("data/reframejudge_v1/prompt_banks/reframegen_positive_composition_prompts.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.getenv("AESRECON_DATASET_ROOT", "../../shared/ai-camera/AesRecon_dataset")),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_matched_150.jsonl"),
    )
    parser.add_argument("--raw-jsonl", type=Path, default=Path("outputs/reframegen_prompt_matching_raw.jsonl"))
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=Path("data/reframejudge_v1/generated/reframegen_pilot_seedream/images"),
    )
    parser.add_argument("--output-ext", default=".png")
    parser.add_argument("--candidate-tag", default="seedream")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-5.4"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL_DEFAULT))
    parser.add_argument("--seedream-model", default=os.getenv("SEEDREAM_MODEL", "doubao-seedream-4-0"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key if args.api_key else os.getenv(args.api_key_env, OPENAI_API_KEY_DEFAULT)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}=... or pass --api-key.")

    prompt_bank = read_json(args.prompt_bank)
    prompt_ids = {record["id"] for record in prompt_bank}
    prompt_by_id = {record["id"]: record for record in prompt_bank}
    sources = read_jsonl(args.source_manifest)
    if args.limit_sources is not None:
        sources = sources[: args.limit_sources]

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    output_records = []
    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_jsonl.open("a", encoding="utf-8") as raw_handle:
        for index, source in enumerate(sources, 1):
            source_image_path = args.dataset_root / source["source_image"]
            if args.check_images and not source_image_path.exists():
                raise FileNotFoundError(source_image_path)
            print(f"[{index}/{len(sources)}] matching prompts for {source['id']}")
            try:
                content, raw_response = call_model(
                    client,
                    args.model,
                    build_messages(source_image_path, prompt_bank, args.top_k),
                    args.temperature,
                    args.max_tokens,
                    args.retries,
                    args.retry_sleep,
                )
                parsed = extract_json(content)
                diagnosis, matches = normalize_matches(parsed, prompt_ids, args.top_k)
                if len(matches) < args.top_k:
                    raise ValueError(f"Only {len(matches)} valid matches returned")
                error = ""
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                diagnosis = {"error": str(exc)}
                matches = []
                raw_response = {"error": str(exc)}
                content = ""
                error = str(exc)

            raw_handle.write(
                json.dumps(
                    {
                        "source_id": source["id"],
                        "content": content,
                        "response": raw_response,
                        "error": error,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            raw_handle.flush()
            output_records.extend(
                make_generation_records(
                    source,
                    diagnosis,
                    matches,
                    prompt_by_id,
                    args.seedream_model,
                    args.output_image_dir,
                    args.output_ext,
                    args.candidate_tag,
                )
            )

    write_jsonl(args.output_jsonl, output_records)
    print(
        json.dumps(
            {
                "sources": len(sources),
                "records": len(output_records),
                "top_k": args.top_k,
                "output_jsonl": str(args.output_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
