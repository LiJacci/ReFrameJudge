#!/usr/bin/env python3
"""Label generated ReFrameGen source/edit pairs with a VLM.

The input is a generation manifest produced by scripts/generate_reframegen_seedream.py.
For each generated source/edit pair, the VLM assigns composition preference labels
and training-usability diagnostics.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import traceback
from collections import Counter
from pathlib import Path


OPENAI_API_KEY_DEFAULT = ""
OPENAI_BASE_URL_DEFAULT = "https://www.aifast.club/v1"


PROMPT = """You are an expert photographic composition annotator for an image generation evaluation dataset.

You will receive two images:
1. Source image
2. Generated edited image

The edited image was generated from the source image with a composition-improvement instruction. Your task is to label whether the edited image is a better recomposition of the source image.

Always compare the edited image against the source image. Do not judge the edited image as a standalone pretty picture.

Judge composition-related changes:
- framing and crop
- subject placement and prominence
- visual balance
- empty space
- background cleanliness
- leading lines, symmetry, horizon placement, viewpoint, and visual focus when relevant

Also judge whether the generated image preserves the source content and looks like a natural photograph.

Important:
- Ignore small watermarks, provider marks, logos, or corner text when assigning overall_label, improvement_score, composition_gain, content_preservation, visual_naturalness, composition_relevance, and label_confidence. If a watermark exists, mention it only in negative_tags or reason. Only penalize it when it blocks important content or dominates the image.
- If the edited image mainly removes phone UI, black borders, screenshots, or other capture artifacts without a meaningful photographic recomposition, mark composition_relevance as low or label_confidence as low.
- If the main subject identity or scene semantics change, the final label should usually be tie or lose even when the edited image looks prettier.
- If composition improves but realism/content preservation is poor, do not mark it as win.
- Prefer tie when the change is too subtle, ambiguous, or mostly non-compositional.

Return only one valid JSON object:
{
  "overall_label": "win|tie|lose",
  "improvement_score": 0.0,
  "composition_gain": 3,
  "content_preservation": 5,
  "visual_naturalness": 5,
  "change_strength": 1,
  "composition_relevance": "high|medium|low",
  "label_confidence": "high|medium|low",
  "identity_preserved": true,
  "realism_ok": true,
  "artifact_issue": false,
  "positive_tags": [],
  "negative_tags": [],
  "reason": ""
}

Rules:
- overall_label:
  - "win" means the edited image clearly improves composition while preserving main content and acceptable visual realism.
  - "tie" means composition is similar, the change is too subtle, or gains are offset by content/quality issues.
  - "lose" means composition is worse, content is damaged, or generated artifacts make it unusable.
- improvement_score must be a number from -2.0 to 2.0, rounded to one decimal place:
  - -2.0 much worse, -1.0 slightly worse, 0.0 similar/ambiguous, 1.0 slightly better, 2.0 much better.
- composition_gain must be an integer from 1 to 5:
  - 1 much worse composition, 2 slightly worse, 3 similar, 4 better, 5 much better.
- content_preservation must be an integer from 1 to 5:
  - 1 main content lost, 3 partial preservation, 5 fully preserved.
- visual_naturalness must be an integer from 1 to 5:
  - 1 severe artifacts, 3 acceptable but flawed, 5 very natural.
- change_strength must be an integer from 0 to 3:
  - 0 almost no visible change, 1 small change, 2 clear recomposition, 3 strong recomposition/viewpoint/layout change.
- composition_relevance:
  - "high" when the pair is mainly about composition.
  - "medium" when composition matters but quality/content/UI cleanup also matters.
  - "low" when the difference is mostly not composition.
- label_confidence:
  - "high" only when the comparison is obvious.
  - "medium" when plausible but mixed.
  - "low" when hard to judge.
- Use positive_tags for visible composition improvements.
- Use negative_tags for remaining issues, artifacts, or reasons to filter.
"""


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def image_data_url(path):
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_path(path_value, root=None):
    path = Path(path_value)
    if path.is_absolute():
        return path
    if root is not None:
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


def build_messages(prompt, source_path, edited_path, record):
    context = {
        "pair_id": record["id"],
        "prompt_id": record.get("prompt_id", ""),
        "composition_principle": record.get("composition_principle", ""),
        "expected_change": record.get("expected_change", []),
        "match_reason": record.get("match_reason", ""),
    }
    return [
        {
            "role": "system",
            "content": "You are a strict composition dataset annotator. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "text",
                    "text": "Pair metadata for context only. Do not assume the edit succeeded:\n"
                    + json.dumps(context, ensure_ascii=True, indent=2),
                },
                {"type": "text", "text": "Source image:"},
                {"type": "image_url", "image_url": {"url": image_data_url(source_path)}},
                {"type": "text", "text": "Generated edited image:"},
                {"type": "image_url", "image_url": {"url": image_data_url(edited_path)}},
            ],
        },
    ]


def extract_json(text):
    if text is None:
        raise ValueError("Response content is None")
    text = text.strip()
    if not text:
        raise ValueError("Response content is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    return json.loads(match.group(0))


def clamp_int(value, minimum, maximum, default):
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def clamp_float(value, minimum, maximum, default, ndigits=1):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    value = max(minimum, min(maximum, value))
    return round(value, ndigits)


def normalize_level(value, valid_values, default):
    value = str(value).strip().lower()
    return value if value in valid_values else default


def normalize_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def normalize_list(value):
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_response(raw):
    label = normalize_level(raw.get("overall_label"), {"win", "tie", "lose"}, "tie")
    improvement_score = clamp_float(raw.get("improvement_score"), -2.0, 2.0, 0.0)
    composition_gain = clamp_int(raw.get("composition_gain"), 1, 5, 3)
    content_preservation = clamp_int(raw.get("content_preservation"), 1, 5, 3)
    visual_naturalness = clamp_int(raw.get("visual_naturalness"), 1, 5, 3)
    change_strength = clamp_int(raw.get("change_strength"), 0, 3, 1)
    artifact_issue = normalize_bool(raw.get("artifact_issue"), visual_naturalness <= 2)
    identity_preserved = normalize_bool(raw.get("identity_preserved"), content_preservation >= 4)
    realism_ok = normalize_bool(raw.get("realism_ok"), visual_naturalness >= 3)
    return {
        "overall_label": label,
        "improvement_score": improvement_score,
        "composition_gain": composition_gain,
        "content_preservation": content_preservation,
        "visual_naturalness": visual_naturalness,
        "change_strength": change_strength,
        "composition_relevance": normalize_level(
            raw.get("composition_relevance"),
            {"high", "medium", "low"},
            "medium",
        ),
        "label_confidence": normalize_level(
            raw.get("label_confidence"),
            {"high", "medium", "low"},
            "medium",
        ),
        "identity_preserved": identity_preserved,
        "realism_ok": realism_ok,
        "artifact_issue": artifact_issue,
        "positive_tags": normalize_list(raw.get("positive_tags", [])),
        "negative_tags": normalize_list(raw.get("negative_tags", [])),
        "reason": str(raw.get("reason", "")),
    }


def call_model(client, model, messages, temperature, max_tokens, retries, retry_sleep):
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content, response.model_dump()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[Call Error] attempt {attempt + 1}/{retries + 1}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if attempt < retries:
                sleep_time = retry_sleep * (2**attempt)
                print(f"  retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
    raise last_error


def read_existing_ids(path):
    if not path.exists():
        return set()
    ids = set()
    for record in read_jsonl(path):
        ids.add(record["id"])
    return ids


def write_summary(path, records, total_requested, skipped_existing):
    if path is None:
        return
    label_counts = Counter(record.get("overall_label", "error") for record in records)
    relevance_counts = Counter(record.get("composition_relevance", "unknown") for record in records)
    confidence_counts = Counter(record.get("label_confidence", "unknown") for record in records)
    change_counts = Counter(record.get("change_strength", "unknown") for record in records)
    summary = {
        "records_written_this_run": len(records),
        "total_requested": total_requested,
        "skipped_existing": skipped_existing,
        "label_counts": dict(label_counts),
        "composition_relevance_counts": dict(relevance_counts),
        "label_confidence_counts": dict(confidence_counts),
        "change_strength_counts": dict(change_counts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=Path("data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_strong_generated_150.jsonl"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.getenv("AESRECON_DATASET_ROOT", "../../shared/ai-camera/AesRecon_dataset")),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/reframejudge_v1/annotations/reframegen_seedream_strong150_vlm_labels_ignore_watermark.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/reframegen_seedream_strong150_vlm_label_ignore_watermark_summary.json"),
    )
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path("outputs/reframegen_seedream_strong150_vlm_label_ignore_watermark_raw.jsonl"),
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-5.4"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL_DEFAULT))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()

    api_key = args.api_key if args.api_key else os.getenv(args.api_key_env, OPENAI_API_KEY_DEFAULT)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}=... or pass --api-key.")

    prompt = PROMPT if args.prompt_file is None else args.prompt_file.read_text(encoding="utf-8")
    records = read_jsonl(args.generation_manifest)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    existing_ids = read_existing_ids(args.output_jsonl)
    skipped_existing = 0
    written_records = []

    if args.check_images:
        for record in records:
            source_path = resolve_path(record["source_image"], args.dataset_root)
            edited_path = resolve_path(record["edited_image"])
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            if not edited_path.exists():
                raise FileNotFoundError(edited_path)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("a", encoding="utf-8") as output_handle, args.raw_jsonl.open(
        "a", encoding="utf-8"
    ) as raw_handle:
        for index, record in enumerate(records, 1):
            if record["id"] in existing_ids:
                skipped_existing += 1
                continue

            source_path = resolve_path(record["source_image"], args.dataset_root)
            edited_path = resolve_path(record["edited_image"])
            for path in [source_path, edited_path]:
                if not path.exists():
                    raise FileNotFoundError(path)

            print(f"[{index}/{len(records)}] labeling {record['id']} ({record.get('prompt_id', '')})")
            try:
                content, raw_response = call_model(
                    client,
                    args.model,
                    build_messages(prompt, source_path, edited_path, record),
                    args.temperature,
                    args.max_tokens,
                    args.retries,
                    args.retry_sleep,
                )
                parsed = normalize_response(extract_json(content))
                error = ""
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                print(f"[Annotation Error] {record['id']}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                content = ""
                raw_response = {"error": str(exc)}
                parsed = {
                    "overall_label": "tie",
                    "improvement_score": 0.0,
                    "composition_gain": 3,
                    "content_preservation": 3,
                    "visual_naturalness": 3,
                    "change_strength": 0,
                    "composition_relevance": "low",
                    "label_confidence": "low",
                    "identity_preserved": False,
                    "realism_ok": False,
                    "artifact_issue": True,
                    "positive_tags": [],
                    "negative_tags": ["api_error"],
                    "reason": str(exc),
                }
                error = str(exc)

            issue_tags = sorted(set(parsed["positive_tags"] + parsed["negative_tags"]))
            output_record = {
                "id": record["id"],
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "edit_type": "generated_recomposition",
                "data_source": "ReFrameGen-Seedream",
                "pair_type": "generated_recomposition_pair",
                "label_source": f"{args.model}_vlm_composition_annotation",
                "annotation_policy": "composition_focused_ignore_small_watermarks",
                "overall_label": parsed["overall_label"],
                "improvement_score": parsed["improvement_score"],
                "composition_gain": parsed["composition_gain"],
                "content_preservation": parsed["content_preservation"],
                "visual_naturalness": parsed["visual_naturalness"],
                "change_strength": parsed["change_strength"],
                "composition_relevance": parsed["composition_relevance"],
                "label_confidence": parsed["label_confidence"],
                "identity_preserved": parsed["identity_preserved"],
                "realism_ok": parsed["realism_ok"],
                "artifact_issue": parsed["artifact_issue"],
                "issue_tags": issue_tags,
                "positive_tags": parsed["positive_tags"],
                "negative_tags": parsed["negative_tags"],
                "reason": parsed["reason"],
                "source_manifest_id": record.get("source_manifest_id", ""),
                "source_dataset": record.get("source_dataset", ""),
                "source_record_id": record.get("source_record_id", ""),
                "generation_model": record.get("generation_model", ""),
                "generation_provider": record.get("generation_provider", ""),
                "prompt_id": record.get("prompt_id", ""),
                "composition_principle": record.get("composition_principle", ""),
                "expected_change": record.get("expected_change", []),
                "match_reason": record.get("match_reason", ""),
                "paired_good_image": record.get("paired_good_image", ""),
                "error": error,
                "notes": (
                    "Generated ReFrameGen pair. overall_label is assigned by VLM by comparing source and "
                    "generated edit for composition improvement, content preservation, and visual realism."
                ),
            }
            output_handle.write(json.dumps(output_record, ensure_ascii=True) + "\n")
            output_handle.flush()
            written_records.append(output_record)

            raw_handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "content": content,
                        "response": raw_response,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            raw_handle.flush()

    write_summary(args.summary_json, written_records, len(records), skipped_existing)


if __name__ == "__main__":
    main()
