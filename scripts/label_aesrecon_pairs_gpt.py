#!/usr/bin/env python3
"""Add GPT composition annotations to AesRecon good/poor pairs.

The AesRecon test JSON maps a high-aesthetic image filename to a low-aesthetic
image filename. We use the dataset direction as the label:

    poor image -> good image = win

GPT is used to add weak composition annotations such as composition gain, tags,
confidence, and rationale. It is not asked to decide the winner.
"""

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI


PROMPT = """You are an expert photographic composition annotator.

You will be given two real photographs from an aesthetics reconstruction dataset:
1. Lower-quality reference image
2. Higher-quality candidate image

The dataset defines the higher-quality candidate as the preferred image. Your task is not to choose a winner. Your task is to annotate whether this known preference is explained by photographic composition, and to describe the composition improvement.

Judge only composition-related changes:
- framing and crop
- subject placement and prominence
- visual balance
- empty space
- background cleanliness
- leading lines, symmetry, rule of thirds, and visual focus when relevant

Do not base the annotation mainly on:
- semantic attractiveness of the subject
- color grading alone
- sharpness or resolution alone
- lighting mood alone
- whether the image content is more interesting

Return only one valid JSON object:
{
  "composition_relevance": "high|medium|low",
  "label_confidence": "high|medium|low",
  "composition_score": 1,
  "composition_gain": 4,
  "positive_tags": [],
  "negative_tags": [],
  "reason": ""
}

Rules:
- composition_relevance: whether the known preference is mainly about composition.
- label_confidence: how confident you are in the composition annotation.
- composition_score must be an integer from 0 to 2, where 0 means no clear composition improvement, 1 means slight/moderate improvement, and 2 means strong improvement.
- composition_gain must be an integer from 3 to 5, where 3 means composition is similar, 4 means better, and 5 means much better.
- Use positive_tags for composition improvements in the higher-quality candidate.
- Use negative_tags for remaining composition issues in the higher-quality candidate, if any.
"""


def read_pairs(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict good_image -> poor_image, got {type(data).__name__}")
    pairs = []
    for index, (good_name, poor_name) in enumerate(sorted(data.items()), 1):
        pairs.append(
            {
                "id": f"aesrecon_test_{index:06d}",
                "good_image_name": good_name,
                "poor_image_name": poor_name,
            }
        )
    return pairs


def select_pairs(pairs, max_samples, seed):
    if max_samples is None or max_samples >= len(pairs):
        return pairs
    rng = random.Random(seed)
    selected = rng.sample(pairs, max_samples)
    selected.sort(key=lambda item: item["id"])
    return selected


def image_data_url(path):
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_path(dataset_root, pair, quality):
    if quality == "good":
        return dataset_root / "images" / "good_images" / pair["good_image_name"]
    if quality == "poor":
        return dataset_root / "images" / "poor_images" / pair["poor_image_name"]
    raise ValueError(f"Unknown quality: {quality}")


def build_messages(prompt, poor_path, good_path):
    return [
        {
            "role": "system",
            "content": "You are a strict composition annotation assistant. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Lower-quality reference image:"},
                {"type": "image_url", "image_url": {"url": image_data_url(poor_path)}},
                {"type": "text", "text": "Higher-quality candidate image:"},
                {"type": "image_url", "image_url": {"url": image_data_url(good_path)}},
            ],
        },
    ]


def extract_json(text):
    text = text.strip()
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


def normalize_level(value, valid_values, default):
    value = str(value).strip().lower()
    if value not in valid_values:
        return default
    return value


def normalize_list(value):
    return value if isinstance(value, list) else []


def normalize_response(raw):
    return {
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
        "composition_score": clamp_int(raw.get("composition_score"), 0, 2, 1),
        "composition_gain": clamp_int(raw.get("composition_gain"), 3, 5, 4),
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
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content, response.model_dump()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (2**attempt))
    raise last_error


def read_existing_ids(output_jsonl):
    if output_jsonl is None or not output_jsonl.exists():
        return set()
    ids = set()
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def write_summary(path, records, total_requested, skipped_existing):
    if path is None:
        return
    counts = {}
    relevance_counts = {}
    confidence_counts = {}
    for record in records:
        label = record.get("overall_label", "error")
        counts[label] = counts.get(label, 0) + 1
        relevance = record.get("composition_relevance", "unknown")
        relevance_counts[relevance] = relevance_counts.get(relevance, 0) + 1
        confidence = record.get("label_confidence", "unknown")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
    summary = {
        "records_written_this_run": len(records),
        "total_requested": total_requested,
        "skipped_existing": skipped_existing,
        "label_counts": counts,
        "composition_relevance_counts": relevance_counts,
        "label_confidence_counts": confidence_counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/jacci_loopy/Downloads/AesRecon_dataset"),
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=Path("/Users/jacci_loopy/Downloads/AesRecon_dataset/jsons/test/test.json"),
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--raw-jsonl", type=Path)
    parser.add_argument("--model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-4o"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}=...")

    prompt = PROMPT if args.prompt_file is None else args.prompt_file.read_text(encoding="utf-8")
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    pairs = select_pairs(read_pairs(args.test_json), args.max_samples, args.seed)
    existing_ids = read_existing_ids(args.output_jsonl)
    skipped_existing = 0
    written_records = []

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = None
    if args.raw_jsonl is not None:
        args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
        raw_handle = args.raw_jsonl.open("a", encoding="utf-8")

    with args.output_jsonl.open("a", encoding="utf-8") as output_handle:
        for index, pair in enumerate(pairs, 1):
            if pair["id"] in existing_ids:
                skipped_existing += 1
                continue

            good_path = image_path(args.dataset_root, pair, "good")
            poor_path = image_path(args.dataset_root, pair, "poor")
            for path in [poor_path, good_path]:
                if not path.exists():
                    raise FileNotFoundError(path)

            print(f"[{index}/{len(pairs)}] annotating {pair['id']}")
            try:
                content, raw_response = call_model(
                    client,
                    args.model,
                    build_messages(prompt, poor_path, good_path),
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
                content = ""
                raw_response = {"error": str(exc)}
                parsed = {
                    "composition_relevance": "low",
                    "label_confidence": "low",
                    "composition_score": 1,
                    "composition_gain": 3,
                    "positive_tags": [],
                    "negative_tags": ["api_error"],
                    "reason": str(exc),
                }
                error = str(exc)

            issue_tags = list(parsed["positive_tags"]) + list(parsed["negative_tags"])
            record = {
                "id": pair["id"],
                "source_image": str(poor_path),
                "edited_image": str(good_path),
                "edit_type": "aesthetic_reconstruction",
                "data_source": "AesRecon",
                "pair_type": "real_photo_aesthetic_pair",
                "label_source": "dataset_direction+gpt_weak_composition_annotation",
                "overall_label": "win",
                "improvement_score": parsed["composition_score"],
                "composition_gain": parsed["composition_gain"],
                "content_preservation": 5,
                "visual_naturalness": 5,
                "issue_tags": issue_tags,
                "composition_relevance": parsed["composition_relevance"],
                "label_confidence": parsed["label_confidence"],
                "positive_tags": parsed["positive_tags"],
                "negative_tags": parsed["negative_tags"],
                "reason": parsed["reason"],
                "lower_quality_image": str(poor_path),
                "higher_quality_image": str(good_path),
                "good_image_name": pair["good_image_name"],
                "poor_image_name": pair["poor_image_name"],
                "expected_label": "win",
                "error": error,
                "notes": (
                    "AesRecon good/poor pair. overall_label follows dataset direction "
                    "(poor -> good = win). GPT adds weak composition annotations only. "
                    "content_preservation and visual_naturalness are fixed because both images are real photos."
                ),
            }
            output_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            output_handle.flush()
            written_records.append(record)

            if raw_handle is not None:
                raw_handle.write(
                    json.dumps(
                        {
                            "id": pair["id"],
                            "content": content,
                            "response": raw_response,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                raw_handle.flush()

    if raw_handle is not None:
        raw_handle.close()
    write_summary(args.summary_json, written_records, len(pairs), skipped_existing)


if __name__ == "__main__":
    main()
