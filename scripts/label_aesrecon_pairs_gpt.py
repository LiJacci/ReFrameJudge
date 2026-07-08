#!/usr/bin/env python3
"""Label AesRecon test image pairs with an OpenAI vision model.

The AesRecon test JSON maps a high-aesthetic image filename to a low-aesthetic
image filename. For ReFrameJudge-style composition labeling, this script treats
the low-aesthetic image as `source_image` and the high-aesthetic image as
`edited_image`, while showing the model only blind Candidate A/B images.
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


PROMPT = """You are an expert photographic composition judge.

You will be given two real photographs:
1. Candidate A
2. Candidate B

Both images are real photographs. Judge only composition quality. Do not judge generated-image artifacts, realism, or content preservation. Do not assume either candidate is intended to be better.

Consider:
- framing and crop
- subject placement and prominence
- visual balance
- empty space
- background cleanliness
- leading lines, symmetry, rule of thirds, and visual focus when relevant

Return only one valid JSON object:
{
  "choice": "A|B|tie",
  "composition_score": -2,
  "composition_gain": 1,
  "issue_tags": [],
  "reason": ""
}

Rules:
- Choose A if Candidate A has clearly better photographic composition.
- Choose B if Candidate B has clearly better photographic composition.
- Choose tie if neither candidate is clearly better, or the preference is mostly subjective.
- composition_score must be an integer from -2 to 2, where negative favors A, positive favors B, and 0 means tie.
- composition_gain must be an integer from 1 to 5, where 1 means Candidate B is much worse than Candidate A, 3 means similar, and 5 means Candidate B is much better than Candidate A.
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


def candidate_roles(pair_id, seed, shuffle_order):
    if not shuffle_order:
        return "poor", "good"
    rng = random.Random(f"{seed}:{pair_id}")
    if rng.random() < 0.5:
        return "poor", "good"
    return "good", "poor"


def path_for_role(dataset_root, pair, role):
    if role == "good":
        return dataset_root / "images" / "good_images" / pair["good_image_name"]
    if role == "poor":
        return dataset_root / "images" / "poor_images" / pair["poor_image_name"]
    raise ValueError(f"Unknown role: {role}")


def label_from_choice(choice, candidate_a_role, candidate_b_role):
    if choice == "tie":
        return "tie"
    winning_role = candidate_a_role if choice == "A" else candidate_b_role
    return "win" if winning_role == "good" else "lose"


def build_messages(prompt, candidate_a_path, candidate_b_path):
    return [
        {
            "role": "system",
            "content": "You are a strict image-pair composition evaluator. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Candidate A:"},
                {"type": "image_url", "image_url": {"url": image_data_url(candidate_a_path)}},
                {"type": "text", "text": "Candidate B:"},
                {"type": "image_url", "image_url": {"url": image_data_url(candidate_b_path)}},
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


def normalize_response(raw):
    choice = str(raw.get("choice", "")).strip().upper()
    if choice not in {"A", "B", "TIE"}:
        score = clamp_int(raw.get("composition_score"), -2, 2, 0)
        if score > 0:
            choice = "B"
        elif score < 0:
            choice = "A"
        else:
            choice = "TIE"
    return {
        "choice": "tie" if choice == "TIE" else choice,
        "composition_score": clamp_int(raw.get("composition_score"), -2, 2, 0),
        "composition_gain": clamp_int(raw.get("composition_gain"), 1, 5, 3),
        "issue_tags": raw.get("issue_tags", []) if isinstance(raw.get("issue_tags", []), list) else [],
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
    for record in records:
        label = record.get("overall_label", "error")
        counts[label] = counts.get(label, 0) + 1
    summary = {
        "records_written_this_run": len(records),
        "total_requested": total_requested,
        "skipped_existing": skipped_existing,
        "label_counts": counts,
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
    parser.add_argument("--shuffle-order", action="store_true", default=True)
    parser.add_argument("--no-shuffle-order", dest="shuffle_order", action="store_false")
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

            candidate_a_role, candidate_b_role = candidate_roles(
                pair["id"],
                args.seed,
                args.shuffle_order,
            )
            candidate_a_path = path_for_role(args.dataset_root, pair, candidate_a_role)
            candidate_b_path = path_for_role(args.dataset_root, pair, candidate_b_role)
            good_path = path_for_role(args.dataset_root, pair, "good")
            poor_path = path_for_role(args.dataset_root, pair, "poor")

            for image_path in [candidate_a_path, candidate_b_path]:
                if not image_path.exists():
                    raise FileNotFoundError(image_path)

            print(f"[{index}/{len(pairs)}] labeling {pair['id']}")
            try:
                content, raw_response = call_model(
                    client,
                    args.model,
                    build_messages(prompt, candidate_a_path, candidate_b_path),
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
                    "choice": "tie",
                    "composition_score": 0,
                    "composition_gain": 3,
                    "issue_tags": ["api_error"],
                    "reason": str(exc),
                }
                error = str(exc)

            label = label_from_choice(parsed["choice"], candidate_a_role, candidate_b_role)
            record = {
                "id": pair["id"],
                "source_image": str(poor_path),
                "edited_image": str(good_path),
                "edit_type": "aesthetic_reconstruction",
                "data_source": "AesRecon",
                "overall_label": label,
                "improvement_score": parsed["composition_score"],
                "composition_gain": parsed["composition_gain"],
                "content_preservation": 5,
                "visual_naturalness": 5,
                "issue_tags": parsed["issue_tags"],
                "reason": parsed["reason"],
                "candidate_a_image": str(candidate_a_path),
                "candidate_b_image": str(candidate_b_path),
                "candidate_a_role": candidate_a_role,
                "candidate_b_role": candidate_b_role,
                "choice": parsed["choice"],
                "good_image_name": pair["good_image_name"],
                "poor_image_name": pair["poor_image_name"],
                "expected_label": "win",
                "error": error,
                "notes": (
                    "AesRecon good/poor pair. Label is GPT-composed composition judgment only; "
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
