#!/usr/bin/env python3
"""Qwen VLM judge baseline for ReFrameJudge image pairs.

This script sends blind A/B image pairs to a Qwen vision model through an
OpenAI-compatible endpoint and asks it to return structured composition choices.
It is intended for small sampled evaluations first because VLM calls cost money.
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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABEL_TO_ID = {"lose": 0, "tie": 1, "win": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABELS = [LABEL_TO_ID["lose"], LABEL_TO_ID["tie"], LABEL_TO_ID["win"]]
BINARY_LABELS = [LABEL_TO_ID["lose"], LABEL_TO_ID["win"]]


DEFAULT_PROMPT = """You are an expert photographic composition judge.

You will be given two images:
1. Candidate A
2. Candidate B

Both candidates are alternative crops or framings of the same photo. Judge them only by photographic composition and visible image quality. Do not assume either candidate is intended to be better.

Evaluate:
- Composition: framing, subject placement, balance, crop, empty space, visual focus.
- Content coverage: main subject, identity, important objects, scene semantics.
- Visual naturalness: artifacts, lighting, perspective, texture, realism.

Return only one valid JSON object:
{
  "choice": "A|B|tie",
  "preference_score": -2,
  "composition_gain": 1,
  "content_coverage": 1,
  "visual_naturalness": 1,
  "issue_tags": [],
  "reason": ""
}

Rules:
- A: Candidate A is clearly better composed while preserving important content and visual quality.
- B: Candidate B is clearly better composed while preserving important content and visual quality.
- tie: no clear winner, or a composition advantage is offset by content/quality loss.
- preference_score must be an integer from -2 to 2, where negative favors A, positive favors B, and 0 means tie.
- composition_gain, content_coverage, and visual_naturalness must be integers from 1 to 5.
"""


BINARY_PROMPT = """You are an expert photographic composition judge.

You will be given two images:
1. Candidate A
2. Candidate B

Both candidates are alternative crops or framings of the same photo. Judge them only by photographic composition and visible image quality. Do not assume either candidate is intended to be better.

Evaluate:
- Composition: framing, subject placement, balance, crop, empty space, visual focus.
- Content coverage: main subject, identity, important objects, scene semantics.
- Visual naturalness: artifacts, lighting, perspective, texture, realism.

Return only one valid JSON object:
{
  "choice": "A|B",
  "preference_score": -2,
  "composition_gain": 1,
  "content_coverage": 1,
  "visual_naturalness": 1,
  "issue_tags": [],
  "reason": ""
}

Rules:
- Choose A if Candidate A has better photographic composition while preserving important content and visual quality.
- Choose B if Candidate B has better photographic composition while preserving important content and visual quality.
- You must choose A or B. Do not return tie.
- preference_score must be an integer from -2 to 2, where negative favors A and positive favors B. Do not return 0 unless the API cannot evaluate the pair.
- composition_gain, content_coverage, and visual_naturalness must be integers from 1 to 5.
"""


def read_jsonl(path, max_samples=None, seed=42):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_samples is not None and max_samples < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)
        records.sort(key=lambda record: record["id"])
    return records


def encode_image(path):
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def load_prompt(path, label_mode):
    if path is None:
        return BINARY_PROMPT if label_mode == "binary" else DEFAULT_PROMPT
    return path.read_text(encoding="utf-8")


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


def normalize_choice(raw, allow_tie):
    choice = str(raw.get("choice", "")).strip().upper()
    valid_choices = {"A", "B", "TIE"} if allow_tie else {"A", "B"}
    if choice not in valid_choices:
        legacy_label = str(raw.get("overall_label", "")).strip().lower()
        if allow_tie and legacy_label == "tie":
            choice = "TIE"
        elif legacy_label in {"win", "b"}:
            choice = "B"
        elif legacy_label in {"lose", "a"}:
            choice = "A"
        else:
            score = clamp_int(raw.get("preference_score"), -2, 2, 0)
            if score > 0:
                choice = "B"
            elif score < 0:
                choice = "A"
            elif allow_tie:
                choice = "TIE"
            else:
                choice = "B"

    return {
        "choice": "tie" if choice == "TIE" else choice,
        "preference_score": clamp_int(raw.get("preference_score"), -2, 2, 0),
        "composition_gain": clamp_int(raw.get("composition_gain"), 1, 5, 3),
        "content_coverage": clamp_int(
            raw.get("content_coverage", raw.get("content_preservation")),
            1,
            5,
            5,
        ),
        "visual_naturalness": clamp_int(raw.get("visual_naturalness"), 1, 5, 5),
        "issue_tags": raw.get("issue_tags", []) if isinstance(raw.get("issue_tags", []), list) else [],
        "reason": str(raw.get("reason", "")),
    }


def candidate_roles(record, seed, shuffle_order):
    if not shuffle_order:
        return "source", "edited"
    rng = random.Random(f"{seed}:{record['id']}")
    if rng.random() < 0.5:
        return "source", "edited"
    return "edited", "source"


def image_for_role(record, role):
    if role == "source":
        return record["source_image"]
    if role == "edited":
        return record["edited_image"]
    raise ValueError(f"Unknown candidate role: {role}")


def label_from_choice(choice, candidate_a_role, candidate_b_role):
    if choice == "tie":
        return "tie"
    winning_role = candidate_a_role if choice == "A" else candidate_b_role
    return "lose" if winning_role == "source" else "win"


def build_messages(record, project_root, prompt, candidate_a_role, candidate_b_role):
    candidate_a_path = project_root / image_for_role(record, candidate_a_role)
    candidate_b_path = project_root / image_for_role(record, candidate_b_role)
    return [
        {
            "role": "system",
            "content": "You are a strict image-pair evaluation model. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Candidate A:"},
                {"type": "image_url", "image_url": {"url": encode_image(candidate_a_path)}},
                {"type": "text", "text": "Candidate B:"},
                {"type": "image_url", "image_url": {"url": encode_image(candidate_b_path)}},
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
            )
            return response.choices[0].message.content, response.model_dump()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (2**attempt))
    raise last_error


def evaluate(predictions, label_mode):
    labels = BINARY_LABELS if label_mode == "binary" else LABELS
    y_true = [LABEL_TO_ID[item["true_label"]] for item in predictions]
    y_pred = [LABEL_TO_ID[item["pred_label"]] for item in predictions]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix_labels": [ID_TO_LABEL[label_id] for label_id in labels],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            ID_TO_LABEL[label_id]: {
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for label_id, class_precision, class_recall, class_f1, class_support in zip(
                labels,
                precision,
                recall,
                f1,
                support,
            )
        },
    }


def write_jsonl(path, records):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--label-mode", choices=["threeway", "binary"], default="threeway")
    parser.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key", help="Direct API key. Prefer --api-key-env for normal use.")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-order",
        action="store_true",
        help="Randomly assign each pair to Candidate A/B using --seed, then map the choice back.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--save-raw-responses",
        type=Path,
        help="Optional JSONL path for full API response payloads. This can be large.",
    )
    args = parser.parse_args()

    api_key = args.api_key if args.api_key else os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}=... or pass --api-key <key>")

    records = read_jsonl(args.input_jsonl, args.max_samples, args.seed)
    prompt = load_prompt(args.prompt_file, args.label_mode)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    predictions = []
    raw_responses = []
    for index, record in enumerate(records, 1):
        candidate_a_role, candidate_b_role = candidate_roles(record, args.seed, args.shuffle_order)
        messages = build_messages(
            record,
            args.project_root,
            prompt,
            candidate_a_role,
            candidate_b_role,
        )
        print(f"[{index}/{len(records)}] judging {record['id']}")
        try:
            content, raw_response = call_model(
                client,
                args.model,
                messages,
                args.temperature,
                args.max_tokens,
                args.retries,
                args.retry_sleep,
            )
            parsed = normalize_choice(extract_json(content), allow_tie=args.label_mode != "binary")
            error = ""
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            content = ""
            raw_response = {"error": str(exc)}
            parsed = {
                "choice": "A" if args.label_mode == "binary" else "tie",
                "preference_score": 0,
                "composition_gain": 3,
                "content_coverage": 5,
                "visual_naturalness": 5,
                "issue_tags": ["api_error"],
                "reason": str(exc),
            }
            error = str(exc)
        pred_label = label_from_choice(parsed["choice"], candidate_a_role, candidate_b_role)
        prediction = {
            "id": record["id"],
            "split": Path(args.input_jsonl).stem,
            "source_image": record["source_image"],
            "edited_image": record["edited_image"],
            "candidate_a_image": image_for_role(record, candidate_a_role),
            "candidate_b_image": image_for_role(record, candidate_b_role),
            "candidate_a_role": candidate_a_role,
            "candidate_b_role": candidate_b_role,
            "choice": parsed["choice"],
            "true_label": record["overall_label"],
            "pred_label": pred_label,
            "preference_score": parsed["preference_score"],
            "composition_gain": parsed["composition_gain"],
            "content_coverage": parsed["content_coverage"],
            "content_preservation": parsed["content_coverage"],
            "visual_naturalness": parsed["visual_naturalness"],
            "issue_tags": parsed["issue_tags"],
            "reason": parsed["reason"],
            "error": error,
            "correct": record["overall_label"] == pred_label,
            "target_vote_margin": record.get("vote_margin"),
            "target_preference_strength": record.get("preference_strength"),
            "notes": record.get("notes", ""),
        }
        predictions.append(prediction)
        raw_responses.append(
            {
                "id": record["id"],
                "candidate_a_role": candidate_a_role,
                "candidate_b_role": candidate_b_role,
                "content": content,
                "response": raw_response,
            }
        )

    result = {
        "model": "qwen_vlm_judge",
        "qwen_model": args.model,
        "base_url": args.base_url,
        "input_jsonl": str(args.input_jsonl),
        "records": len(predictions),
        "judge_mode": "blind_ab",
        "label_mode": args.label_mode,
        "shuffle_order": args.shuffle_order,
        "temperature": args.temperature,
        "metrics": evaluate(predictions, args.label_mode),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_jsonl(args.predictions_jsonl, predictions)
    write_jsonl(args.save_raw_responses, raw_responses)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
