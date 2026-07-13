#!/usr/bin/env python3
"""Local Qwen2.5-VL judge baseline for ReFrameJudge-v1.

This script evaluates source/edited image pairs with an open-source VLM. It can
run the base model directly or load a LoRA adapter for the same prompt format.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import AutoProcessor


LABEL_TO_ID = {"lose": 0, "tie": 1, "win": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_IDS = [LABEL_TO_ID["lose"], LABEL_TO_ID["tie"], LABEL_TO_ID["win"]]
REGRESSION_TARGETS = [
    "improvement_score",
    "composition_gain",
    "content_preservation",
    "visual_naturalness",
]


DEFAULT_PROMPT = """You are an expert evaluator for photographic recomposition.

You will receive two images:
1. Source image: the original image.
2. Candidate image: a recomposed, cropped, edited, or generated version.

Judge whether the candidate is better than the source for composition-oriented image improvement. Consider:
- Composition: framing, subject placement, balance, crop, empty space, visual focus, leading lines.
- Content preservation: whether the main subject, identity, important objects, and scene semantics are preserved.
- Visual naturalness: artifacts, lighting, perspective, texture, realism, and awkward generated details.

Return only one valid JSON object:
{
  "overall_label": "win|tie|lose",
  "improvement_score": 0,
  "composition_gain": 3,
  "content_preservation": 5,
  "visual_naturalness": 5,
  "issue_tags": [],
  "reason": ""
}

Rules:
- "win": the candidate is clearly better overall.
- "tie": no clear winner, or composition gain is offset by content/quality loss.
- "lose": the candidate is worse overall.
- improvement_score must be an integer or float from -2 to 2.
- composition_gain, content_preservation, and visual_naturalness must be integers from 1 to 5.
- Keep reason concise.
"""


def read_jsonl(path, max_samples=None, seed=42):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("overall_label") in LABEL_TO_ID:
                    records.append(record)
    if max_samples is not None and max_samples < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)
        records.sort(key=lambda item: item["id"])
    return records


def resolve_image_path(project_root, image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_prompt(path):
    if path is None:
        return DEFAULT_PROMPT
    return path.read_text(encoding="utf-8")


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    return json.loads(match.group(0))


def clamp_float(value, minimum, maximum, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def clamp_int(value, minimum, maximum, default):
    return int(round(clamp_float(value, minimum, maximum, default)))


def normalize_output(raw):
    label = str(raw.get("overall_label", "")).strip().lower()
    if label not in LABEL_TO_ID:
        choice = str(raw.get("choice", "")).strip().lower()
        if choice in {"candidate", "candidate image", "edited", "b", "win"}:
            label = "win"
        elif choice in {"source", "source image", "a", "lose"}:
            label = "lose"
        else:
            score = clamp_float(raw.get("improvement_score", raw.get("preference_score")), -2, 2, 0)
            if score > 0.25:
                label = "win"
            elif score < -0.25:
                label = "lose"
            else:
                label = "tie"
    return {
        "overall_label": label,
        "improvement_score": clamp_float(raw.get("improvement_score", raw.get("preference_score")), -2, 2, 0),
        "composition_gain": clamp_int(raw.get("composition_gain"), 1, 5, 3),
        "content_preservation": clamp_int(
            raw.get("content_preservation", raw.get("content_coverage")),
            1,
            5,
            5,
        ),
        "visual_naturalness": clamp_int(raw.get("visual_naturalness"), 1, 5, 5),
        "issue_tags": raw.get("issue_tags", []) if isinstance(raw.get("issue_tags", []), list) else [],
        "reason": str(raw.get("reason", "")),
    }


def build_messages(record, project_root, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(resolve_image_path(project_root, record["source_image"]))},
                {"type": "image", "image": str(resolve_image_path(project_root, record["edited_image"]))},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_model(args):
    model_kwargs = {
        "device_map": args.device_map,
        "cache_dir": args.hf_cache_dir,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.torch_dtype != "auto":
        model_kwargs["dtype"] = getattr(torch, args.torch_dtype)
    else:
        model_kwargs["dtype"] = "auto"

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    except ImportError:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(args.model_name, **model_kwargs)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(args.adapter),
            local_files_only=args.local_files_only,
        )
    model.eval()
    return model


def generate_one(model, processor, record, project_root, prompt, args):
    from qwen_vl_utils import process_vision_info

    messages = build_messages(record, project_root, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
    else:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def label_metrics(predictions):
    y_true = [LABEL_TO_ID[item["true_label"]] for item in predictions]
    y_pred = [LABEL_TO_ID[item["pred_label"]] for item in predictions]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_labels": [ID_TO_LABEL[label_id] for label_id in LABEL_IDS],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist(),
        "per_class": {
            ID_TO_LABEL[label_id]: {
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for label_id, class_precision, class_recall, class_f1, class_support in zip(
                LABEL_IDS,
                precision,
                recall,
                f1,
                support,
            )
        },
    }


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def safe_corr(x, y, spearman=False):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if len(x) < 2:
        return None
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(predictions):
    metrics = {}
    for target in REGRESSION_TARGETS:
        true_values = [item["true_scores"][target] for item in predictions if item["true_scores"][target] is not None]
        pred_values = [item["pred_scores"][target] for item in predictions if item["true_scores"][target] is not None]
        if not true_values:
            metrics[target] = {"count": 0, "mae": None, "pearson": None, "spearman": None}
            continue
        metrics[target] = {
            "count": len(true_values),
            "mae": float(mean_absolute_error(true_values, pred_values)),
            "pearson": safe_corr(true_values, pred_values, spearman=False),
            "spearman": safe_corr(true_values, pred_values, spearman=True),
        }
    return metrics


def subset_metrics(predictions):
    by_subset = defaultdict(list)
    for item in predictions:
        by_subset[item.get("subset") or item.get("data_source") or "unknown"].append(item)
    return {
        subset: {
            "count": len(items),
            "overall_label": label_metrics(items),
            "regression": regression_metrics(items),
        }
        for subset, items in sorted(by_subset.items())
    }


def evaluate(predictions):
    return {
        "overall_label": label_metrics(predictions),
        "regression": regression_metrics(predictions),
        "by_subset": subset_metrics(predictions),
        "prediction_distribution": dict(Counter(item["pred_label"] for item in predictions)),
        "target_distribution": dict(Counter(item["true_label"] for item in predictions)),
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
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", type=Path, help="Optional LoRA adapter directory.")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("data/cache/huggingface"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Only load models from local cache (HF_HUB_OFFLINE). "
                             "Use after downloading weights via huggingface-cli or a mirror.")
    parser.add_argument("--hf-endpoint", type=str, default=None,
                        help="Hugging Face mirror endpoint (e.g. https://hf-mirror.com). "
                             "Alternatively set the HF_ENDPOINT environment variable.")
    args = parser.parse_args()

    if args.hf_endpoint:
        import os
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    records = read_jsonl(args.input_jsonl, args.max_samples, args.seed)
    prompt = load_prompt(args.prompt_file)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = load_model(args)

    predictions = []
    for record in tqdm(records, desc="Judge pairs"):
        try:
            raw_text = generate_one(model, processor, record, args.project_root.resolve(), prompt, args)
            parsed = normalize_output(extract_json(raw_text))
            error = ""
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            raw_text = ""
            parsed = {
                "overall_label": "tie",
                "improvement_score": 0.0,
                "composition_gain": 3,
                "content_preservation": 5,
                "visual_naturalness": 5,
                "issue_tags": ["inference_error"],
                "reason": str(exc),
            }
            error = str(exc)

        true_scores = {target: record.get(target) for target in REGRESSION_TARGETS}
        pred_scores = {target: parsed[target] for target in REGRESSION_TARGETS}
        predictions.append(
            {
                "id": record["id"],
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "true_label": record["overall_label"],
                "pred_label": parsed["overall_label"],
                "correct": record["overall_label"] == parsed["overall_label"],
                "true_scores": true_scores,
                "pred_scores": pred_scores,
                "issue_tags": parsed["issue_tags"],
                "reason": parsed["reason"],
                "raw_response": raw_text,
                "error": error,
                "subset": record.get("subset"),
                "data_source": record.get("data_source"),
                "pair_type": record.get("pair_type"),
                "edit_type": record.get("edit_type"),
            }
        )

    result = {
        "model": "qwen_vl_local_judge",
        "model_name": args.model_name,
        "adapter": str(args.adapter) if args.adapter else None,
        "input_jsonl": str(args.input_jsonl),
        "records": len(predictions),
        "max_samples": args.max_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "load_in_4bit": args.load_in_4bit,
        "metrics": evaluate(predictions),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_jsonl(args.predictions_jsonl, predictions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
