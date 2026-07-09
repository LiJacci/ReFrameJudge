#!/usr/bin/env python3
"""Generate ReFrameGen edited candidates with Seedream-style image APIs.

The script reads source images and positive composition prompts, randomly
assigns several composition principles to each source, and optionally calls an
OpenAI-compatible image generation/editing endpoint.

Use --dry-run first to create the generation manifest without spending API
credits.
"""

import argparse
import base64
import json
import mimetypes
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
DEFAULT_MODEL = "doubao-seedream-4-0"
GENERAL_CONSTRAINT = (
    "Edit the input photo as a realistic photograph. Preserve the same main subject, "
    "identity, clothing, important objects, and scene identity. Do not add new important "
    "objects. Do not remove the main subject. Do not change the person identity. Do not "
    "turn the image into an illustration, painting, poster, or stylized artwork. Keep "
    "the result natural and photographically realistic."
)


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


def build_prompt(prompt_record):
    return (
        f"{GENERAL_CONSTRAINT}\n\n"
        f"Composition principle: {prompt_record['principle']}.\n"
        f"Editing instruction: {prompt_record['prompt']}\n\n"
        "Make the composition change visible but not extreme. Keep the output as a natural "
        "real photo rather than a stylized reinterpretation."
    )


def choose_prompts(prompt_bank, source_id, prompts_per_source, seed):
    if prompts_per_source > len(prompt_bank):
        raise ValueError(
            f"prompts_per_source={prompts_per_source} exceeds prompt bank size={len(prompt_bank)}"
        )
    rng = random.Random(f"{seed}:{source_id}")
    selected = rng.sample(prompt_bank, prompts_per_source)
    selected.sort(key=lambda item: item["id"])
    return selected


def output_image_path(output_image_dir, candidate_id, extension):
    return output_image_dir / f"{candidate_id}{extension}"


def make_tasks(sources, prompt_bank, args):
    tasks = []
    for source_index, source in enumerate(sources, 1):
        if args.limit_sources is not None and source_index > args.limit_sources:
            break
        selected_prompts = choose_prompts(
            prompt_bank,
            source["id"],
            args.prompts_per_source,
            args.seed,
        )
        source_image_path = args.dataset_root / source["source_image"]
        if args.check_images and not source_image_path.exists():
            raise FileNotFoundError(source_image_path)

        for candidate_index, prompt_record in enumerate(selected_prompts, 1):
            if args.limit_tasks is not None and len(tasks) >= args.limit_tasks:
                return tasks
            candidate_id = f"{source['id']}_seedream_{candidate_index:02d}"
            tasks.append(
                {
                    "id": candidate_id,
                    "source_manifest_id": source["id"],
                    "source_dataset": source["source_dataset"],
                    "source_record_id": source["source_record_id"],
                    "source_image": source["source_image"],
                    "edited_image": str(output_image_path(args.output_image_dir, candidate_id, args.output_ext)),
                    "generation_model": args.model,
                    "generation_provider": "seedream",
                    "generation_task": "composition_recomposition",
                    "target_behavior": "positive_composition_edit",
                    "prompt_id": prompt_record["id"],
                    "composition_principle": prompt_record["principle"],
                    "expected_change": prompt_record.get("expected_change", []),
                    "generation_prompt": build_prompt(prompt_record),
                    "paired_good_image": source.get("paired_good_image", ""),
                    "notes": (
                        "Generated candidate for ReFrameGen pilot. The source image is from "
                        "AesRecon but excluded from AesRecon-500. Final win/tie/lose labels "
                        "must be assigned later by VLM or human review."
                    ),
                }
            )
    return tasks


def endpoint_url(base_url):
    return base_url.rstrip("/") + "/images/generations"


def build_payload(task, image_url, args):
    payload = {
        "model": args.model,
        "prompt": task["generation_prompt"],
        "n": 1,
        "size": args.size,
        "response_format": args.response_format,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.extra_payload:
        payload.update(json.loads(args.extra_payload))

    if args.image_field:
        if args.image_field.endswith("[]"):
            payload[args.image_field[:-2]] = [image_url]
        else:
            payload[args.image_field] = image_url
    return payload


def response_items(response_json):
    if isinstance(response_json.get("data"), list):
        return response_json["data"]
    if isinstance(response_json.get("images"), list):
        return response_json["images"]
    if isinstance(response_json.get("result"), dict):
        result = response_json["result"]
        if isinstance(result.get("data"), list):
            return result["data"]
        if isinstance(result.get("images"), list):
            return result["images"]
    return []


def save_image_from_item(item, output_path, timeout):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(item, str):
        if item.startswith("http://") or item.startswith("https://"):
            content = requests.get(item, timeout=timeout).content
            output_path.write_bytes(content)
            return "url"
        output_path.write_bytes(base64.b64decode(item))
        return "base64"

    b64_value = item.get("b64_json") or item.get("base64") or item.get("image_base64")
    if b64_value:
        if "," in b64_value and b64_value.strip().startswith("data:image/"):
            b64_value = b64_value.split(",", 1)[1]
        output_path.write_bytes(base64.b64decode(b64_value))
        return "base64"

    url = item.get("url") or item.get("image_url")
    if url:
        content = requests.get(url, timeout=timeout).content
        output_path.write_bytes(content)
        return "url"

    raise ValueError(f"No image content found in response item keys={sorted(item)}")


def call_seedream(task, args, session):
    image_url = image_data_url(args.dataset_root / task["source_image"])
    payload = build_payload(task, image_url, args)
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = session.post(
                endpoint_url(args.base_url),
                headers=headers,
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            response_json = response.json()
            items = response_items(response_json)
            if not items:
                raise ValueError(f"No image items in response: {response_json}")
            saved_from = save_image_from_item(items[0], Path(task["edited_image"]), args.timeout)
            return {
                "status": "ok",
                "saved_from": saved_from,
                "response": response_json,
                "request_payload": payload if args.save_request_payload else {},
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep * (2**attempt))
    return {
        "status": "error",
        "error": str(last_error),
        "request_payload": payload if args.save_request_payload else {},
    }


def read_existing_ids(path):
    if not path.exists():
        return set()
    return {record["id"] for record in read_jsonl(path)}


def write_summary(path, tasks, completed, errors, args):
    summary = {
        "tasks": len(tasks),
        "completed_this_run": completed,
        "errors_this_run": errors,
        "model": args.model,
        "base_url_host": urlparse(args.base_url).netloc,
        "prompts_per_source": args.prompts_per_source,
        "dry_run": args.dry_run,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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
        default=Path(os.getenv("AESRECON_DATASET_ROOT", "/Users/jacci_loopy/Downloads/AesRecon_dataset")),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/reframejudge_v1/generated_manifests/reframegen_pilot_seedream_250.jsonl"),
    )
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path("outputs/reframegen_seedream_raw.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/reframegen_seedream_summary.json"),
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=Path("data/reframejudge_v1/generated/reframegen_pilot_seedream/images"),
    )
    parser.add_argument("--output-ext", default=".png")
    parser.add_argument("--prompts-per-source", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--api-key", default=os.getenv("SEEDREAM_API_KEY") or os.getenv("ARK_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("SEEDREAM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("SEEDREAM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--image-field",
        default=os.getenv("SEEDREAM_IMAGE_FIELD", "image"),
        help="Payload field for the input image data URL. Use image_urls[] for an array field.",
    )
    parser.add_argument("--size", default=os.getenv("SEEDREAM_SIZE", "2K"))
    parser.add_argument("--response-format", default=os.getenv("SEEDREAM_RESPONSE_FORMAT", "b64_json"))
    parser.add_argument("--extra-payload", help="JSON object merged into every API payload.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-request-payload", action="store_true")
    args = parser.parse_args()

    sources = read_jsonl(args.source_manifest)
    prompt_bank = read_json(args.prompt_bank)
    tasks = make_tasks(sources, prompt_bank, args)

    if args.dry_run:
        write_jsonl(args.output_jsonl, tasks)
        write_summary(args.summary_json, tasks, 0, 0, args)
        print(json.dumps({"dry_run_tasks": len(tasks), "output_jsonl": str(args.output_jsonl)}, indent=2))
        return

    if not args.api_key:
        raise SystemExit("Missing API key. Set SEEDREAM_API_KEY or ARK_API_KEY, or pass --api-key.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = read_existing_ids(args.output_jsonl)
    completed = 0
    errors = 0
    session = requests.Session()

    with args.output_jsonl.open("a", encoding="utf-8") as out_handle, args.raw_jsonl.open(
        "a", encoding="utf-8"
    ) as raw_handle:
        for index, task in enumerate(tasks, 1):
            if task["id"] in existing_ids:
                continue
            print(f"[{index}/{len(tasks)}] generating {task['id']} ({task['prompt_id']})")
            result = call_seedream(task, args, session)
            record = dict(task)
            record["generation_status"] = result["status"]
            record["generation_error"] = result.get("error", "")
            record["generated_image_exists"] = Path(record["edited_image"]).exists()

            out_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            out_handle.flush()
            raw_handle.write(
                json.dumps(
                    {
                        "id": task["id"],
                        "status": result["status"],
                        "error": result.get("error", ""),
                        "response": result.get("response", {}),
                        "request_payload": result.get("request_payload", {}),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            raw_handle.flush()

            if result["status"] == "ok":
                completed += 1
            else:
                errors += 1
                if not args.continue_on_error:
                    raise SystemExit(result["error"])

    write_summary(args.summary_json, tasks, completed, errors, args)


if __name__ == "__main__":
    main()
