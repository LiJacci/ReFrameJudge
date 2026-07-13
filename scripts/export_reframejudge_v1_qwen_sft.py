#!/usr/bin/env python3
"""Export ReFrameJudge-v1 splits to Qwen-VL supervised fine-tuning JSONL."""

import argparse
import json
from pathlib import Path


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
"""


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(project_root, image_path, absolute):
    path = Path(image_path)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def target_json(record, include_reason):
    output = {
        "overall_label": record["overall_label"],
        "improvement_score": record.get("improvement_score", 0),
        "composition_gain": record.get("composition_gain", 3),
        "content_preservation": record.get("content_preservation", 5),
        "visual_naturalness": record.get("visual_naturalness", 5),
        "issue_tags": record.get("issue_tags", []),
    }
    if include_reason:
        output["reason"] = record.get("reason") or record.get("notes", "")
    else:
        output["reason"] = ""
    return json.dumps(output, ensure_ascii=False)


def convert_record(record, project_root, prompt, absolute_paths, include_reason):
    source = resolve_path(project_root, record["source_image"], absolute_paths)
    edited = resolve_path(project_root, record["edited_image"], absolute_paths)
    return {
        "id": record["id"],
        "messages": [
            {
                "role": "user",
                "content": "<image><image>\n" + prompt,
            },
            {
                "role": "assistant",
                "content": target_json(record, include_reason),
            },
        ],
        "images": [source, edited],
        "metadata": {
            "subset": record.get("subset"),
            "data_source": record.get("data_source"),
            "pair_type": record.get("pair_type"),
            "edit_type": record.get("edit_type"),
            "overall_label": record.get("overall_label"),
        },
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--include-reason", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    records = [
        convert_record(record, project_root, prompt, args.absolute_paths, args.include_reason)
        for record in read_jsonl(args.input_jsonl)
    ]
    write_jsonl(args.output_jsonl, records)
    print(json.dumps({"input": str(args.input_jsonl), "output": str(args.output_jsonl), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
