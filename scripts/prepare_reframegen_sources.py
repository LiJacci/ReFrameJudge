#!/usr/bin/env python3
"""Prepare source images for the ReFrameGen pilot subset.

The first pilot source pool uses AesRecon test pairs that were not included in
the frozen AesRecon-500 ReFrameJudge-v1 subset. We use the poor image as the
source image to be recomposed by a generation/editing model.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


GENERATION_TASKS = [
    "crop_reframe",
    "outpainting_recomposition",
    "subject_reposition",
]


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


def read_aesrecon_pairs(test_json):
    data = json.loads(test_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict good_image -> poor_image, got {type(data).__name__}")

    pairs = []
    for index, (good_name, poor_name) in enumerate(sorted(data.items()), 1):
        pairs.append(
            {
                "source_record_id": f"aesrecon_test_{index:06d}",
                "good_image_name": good_name,
                "poor_image_name": poor_name,
                "source_image": f"images/poor_images/{poor_name}",
                "paired_good_image": f"images/good_images/{good_name}",
            }
        )
    return pairs


def read_excluded_source_record_ids(paths):
    excluded = set()
    for path in paths:
        for record in read_jsonl(path):
            source_record_id = record.get("source_record_id")
            if source_record_id:
                excluded.add(source_record_id)
    return excluded


def check_images(records, dataset_root):
    missing = []
    for record in records:
        for field in ["source_image", "paired_good_image"]:
            image_path = dataset_root / record[field]
            if not image_path.exists():
                missing.append(str(image_path))
    return missing


def select_records(records, max_sources, seed):
    rng = random.Random(seed)
    selected = rng.sample(records, max_sources)
    selected.sort(key=lambda record: record["source_record_id"])
    return selected


def make_manifest_record(record, index, dataset_name, excluded_from):
    return {
        "id": f"reframegen_pilot_source_{index:06d}",
        "source_dataset": dataset_name,
        "source_record_id": record["source_record_id"],
        "source_image": record["source_image"],
        "source_image_name": record["poor_image_name"],
        "source_role": "lower_quality_reference",
        "paired_good_image": record["paired_good_image"],
        "paired_good_image_name": record["good_image_name"],
        "paired_good_role": "aesthetic_reference_only_not_generation_target",
        "excluded_from": excluded_from,
        "intended_pair_type": "generated_recomposition_pair",
        "recommended_generation_tasks": GENERATION_TASKS,
        "recommended_candidates_per_source": 2,
        "notes": (
            "Use source_image as the input for generation/editing. paired_good_image is kept "
            "only for traceability and should not be used as the generated target."
        ),
    }


def markdown_table(counts):
    lines = ["| Value | Count |", "| --- | ---: |"]
    for value, count in counts:
        lines.append(f"| {value} | {count} |")
    return "\n".join(lines)


def write_report(path, records, available_count, excluded_count, seed, output_path):
    task_counts = Counter()
    for record in records:
        task_counts.update(record["recommended_generation_tasks"])

    lines = [
        "# ReFrameGen Pilot Source Manifest Report",
        "",
        "## Overview",
        "",
        f"- Output manifest: `{output_path}`",
        f"- Selected sources: {len(records)}",
        f"- Available non-overlapping AesRecon test pairs: {available_count}",
        f"- Excluded frozen AesRecon records: {excluded_count}",
        f"- Seed: {seed}",
        "",
        "## Generation Plan",
        "",
        "- Use each `source_image` as the input image for generated recomposition.",
        "- Generate 2 candidates per source for the pilot, for about 100 generated pairs.",
        "- Do not use `paired_good_image` as a generated target; it is kept only for traceability.",
        "",
        "## Recommended Task Mix",
        "",
        markdown_table(task_counts.most_common()),
        "",
        "## First 10 Sources",
        "",
        "| id | source_record_id | source_image |",
        "| --- | --- | --- |",
    ]
    for record in records[:10]:
        lines.append(f"| {record['id']} | {record['source_record_id']} | `{record['source_image']}` |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../../shared/ai-camera/AesRecon_dataset"),
        help="Used only for optional image existence checks.",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=Path("../../shared/ai-camera/AesRecon_dataset/jsons/test/test.json"),
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[Path("data/reframejudge_v1/annotations/aesrecon_500.jsonl")],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reframejudge_v1/source_manifests/reframegen_pilot_aesrecon_sources_50.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/reframegen_pilot_sources_50.md"),
    )
    parser.add_argument("--dataset-name", default="AesRecon")
    parser.add_argument("--excluded-from", default="reframejudge_v1_aesrecon_500")
    parser.add_argument("--max-sources", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    pairs = read_aesrecon_pairs(args.test_json)
    excluded = read_excluded_source_record_ids(args.exclude_jsonl)
    available = [record for record in pairs if record["source_record_id"] not in excluded]

    if len(available) < args.max_sources:
        raise SystemExit(
            f"Not enough available records: requested {args.max_sources}, found {len(available)}"
        )

    selected = select_records(available, args.max_sources, args.seed)
    if args.check_images:
        missing = check_images(selected, args.dataset_root)
        if missing:
            raise SystemExit("Missing images:\n" + "\n".join(missing[:20]))

    manifest = [
        make_manifest_record(record, index, args.dataset_name, args.excluded_from)
        for index, record in enumerate(selected, 1)
    ]
    write_jsonl(args.output, manifest)
    write_report(args.report, manifest, len(available), len(excluded), args.seed, args.output)

    print(
        json.dumps(
            {
                "total_aesrecon_test_pairs": len(pairs),
                "excluded_records": len(excluded),
                "available_records": len(available),
                "selected_sources": len(manifest),
                "output": str(args.output),
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
