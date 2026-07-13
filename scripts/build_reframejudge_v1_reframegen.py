#!/usr/bin/env python3
"""Freeze ReFrameGen VLM labels as a ReFrameJudge-v1 generated subset."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL_TAGS = [
    "better_subject_prominence",
    "better_balance",
    "better_crop",
    "cleaner_background",
    "better_rule_of_thirds",
    "better_symmetry",
    "better_leading_lines",
    "better_visual_focus",
    "more_natural_framing",
    "composition_not_improved",
    "composition_worse",
    "bad_empty_space",
    "over_cropping",
    "subject_deformed",
    "identity_changed",
    "important_content_missing",
    "background_changed_too_much",
    "unnatural_perspective",
    "texture_artifacts",
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


def tag_text(record):
    values = []
    values.extend(record.get("positive_tags", []))
    values.extend(record.get("negative_tags", []))
    values.append(record.get("reason", ""))
    return " ".join(str(value).lower().replace("-", "_") for value in values)


def negative_text(record):
    values = []
    values.extend(record.get("negative_tags", []))
    values.append(record.get("reason", ""))
    return " ".join(str(value).lower().replace("-", "_") for value in values)


def canonical_issue_tags(record):
    text = tag_text(record)
    negative = negative_text(record)
    tags = set()

    if any(token in text for token in ["subject_prominence", "larger_subject", "more_prominent"]):
        tags.add("better_subject_prominence")
    if any(token in text for token in ["balance", "balanced", "visual_balance"]):
        tags.add("better_balance")
    if any(token in text for token in ["crop", "framing", "frame_fill", "cleaner_edges", "breathing_room"]):
        tags.add("better_crop")
        tags.add("more_natural_framing")
    if any(token in text for token in ["cleaner_background", "reduced_clutter", "reduced_distraction", "background_simpl"]):
        tags.add("cleaner_background")
    if any(token in text for token in ["rule_of_thirds", "third"]):
        tags.add("better_rule_of_thirds")
    if any(token in text for token in ["symmetry", "symmetrical", "centered_symmetry"]):
        tags.add("better_symmetry")
    if any(token in text for token in ["leading_line", "leading_lines", "visual_flow"]):
        tags.add("better_leading_lines")
    if any(token in text for token in ["visual_focus", "subject_separation", "subject_background", "focal"]):
        tags.add("better_visual_focus")

    if record.get("overall_label") == "tie" or record.get("improvement_score", 0) == 0:
        tags.add("composition_not_improved")
    if record.get("overall_label") == "lose" or any(token in text for token in ["worse", "weaker", "awkward"]):
        tags.add("composition_worse")
    if any(token in text for token in ["empty_space", "empty_sky", "excessive_headroom", "dead_space"]):
        tags.add("bad_empty_space")
    if any(token in text for token in ["over_crop", "too_tight", "cut_off", "cropped_feet", "cropped_at"]):
        tags.add("over_cropping")
    if any(token in negative for token in ["anatomy", "body_proportion", "body_shape", "deformed", "distortion"]):
        tags.add("subject_deformed")
    if any(
        token in negative
        for token in [
            "identity_change",
            "identity_changed",
            "subject_identity_changed",
            "face_regenerated",
            "face_changed",
            "main_subject_changed",
        ]
    ):
        tags.add("identity_changed")
    if any(token in negative for token in ["content_changed", "content_missing", "scene_semantics"]):
        tags.add("important_content_missing")
    if any(token in negative for token in ["background_changed", "scene_reconstruction", "geometry_distortion"]):
        tags.add("background_changed_too_much")
    if any(token in negative for token in ["unnatural_perspective", "perspective_distortion", "viewpoint_unnatural"]):
        tags.add("unnatural_perspective")
    if any(token in negative for token in ["synthetic", "texture", "generated_smoothing", "painterly"]):
        tags.add("texture_artifacts")

    if not tags:
        if record.get("overall_label") == "win":
            tags.add("better_crop")
        else:
            tags.add("composition_not_improved")

    return [tag for tag in CANONICAL_TAGS if tag in tags]


def clean_flag(record):
    return (
        record["composition_relevance"] in {"high", "medium"}
        and record["label_confidence"] in {"high", "medium"}
        and record["identity_preserved"]
        and record["realism_ok"]
    )


def normalize_record(record, index, dataset_version):
    return {
        "id": f"rfj_v1_reframegen_seedream_strong150_{index:06d}",
        "source_image": record["source_image"],
        "edited_image": record["edited_image"],
        "edit_type": "generated_recomposition",
        "data_source": "ReFrameGen-Seedream",
        "pair_type": "generated_recomposition_pair",
        "dataset_version": dataset_version,
        "label_source": record["label_source"],
        "annotation_policy": record.get("annotation_policy", ""),
        "overall_label": record["overall_label"],
        "improvement_score": float(record["improvement_score"]),
        "composition_gain": int(record["composition_gain"]),
        "composition_relevance": record["composition_relevance"],
        "label_confidence": record["label_confidence"],
        "content_preservation": int(record["content_preservation"]),
        "visual_naturalness": int(record["visual_naturalness"]),
        "change_strength": int(record["change_strength"]),
        "identity_preserved": bool(record["identity_preserved"]),
        "realism_ok": bool(record["realism_ok"]),
        "artifact_issue": bool(record["artifact_issue"]),
        "clean_for_training": clean_flag(record),
        "strict_clean_for_training": clean_flag(record) and not record["artifact_issue"],
        "issue_tags": canonical_issue_tags(record),
        "positive_tags": record.get("positive_tags", []),
        "negative_tags": record.get("negative_tags", []),
        "reason": record.get("reason", ""),
        "notes": record.get("notes", ""),
        "source_record_id": record["id"],
        "source_manifest_id": record.get("source_manifest_id", ""),
        "source_dataset": record.get("source_dataset", ""),
        "source_aesrecon_record_id": record.get("source_record_id", ""),
        "generation_model": record.get("generation_model", ""),
        "generation_provider": record.get("generation_provider", ""),
        "prompt_id": record.get("prompt_id", ""),
        "composition_principle": record.get("composition_principle", ""),
        "expected_change": record.get("expected_change", []),
        "match_reason": record.get("match_reason", ""),
        "paired_good_image": record.get("paired_good_image", ""),
    }


def validate_records(records, expected_count):
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} records, got {len(records)}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate normalized ids found")
    source_groups = defaultdict(list)
    for record in records:
        if record["overall_label"] not in {"win", "tie", "lose"}:
            raise ValueError(f"Invalid label for {record['id']}: {record['overall_label']}")
        if not -2.0 <= record["improvement_score"] <= 2.0:
            raise ValueError(f"Invalid improvement_score for {record['id']}")
        if round(record["improvement_score"], 1) != record["improvement_score"]:
            raise ValueError(f"improvement_score must be one decimal for {record['id']}")
        if record["composition_gain"] not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Invalid composition_gain for {record['id']}")
        if record["composition_relevance"] not in {"high", "medium", "low"}:
            raise ValueError(f"Invalid composition_relevance for {record['id']}")
        if record["label_confidence"] not in {"high", "medium", "low"}:
            raise ValueError(f"Invalid label_confidence for {record['id']}")
        if not record["source_manifest_id"]:
            raise ValueError(f"Missing source_manifest_id for {record['id']}")
        if not record["reason"].strip():
            raise ValueError(f"Empty reason for {record['id']}")
        source_groups[record["source_manifest_id"]].append(record)
    if any(len(group) != 3 for group in source_groups.values()):
        raise ValueError("Expected exactly 3 edits per source group")


def split_by_source(records, train_sources, val_sources, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["source_manifest_id"]].append(record)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_keys = set(keys[:train_sources])
    val_keys = set(keys[train_sources : train_sources + val_sources])
    test_keys = set(keys[train_sources + val_sources :])

    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        if key in train_keys:
            split = "train"
        elif key in val_keys:
            split = "val"
        else:
            split = "test"
        splits[split].extend(sorted(groups[key], key=lambda record: record["id"]))
    return splits


def counter(records, field):
    return Counter(record[field] for record in records)


def top_tags(records, field, limit=20):
    counts = Counter()
    for record in records:
        counts.update(record.get(field, []))
    return counts.most_common(limit)


def markdown_table(counts):
    lines = ["| Value | Count |", "| --- | ---: |"]
    for value, count in counts:
        lines.append(f"| {value} | {count} |")
    return "\n".join(lines)


def split_summary(records):
    return {
        "records": len(records),
        "sources": len({record["source_manifest_id"] for record in records}),
        "labels": dict(counter(records, "overall_label")),
        "clean_for_training": dict(counter(records, "clean_for_training")),
        "strict_clean_for_training": dict(counter(records, "strict_clean_for_training")),
    }


def prompt_label_table(records):
    prompt_ids = sorted({record["prompt_id"] for record in records})
    lines = ["| Prompt | Records | Win | Tie | Lose |", "| --- | ---: | ---: | ---: | ---: |"]
    for prompt_id in prompt_ids:
        subset = [record for record in records if record["prompt_id"] == prompt_id]
        labels = counter(subset, "overall_label")
        lines.append(
            f"| {prompt_id} | {len(subset)} | {labels.get('win', 0)} | {labels.get('tie', 0)} | {labels.get('lose', 0)} |"
        )
    return "\n".join(lines)


def write_report(path, records, splits, source_path, output_path):
    clean = [record for record in records if record["clean_for_training"]]
    strict_clean = [record for record in records if record["strict_clean_for_training"]]
    lines = [
        "# ReFrameGen Seedream Strong150 Quality Report",
        "",
        "## Overview",
        "",
        f"- Source label file: `{source_path}`",
        f"- Frozen annotation file: `{output_path}`",
        f"- Total records: {len(records)}",
        f"- Source groups: {len({record['source_manifest_id'] for record in records})}",
        f"- Edits per source: 3",
        f"- Clean records: {len(clean)}",
        f"- Strict clean records: {len(strict_clean)}",
        "",
        "## Label Distribution",
        "",
        markdown_table(counter(records, "overall_label").most_common()),
        "",
        "## Improvement Score",
        "",
        markdown_table(sorted(counter(records, "improvement_score").items())),
        "",
        "## Composition Gain",
        "",
        markdown_table(sorted(counter(records, "composition_gain").items())),
        "",
        "## Composition Relevance",
        "",
        markdown_table(counter(records, "composition_relevance").most_common()),
        "",
        "## Label Confidence",
        "",
        markdown_table(counter(records, "label_confidence").most_common()),
        "",
        "## Clean Flags",
        "",
        markdown_table(counter(records, "clean_for_training").most_common()),
        "",
        "## Strict Clean Flags",
        "",
        markdown_table(counter(records, "strict_clean_for_training").most_common()),
        "",
        "## Split Summary",
        "",
        "```json",
        json.dumps({name: split_summary(split_records) for name, split_records in splits.items()}, indent=2),
        "```",
        "",
        "## Prompt By Label",
        "",
        prompt_label_table(records),
        "",
        "## Top Canonical Issue Tags",
        "",
        markdown_table(top_tags(records, "issue_tags")),
        "",
        "## Top VLM Positive Tags",
        "",
        markdown_table(top_tags(records, "positive_tags")),
        "",
        "## Top VLM Negative Tags",
        "",
        markdown_table(top_tags(records, "negative_tags")),
        "",
        "## Recommended Use",
        "",
        "- Treat this as the first generated-pair ReFrameJudge-v1 pilot subset.",
        "- Use `overall_label` for win/tie/lose classification.",
        "- Use `improvement_score` as a continuous weak preference score in [-2.0, 2.0].",
        "- Use `clean_for_training` or `strict_clean_for_training` for cleaner training/evaluation subsets.",
        "- Splits are source-level: all three edits for one source stay in the same split.",
        "- Watermarks are intentionally ignored by the annotation policy; phone UI / screenshot cleanup should be filtered separately if needed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/reframejudge_v1/annotations/reframegen_seedream_strong150_vlm_labels_ignore_watermark.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reframejudge_v1/annotations/reframegen_seedream_strong150.jsonl"),
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/reframejudge_v1/splits"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/reframegen_seedream_strong150_quality_report.md"),
    )
    parser.add_argument("--dataset-version", default="reframejudge_v1_reframegen_seedream_strong150")
    parser.add_argument("--expected-records", type=int, default=150)
    parser.add_argument("--train-sources", type=int, default=35)
    parser.add_argument("--val-sources", type=int, default=5)
    parser.add_argument("--test-sources", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_records = read_jsonl(args.input)
    records = [
        normalize_record(record, index, args.dataset_version)
        for index, record in enumerate(raw_records, 1)
    ]
    validate_records(records, args.expected_records)

    total_sources = len({record["source_manifest_id"] for record in records})
    if args.train_sources + args.val_sources + args.test_sources != total_sources:
        raise SystemExit(
            "train_sources + val_sources + test_sources must equal total source groups, "
            f"got {args.train_sources + args.val_sources + args.test_sources} vs {total_sources}"
        )

    splits = split_by_source(records, args.train_sources, args.val_sources, args.seed)
    write_jsonl(args.output, records)
    write_jsonl(args.split_dir / "reframegen_seedream_strong150_train.jsonl", splits["train"])
    write_jsonl(args.split_dir / "reframegen_seedream_strong150_val.jsonl", splits["val"])
    write_jsonl(args.split_dir / "reframegen_seedream_strong150_test.jsonl", splits["test"])
    write_report(args.report, records, splits, args.input, args.output)

    print(
        json.dumps(
            {
                "input_records": len(raw_records),
                "output": str(args.output),
                "splits": {name: len(split_records) for name, split_records in splits.items()},
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
