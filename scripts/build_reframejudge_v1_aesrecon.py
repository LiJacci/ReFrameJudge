#!/usr/bin/env python3
"""Freeze AesRecon GPT composition annotations as a ReFrameJudge-v1 subset."""

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


def canonical_issue_tags(record):
    text = tag_text(record)
    tags = set()

    if any(token in text for token in ["subject_prominence", "subject_more_prominent", "larger_subject"]):
        tags.add("better_subject_prominence")
    if any(token in text for token in ["balance", "balanced", "visual_balance"]):
        tags.add("better_balance")
    if any(token in text for token in ["crop", "framing", "tighter_framing", "cleaner_framing"]):
        tags.add("better_crop")
        tags.add("more_natural_framing")
    if any(token in text for token in ["cleaner_background", "background_clean", "reduced_distractions", "simplified_background"]):
        tags.add("cleaner_background")
    if any(token in text for token in ["rule_of_thirds", "thirds"]):
        tags.add("better_rule_of_thirds")
    if any(token in text for token in ["symmetry", "symmetrical"]):
        tags.add("better_symmetry")
    if any(token in text for token in ["leading_line", "leading_lines", "guide_the_eye"]):
        tags.add("better_leading_lines")
    if any(token in text for token in ["visual_focus", "clearer_focus", "subject_isolation", "focal"]):
        tags.add("better_visual_focus")

    if record.get("composition_score") == 0 or record.get("composition_relevance") == "low":
        tags.add("composition_not_improved")
    if any(token in text for token in ["worse", "weaken", "distracting_overlay", "text_overlay", "awkward_crop"]):
        tags.add("composition_worse")
    if any(token in text for token in ["empty_space", "large_empty", "awkward_empty", "excess_empty"]):
        tags.add("bad_empty_space")
    if any(token in text for token in ["tight_crop", "over_cropping", "edge_crop", "cut_off", "cropped_tight"]):
        tags.add("over_cropping")

    if not tags:
        tags.add("better_crop" if record.get("composition_score", 0) > 0 else "composition_not_improved")

    return [tag for tag in CANONICAL_TAGS if tag in tags]


def normalize_record(record, index, dataset_version):
    composition_score = int(record["composition_score"])
    return {
        "id": f"rfj_v1_aesrecon_{index:06d}",
        "source_image": record["source_image"],
        "edited_image": record["edited_image"],
        "edit_type": "aesthetic_reconstruction",
        "data_source": "AesRecon",
        "pair_type": "real_photo_aesthetic_pair",
        "dataset_version": dataset_version,
        "label_source": record["label_source"],
        "overall_label": record["overall_label"],
        "expected_label": record.get("expected_label", record["overall_label"]),
        "composition_score": composition_score,
        "improvement_score": int(record["improvement_score"]),
        "composition_gain": int(record["composition_gain"]),
        "composition_relevance": record["composition_relevance"],
        "label_confidence": record["label_confidence"],
        "content_preservation": int(record["content_preservation"]),
        "visual_naturalness": int(record["visual_naturalness"]),
        "issue_tags": canonical_issue_tags(record),
        "positive_tags": record.get("positive_tags", []),
        "negative_tags": record.get("negative_tags", []),
        "reason": record.get("reason", ""),
        "notes": record.get("notes", ""),
        "source_record_id": record["id"],
        "good_image_name": record.get("good_image_name", ""),
        "poor_image_name": record.get("poor_image_name", ""),
        "lower_quality_image": record.get("lower_quality_image", record["source_image"]),
        "higher_quality_image": record.get("higher_quality_image", record["edited_image"]),
    }


def validate_records(records):
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate normalized ids found")
    for record in records:
        if record["overall_label"] != "win":
            raise ValueError(f"Unexpected overall_label for {record['id']}: {record['overall_label']}")
        if record["composition_score"] != record["improvement_score"]:
            raise ValueError(f"Score mismatch for {record['id']}")
        if record["composition_gain"] not in {3, 4, 5}:
            raise ValueError(f"Unexpected composition_gain for {record['id']}: {record['composition_gain']}")
        if record["composition_relevance"] not in {"high", "medium", "low"}:
            raise ValueError(f"Unexpected composition_relevance for {record['id']}")
        if record["label_confidence"] not in {"high", "medium", "low"}:
            raise ValueError(f"Unexpected label_confidence for {record['id']}")
        if not record["reason"].strip():
            raise ValueError(f"Empty reason for {record['id']}")


def split_records(records, train_count, val_count, seed):
    groups = defaultdict(list)
    for record in records:
        key = (record["composition_relevance"], record["composition_score"], record["composition_gain"])
        groups[key].append(record)

    rng = random.Random(seed)
    for group_records in groups.values():
        rng.shuffle(group_records)

    def allocate_counts(source_groups, target_count):
        total = sum(len(group_records) for group_records in source_groups.values())
        if target_count > total:
            raise ValueError(f"Cannot allocate {target_count} records from {total} records")
        allocations = {}
        remainders = []
        allocated = 0
        for key, group_records in source_groups.items():
            exact = len(group_records) * target_count / total
            count = int(exact)
            allocations[key] = count
            allocated += count
            remainders.append((exact - count, key))
        for _, key in sorted(remainders, reverse=True)[: target_count - allocated]:
            allocations[key] += 1
        return allocations

    train_allocations = allocate_counts(groups, train_count)
    train = []
    remaining_groups = {}
    for key, group_records in groups.items():
        count = train_allocations[key]
        train.extend(group_records[:count])
        remaining_groups[key] = group_records[count:]

    val_allocations = allocate_counts(remaining_groups, val_count)
    val = []
    test = []
    for key, group_records in remaining_groups.items():
        count = val_allocations[key]
        val.extend(group_records[:count])
        test.extend(group_records[count:])

    for split_records_ in [train, val, test]:
        split_records_.sort(key=lambda record: record["id"])
    return train, val, test


def counter(records, field):
    return Counter(record[field] for record in records)


def top_tags(records, field, limit=20):
    counts = Counter()
    for record in records:
        counts.update(record.get(field, []))
    return counts.most_common(limit)


def split_summary(records):
    return {
        "records": len(records),
        "composition_relevance": dict(counter(records, "composition_relevance")),
        "label_confidence": dict(counter(records, "label_confidence")),
        "composition_score": dict(counter(records, "composition_score")),
        "composition_gain": dict(counter(records, "composition_gain")),
    }


def markdown_table(counts):
    lines = ["| Value | Count |", "| --- | ---: |"]
    for value, count in counts:
        lines.append(f"| {value} | {count} |")
    return "\n".join(lines)


def write_report(path, records, train, val, test, source_path, output_path):
    high_medium = [
        record
        for record in records
        if record["composition_relevance"] in {"high", "medium"}
        and record["label_confidence"] in {"high", "medium"}
    ]
    strong = [
        record
        for record in records
        if record["composition_relevance"] == "high"
        and record["label_confidence"] == "high"
        and record["composition_score"] == 2
    ]

    lines = [
        "# AesRecon-500 Quality Report",
        "",
        "## Overview",
        "",
        f"- Source file: `{source_path}`",
        f"- Frozen annotation file: `{output_path}`",
        f"- Total records: {len(records)}",
        f"- Main usable records (`high|medium` relevance and confidence): {len(high_medium)}",
        f"- Strong composition records (`high/high`, score 2): {len(strong)}",
        f"- Low composition relevance records: {counter(records, 'composition_relevance').get('low', 0)}",
        "",
        "## Label Distribution",
        "",
        markdown_table(counter(records, "overall_label").most_common()),
        "",
        "## Composition Relevance",
        "",
        markdown_table(counter(records, "composition_relevance").most_common()),
        "",
        "## Label Confidence",
        "",
        markdown_table(counter(records, "label_confidence").most_common()),
        "",
        "## Composition Score",
        "",
        markdown_table(counter(records, "composition_score").most_common()),
        "",
        "## Composition Gain",
        "",
        markdown_table(counter(records, "composition_gain").most_common()),
        "",
        "## Split Summary",
        "",
        "```json",
        json.dumps(
            {
                "train": split_summary(train),
                "val": split_summary(val),
                "test": split_summary(test),
            },
            indent=2,
        ),
        "```",
        "",
        "## Top Canonical Issue Tags",
        "",
        markdown_table(top_tags(records, "issue_tags")),
        "",
        "## Top GPT Positive Tags",
        "",
        markdown_table(top_tags(records, "positive_tags")),
        "",
        "## Top GPT Negative Tags",
        "",
        markdown_table(top_tags(records, "negative_tags")),
        "",
        "## Recommended Use",
        "",
        "- Use all 500 records as the AesRecon real-photo composition preference subset.",
        "- Treat `overall_label=win` as dataset-direction supervision from AesRecon.",
        "- Treat `composition_relevance`, `label_confidence`, `composition_score`, `composition_gain`, tags, and `reason` as GPT weak composition annotations.",
        "- For cleaner composition-specific training, prefer records where relevance and confidence are both `high` or `medium`.",
        "- Keep low-relevance records as useful hard/weak positives: they prevent the evaluator from assuming every preferred image is a strong composition improvement.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/aesrecon_gpt_composition_labels_500.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reframejudge_v1/annotations/aesrecon_500.jsonl"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/reframejudge_v1/splits"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/aesrecon_500_quality_report.md"),
    )
    parser.add_argument("--dataset-version", default="reframejudge_v1_aesrecon_500")
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_records = read_jsonl(args.input)
    expected_records = args.train_count + args.val_count + args.test_count
    if len(raw_records) != expected_records:
        raise SystemExit(
            "Expected train_count + val_count + test_count records, "
            f"got {len(raw_records)} records."
        )

    records = [
        normalize_record(record, index, args.dataset_version)
        for index, record in enumerate(raw_records, 1)
    ]
    validate_records(records)

    train, val, test = split_records(records, args.train_count, args.val_count, args.seed)
    if len(train) + len(val) + len(test) != len(records):
        raise SystemExit("Split record count mismatch")

    write_jsonl(args.output, records)
    write_jsonl(args.split_dir / "aesrecon_train.jsonl", train)
    write_jsonl(args.split_dir / "aesrecon_val.jsonl", val)
    write_jsonl(args.split_dir / "aesrecon_test.jsonl", test)
    write_report(args.report, records, train, val, test, args.input, args.output)

    print(
        json.dumps(
            {
                "input_records": len(raw_records),
                "output": str(args.output),
                "splits": {
                    "train": len(train),
                    "val": len(val),
                    "test": len(test),
                },
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
