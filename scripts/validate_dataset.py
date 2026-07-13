#!/usr/bin/env python3
"""Validate ReFrameJudge annotation JSONL files."""

import argparse
import json
from pathlib import Path


VALID_LABELS = {"win", "tie", "lose"}
VALID_EDIT_TYPES = {
    "zoom_in",
    "shift",
    "crop",
    "outpainting",
    "view_change",
    "subject_reposition",
    "generated_recomposition",
    "aesthetic_reconstruction",
    "unknown",
}
VALID_TAGS = {
    "subject_cropped",
    "subject_deformed",
    "identity_changed",
    "important_content_missing",
    "background_changed_too_much",
    "new_irrelevant_objects",
    "composition_not_improved",
    "composition_worse",
    "bad_empty_space",
    "over_cropping",
    "unnatural_perspective",
    "lighting_inconsistent",
    "texture_artifacts",
    "low_resolution_or_blur",
    "better_subject_prominence",
    "better_balance",
    "better_crop",
    "cleaner_background",
    "better_rule_of_thirds",
    "better_symmetry",
    "better_leading_lines",
    "better_visual_focus",
    "more_natural_framing",
}
REQUIRED_FIELDS = {
    "id",
    "source_image",
    "edited_image",
    "edit_type",
    "data_source",
    "overall_label",
    "improvement_score",
    "composition_gain",
    "content_preservation",
    "visual_naturalness",
    "issue_tags",
    "notes",
}


def check_score(name, value, minimum, maximum):
    if not isinstance(value, (int, float)):
        return f"{name} must be a number"
    if value < minimum or value > maximum:
        return f"{name} must be in [{minimum}, {maximum}]"
    return None


def validate_record(record, line_no, root, check_images):
    errors = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        errors.append(f"line {line_no}: missing fields: {sorted(missing)}")
        return errors

    if record["overall_label"] not in VALID_LABELS:
        errors.append(f"line {line_no}: invalid overall_label: {record['overall_label']}")

    if record["edit_type"] not in VALID_EDIT_TYPES:
        errors.append(f"line {line_no}: invalid edit_type: {record['edit_type']}")

    for field, minimum, maximum in [
        ("improvement_score", -2, 2),
        ("composition_gain", 1, 5),
        ("content_preservation", 1, 5),
        ("visual_naturalness", 1, 5),
    ]:
        error = check_score(field, record[field], minimum, maximum)
        if error:
            errors.append(f"line {line_no}: {error}")

    if not isinstance(record["issue_tags"], list):
        errors.append(f"line {line_no}: issue_tags must be a list")
    else:
        invalid_tags = sorted(set(record["issue_tags"]) - VALID_TAGS)
        if invalid_tags:
            errors.append(f"line {line_no}: invalid issue_tags: {invalid_tags}")

    if check_images:
        for field in ["source_image", "edited_image"]:
            image_path = root / record[field]
            if not image_path.exists():
                errors.append(f"line {line_no}: image not found: {record[field]}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_file", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also verify source_image and edited_image paths exist.",
    )
    args = parser.parse_args()

    if not args.annotation_file.exists():
        raise SystemExit(f"Annotation file not found: {args.annotation_file}")

    errors = []
    ids = set()
    count = 0

    with args.annotation_file.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue

            sample_id = record.get("id")
            if sample_id in ids:
                errors.append(f"line {line_no}: duplicate id: {sample_id}")
            ids.add(sample_id)
            errors.extend(validate_record(record, line_no, args.root, args.check_images))

    if errors:
        print(f"Validation failed: {len(errors)} error(s) in {count} record(s).")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Validation passed: {count} record(s).")


if __name__ == "__main__":
    main()
