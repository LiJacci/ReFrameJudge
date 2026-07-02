#!/usr/bin/env python3
"""Split ReFrameJudge JSONL annotations by source image group."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PHOTO_ID_RE = re.compile(r"photo_id=([^,\s]+)")


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


def group_key(record):
    notes = record.get("notes", "")
    match = PHOTO_ID_RE.search(notes)
    if match:
        return f"photo_id:{match.group(1)}"
    source_image = record.get("source_image", "")
    return f"source:{Path(source_image).stem}"


def label_counts(records):
    return Counter(record["overall_label"] for record in records)


def split_groups(groups, train_ratio, val_ratio, seed):
    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    total = sum(len(groups[key]) for key in keys)
    train_target = int(round(total * train_ratio))
    val_target = int(round(total * val_ratio))

    splits = {"train": [], "val": [], "test": []}
    for key in keys:
        if len(splits["train"]) < train_target:
            split = "train"
        elif len(splits["val"]) < val_target:
            split = "val"
        else:
            split = "test"
        splits[split].extend(groups[key])

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0:
        raise SystemExit("Ratios must be non-negative and train_ratio must be positive")
    if args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio + val_ratio must be less than 1")

    records = read_jsonl(args.input)
    groups = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)

    splits = split_groups(groups, args.train_ratio, args.val_ratio, args.seed)
    output_names = {
        "train": f"{args.prefix}train.jsonl",
        "val": f"{args.prefix}val.jsonl",
        "test": f"{args.prefix}test.jsonl",
    }

    for split, split_records in splits.items():
        write_jsonl(args.output_dir / output_names[split], split_records)

    summary = {
        "input_records": len(records),
        "groups": len(groups),
        "splits": {
            split: {
                "records": len(split_records),
                "labels": dict(label_counts(split_records)),
            }
            for split, split_records in splits.items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
