#!/usr/bin/env python3
"""Build ReFrameJudge-v1 combined full and balanced1000 datasets."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SUBSETS = {
    "fcdb": {
        "pair_type": "crop_pair",
        "dataset_version": "reframejudge_v1_fcdb_3way_5k",
        "full": Path("data/pairs/annotations/fcdb_3way_pairs_5k.jsonl"),
        "train": Path("data/pairs/annotations/fcdb_3way_train.jsonl"),
        "val": Path("data/pairs/annotations/fcdb_3way_val.jsonl"),
        "test": Path("data/pairs/annotations/fcdb_3way_test.jsonl"),
    },
    "aesrecon_500": {
        "pair_type": "real_photo_aesthetic_pair",
        "dataset_version": "reframejudge_v1_aesrecon_500",
        "full": Path("data/reframejudge_v1/annotations/aesrecon_500.jsonl"),
        "train": Path("data/reframejudge_v1/splits/aesrecon_train.jsonl"),
        "val": Path("data/reframejudge_v1/splits/aesrecon_val.jsonl"),
        "test": Path("data/reframejudge_v1/splits/aesrecon_test.jsonl"),
    },
    "reframegen_seedream_strong150": {
        "pair_type": "generated_recomposition_pair",
        "dataset_version": "reframejudge_v1_reframegen_seedream_strong150",
        "full": Path("data/reframejudge_v1/annotations/reframegen_seedream_strong150.jsonl"),
        "train": Path("data/reframejudge_v1/splits/reframegen_seedream_strong150_train.jsonl"),
        "val": Path("data/reframejudge_v1/splits/reframegen_seedream_strong150_val.jsonl"),
        "test": Path("data/reframejudge_v1/splits/reframegen_seedream_strong150_test.jsonl"),
    },
}


BALANCED_COUNTS = {
    "train": {"fcdb": 400, "aesrecon_500": 295, "reframegen_seedream_strong150": 105},
    "val": {"fcdb": 50, "aesrecon_500": 35, "reframegen_seedream_strong150": 15},
    "test": {"fcdb": 50, "aesrecon_500": 20, "reframegen_seedream_strong150": 30},
}


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


def add_subset_metadata(record, subset):
    output = dict(record)
    output["subset"] = subset
    output["subset_record_id"] = record["id"]
    output.setdefault("pair_type", SUBSETS[subset]["pair_type"])
    output.setdefault("dataset_version", SUBSETS[subset]["dataset_version"])
    return output


def load_subset(split_or_full):
    records = []
    for subset, config in SUBSETS.items():
        records.extend(add_subset_metadata(record, subset) for record in read_jsonl(config[split_or_full]))
    return records


def sample_stratified(records, count, seed):
    if count > len(records):
        raise ValueError(f"Cannot sample {count} from {len(records)} records")
    if count == len(records):
        return sorted(records, key=lambda record: record["id"])

    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["overall_label"]].append(record)
    for group_records in groups.values():
        rng.shuffle(group_records)

    allocations = {}
    remainders = []
    allocated = 0
    total = len(records)
    for label, group_records in groups.items():
        exact = len(group_records) * count / total
        take = int(exact)
        allocations[label] = take
        allocated += take
        remainders.append((exact - take, label))
    for _, label in sorted(remainders, reverse=True)[: count - allocated]:
        allocations[label] += 1

    sampled = []
    for label, take in allocations.items():
        sampled.extend(groups[label][:take])
    sampled.sort(key=lambda record: record["id"])
    return sampled


def sample_records(records, count, seed):
    if count > len(records):
        raise ValueError(f"Cannot sample {count} from {len(records)} records")
    if count == len(records):
        return sorted(records, key=lambda record: record["id"])
    rng = random.Random(seed)
    sampled = rng.sample(records, count)
    sampled.sort(key=lambda record: record["id"])
    return sampled


def stable_seed(base_seed, *parts):
    value = base_seed
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % 1_000_000_007
    return value


def build_balanced_split(split, seed):
    split_records = []
    for subset, target_count in BALANCED_COUNTS[split].items():
        records = [add_subset_metadata(record, subset) for record in read_jsonl(SUBSETS[subset][split])]
        subset_seed = stable_seed(seed, split, subset)
        if subset == "fcdb":
            sampled = sample_stratified(records, target_count, subset_seed)
        else:
            sampled = sample_records(records, target_count, subset_seed)
        split_records.extend(sampled)
    split_records.sort(key=lambda record: (record["subset"], record["id"]))
    return split_records


def validate_unique(records, name):
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids found in {name}")


def validate_balanced_splits(splits):
    all_ids = set()
    for split, records in splits.items():
        validate_unique(records, split)
        ids = {record["id"] for record in records}
        overlap = all_ids & ids
        if overlap:
            raise ValueError(f"Split leakage in {split}: {sorted(overlap)[:5]}")
        all_ids.update(ids)

    source_to_split = {}
    for split, records in splits.items():
        for record in records:
            if record["subset"] != "reframegen_seedream_strong150":
                continue
            source_id = record["source_manifest_id"]
            existing = source_to_split.get(source_id)
            if existing is not None and existing != split:
                raise ValueError(f"ReFrameGen source leakage: {source_id} in {existing} and {split}")
            source_to_split[source_id] = split


def counter(records, field):
    return Counter(record[field] for record in records)


def nested_counts(records, field):
    by_subset = {}
    for subset in SUBSETS:
        subset_records = [record for record in records if record["subset"] == subset]
        by_subset[subset] = dict(counter(subset_records, field))
    return by_subset


def markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def split_summary(records):
    return {
        "records": len(records),
        "subsets": dict(counter(records, "subset")),
        "labels": dict(counter(records, "overall_label")),
        "labels_by_subset": nested_counts(records, "overall_label"),
    }


def write_report(path, full_records, balanced_records, balanced_splits, args):
    rows = []
    for subset in SUBSETS:
        subset_records = [record for record in full_records if record["subset"] == subset]
        labels = counter(subset_records, "overall_label")
        rows.append((subset, len(subset_records), labels.get("win", 0), labels.get("tie", 0), labels.get("lose", 0)))

    balanced_rows = []
    for subset in SUBSETS:
        subset_records = [record for record in balanced_records if record["subset"] == subset]
        labels = counter(subset_records, "overall_label")
        balanced_rows.append((subset, len(subset_records), labels.get("win", 0), labels.get("tie", 0), labels.get("lose", 0)))

    lines = [
        "# ReFrameJudge-v1 Combined Dataset Report",
        "",
        "## Outputs",
        "",
        f"- Full combined annotation: `{args.full_output}`",
        f"- Balanced1000 annotation: `{args.balanced_output}`",
        f"- Balanced1000 train split: `{args.split_dir / 'reframejudge_v1_combined_balanced1000_train.jsonl'}`",
        f"- Balanced1000 val split: `{args.split_dir / 'reframejudge_v1_combined_balanced1000_val.jsonl'}`",
        f"- Balanced1000 test split: `{args.split_dir / 'reframejudge_v1_combined_balanced1000_test.jsonl'}`",
        "",
        "## Full Combined",
        "",
        f"- Total records: {len(full_records)}",
        "",
        markdown_table(rows, ["Subset", "Records", "Win", "Tie", "Lose"]),
        "",
        "## Balanced1000",
        "",
        f"- Total records: {len(balanced_records)}",
        "- Sampling policy: FCDB is stratified by `overall_label`; AesRecon is randomly sampled from existing splits; ReFrameGen uses all 150 records.",
        "- Split policy: existing per-subset splits are preserved; ReFrameGen remains source-level split.",
        "",
        markdown_table(balanced_rows, ["Subset", "Records", "Win", "Tie", "Lose"]),
        "",
        "## Balanced1000 Split Summary",
        "",
        "```json",
        json.dumps({split: split_summary(records) for split, records in balanced_splits.items()}, indent=2),
        "```",
        "",
        "## Recommended Use",
        "",
        "- Use `reframejudge_v1_combined_full.jsonl` as the complete data pool.",
        "- Use `reframejudge_v1_combined_balanced1000.jsonl` and its splits for main pilot training/evaluation.",
        "- Report metrics both overall and by `subset`, because FCDB crop pairs, AesRecon real-photo pairs, and ReFrameGen generated pairs test different behavior.",
        "- Avoid duplicating ReFrameGen records in the annotation file; use sampler weighting during training if generated-pair emphasis is needed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-output",
        type=Path,
        default=Path("data/reframejudge_v1/annotations/reframejudge_v1_combined_full.jsonl"),
    )
    parser.add_argument(
        "--balanced-output",
        type=Path,
        default=Path("data/reframejudge_v1/annotations/reframejudge_v1_combined_balanced1000.jsonl"),
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/reframejudge_v1/splits"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/reframejudge_v1_combined_balanced1000_report.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    full_records = load_subset("full")
    validate_unique(full_records, "full")

    balanced_splits = {
        split: build_balanced_split(split, args.seed)
        for split in ["train", "val", "test"]
    }
    validate_balanced_splits(balanced_splits)
    balanced_records = balanced_splits["train"] + balanced_splits["val"] + balanced_splits["test"]
    validate_unique(balanced_records, "balanced1000")

    if len(full_records) != 5650:
        raise SystemExit(f"Expected 5650 full records, got {len(full_records)}")
    if len(balanced_records) != 1000:
        raise SystemExit(f"Expected 1000 balanced records, got {len(balanced_records)}")

    write_jsonl(args.full_output, full_records)
    write_jsonl(args.balanced_output, balanced_records)
    write_jsonl(args.split_dir / "reframejudge_v1_combined_balanced1000_train.jsonl", balanced_splits["train"])
    write_jsonl(args.split_dir / "reframejudge_v1_combined_balanced1000_val.jsonl", balanced_splits["val"])
    write_jsonl(args.split_dir / "reframejudge_v1_combined_balanced1000_test.jsonl", balanced_splits["test"])
    write_report(args.report, full_records, balanced_records, balanced_splits, args)

    print(
        json.dumps(
            {
                "full_records": len(full_records),
                "balanced_records": len(balanced_records),
                "balanced_splits": {split: len(records) for split, records in balanced_splits.items()},
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
