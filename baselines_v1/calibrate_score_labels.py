#!/usr/bin/env python3
"""Calibrate ReFrameJudge labels from predicted improvement scores.

VLMs often produce better scalar judgement scores than discrete labels. This
script learns two thresholds on a validation prediction file:

    score <= low_threshold      -> lose
    low_threshold < score < high_threshold -> tie
    score >= high_threshold     -> win

Then it applies the thresholds to test predictions and reports metrics.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABEL_TO_ID = {"lose": 0, "tie": 1, "win": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_IDS = [LABEL_TO_ID["lose"], LABEL_TO_ID["tie"], LABEL_TO_ID["win"]]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_of(record, score_key):
    if "pred_scores" in record and score_key in record["pred_scores"]:
        return float(record["pred_scores"][score_key])
    return float(record[score_key])


def label_from_score(score, low_threshold, high_threshold):
    if score <= low_threshold:
        return "lose"
    if score >= high_threshold:
        return "win"
    return "tie"


def apply_thresholds(records, low_threshold, high_threshold, score_key):
    calibrated = []
    for record in records:
        item = dict(record)
        score = score_of(record, score_key)
        item["original_pred_label"] = record.get("pred_label")
        item["pred_label"] = label_from_score(score, low_threshold, high_threshold)
        item["calibration_score"] = score
        item["calibration_low_threshold"] = low_threshold
        item["calibration_high_threshold"] = high_threshold
        item["correct"] = item.get("true_label") == item["pred_label"]
        calibrated.append(item)
    return calibrated


def label_metrics(records):
    y_true = [LABEL_TO_ID[item["true_label"]] for item in records]
    y_pred = [LABEL_TO_ID[item["pred_label"]] for item in records]
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


def subset_metrics(records):
    by_subset = defaultdict(list)
    for record in records:
        by_subset[record.get("subset") or record.get("data_source") or "unknown"].append(record)
    return {
        subset: {
            "count": len(items),
            "overall_label": label_metrics(items),
            "prediction_distribution": dict(Counter(item["pred_label"] for item in items)),
            "target_distribution": dict(Counter(item["true_label"] for item in items)),
        }
        for subset, items in sorted(by_subset.items())
    }


def candidate_thresholds(records, score_key, grid_size):
    scores = np.array([score_of(record, score_key) for record in records], dtype="float64")
    unique_scores = sorted(set(scores.tolist()))
    if len(unique_scores) <= grid_size:
        values = unique_scores
    else:
        values = np.linspace(float(scores.min()), float(scores.max()), grid_size).tolist()
    values = sorted(set(values))
    for low in values:
        for high in values:
            if low < high:
                yield float(low), float(high)


def find_best_thresholds(records, score_key, grid_size, objective):
    best = None
    for low, high in candidate_thresholds(records, score_key, grid_size):
        calibrated = apply_thresholds(records, low, high, score_key)
        metrics = label_metrics(calibrated)
        score = metrics["macro_f1"] if objective == "macro_f1" else metrics["accuracy"]
        tie_count = sum(1 for item in calibrated if item["pred_label"] == "tie")
        tie_penalty = abs(tie_count - sum(1 for item in records if item["true_label"] == "tie")) / max(1, len(records))
        ranking_key = (score, -tie_penalty, metrics["accuracy"])
        if best is None or ranking_key > best["ranking_key"]:
            best = {
                "low_threshold": low,
                "high_threshold": high,
                "metrics": metrics,
                "ranking_key": ranking_key,
            }
    return best


def write_jsonl(path, records):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-jsonl", type=Path)
    parser.add_argument("--score-key", default="improvement_score")
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--objective", choices=["macro_f1", "accuracy"], default="macro_f1")
    args = parser.parse_args()

    val_records = read_jsonl(args.val_predictions)
    test_records = read_jsonl(args.test_predictions)
    best = find_best_thresholds(val_records, args.score_key, args.grid_size, args.objective)
    test_calibrated = apply_thresholds(
        test_records,
        best["low_threshold"],
        best["high_threshold"],
        args.score_key,
    )
    result = {
        "method": "score_threshold_calibration",
        "score_key": args.score_key,
        "objective": args.objective,
        "val_predictions": str(args.val_predictions),
        "test_predictions": str(args.test_predictions),
        "low_threshold": best["low_threshold"],
        "high_threshold": best["high_threshold"],
        "val": best["metrics"],
        "test": {
            "overall_label": label_metrics(test_calibrated),
            "by_subset": subset_metrics(test_calibrated),
            "prediction_distribution": dict(Counter(item["pred_label"] for item in test_calibrated)),
            "target_distribution": dict(Counter(item["true_label"] for item in test_calibrated)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_jsonl(args.output_predictions_jsonl, test_calibrated)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
