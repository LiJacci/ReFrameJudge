#!/usr/bin/env python3
"""Feature-fusion Logistic Regression baseline for ReFrameJudge."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABEL_TO_ID = {"lose": 0, "win": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def read_jsonl(path, max_samples=None):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["overall_label"] in LABEL_TO_ID:
                records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def load_cache(cache_path):
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    paths = data["paths"].tolist()
    embeddings = data["embeddings"]
    return {path: embeddings[index] for index, path in enumerate(paths)}


def pair_feature_for_encoder(record, embedding_map, encoder_name):
    missing = [
        path
        for path in [record["source_image"], record["edited_image"]]
        if path not in embedding_map
    ]
    if missing:
        raise KeyError(f"{encoder_name} cache missing embeddings for {missing}")

    src = embedding_map[record["source_image"]]
    edit = embedding_map[record["edited_image"]]
    return np.concatenate([src, edit, edit - src, src * edit], axis=0)


def pair_features(records, cache_maps):
    features = []
    labels = []
    for record in records:
        parts = [
            pair_feature_for_encoder(record, embedding_map, encoder_name)
            for encoder_name, embedding_map in cache_maps
        ]
        features.append(np.concatenate(parts, axis=0))
        labels.append(LABEL_TO_ID[record["overall_label"]])

    x = np.stack(features).astype("float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, np.array(labels)


def evaluate(model, x, y):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pred = model.predict(x)
        probabilities = model.predict_proba(x)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def predict_records(model, records, x, y, split):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pred = model.predict(x)
        probabilities = model.predict_proba(x)[:, 1]

    predictions = []
    for record, true_id, pred_id, prob_win in zip(records, y, pred, probabilities):
        predictions.append(
            {
                "id": record["id"],
                "split": split,
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "true_label": ID_TO_LABEL[int(true_id)],
                "pred_label": ID_TO_LABEL[int(pred_id)],
                "prob_win": float(prob_win),
                "correct": bool(int(true_id) == int(pred_id)),
                "edit_type": record.get("edit_type", ""),
                "improvement_score": record.get("improvement_score"),
                "issue_tags": record.get("issue_tags", []),
                "notes": record.get("notes", ""),
            }
        )
    return predictions


def write_jsonl(path, records):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--vision-cache", type=Path, required=True)
    parser.add_argument("--vision-name", default="dinov2-base")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--logreg-c", type=float, default=0.1)
    args = parser.parse_args()

    train_records = read_jsonl(args.train_jsonl, args.max_train)
    val_records = read_jsonl(args.val_jsonl, args.max_val)
    test_records = read_jsonl(args.test_jsonl, args.max_test)

    cache_maps = [
        ("clip", load_cache(args.clip_cache)),
        (args.vision_name, load_cache(args.vision_cache)),
    ]

    x_train, y_train = pair_features(train_records, cache_maps)
    x_val, y_val = pair_features(val_records, cache_maps)
    x_test, y_test = pair_features(test_records, cache_maps)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.logreg_c,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight="balanced",
            solver="liblinear",
        ),
    )
    classifier.fit(x_train, y_train)

    result = {
        "model": "fusion_logreg",
        "encoders": ["clip", args.vision_name],
        "feature_dim": int(x_train.shape[1]),
        "records": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "logreg": {
            "C": args.logreg_c,
            "solver": "liblinear",
        },
        "label_mapping": LABEL_TO_ID,
        "val": evaluate(classifier, x_val, y_val),
        "test": evaluate(classifier, x_test, y_test),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.predictions_jsonl is not None:
        predictions = []
        predictions.extend(predict_records(classifier, val_records, x_val, y_val, "val"))
        predictions.extend(predict_records(classifier, test_records, x_test, y_test, "test"))
        write_jsonl(args.predictions_jsonl, predictions)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
