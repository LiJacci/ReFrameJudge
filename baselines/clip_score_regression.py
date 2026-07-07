#!/usr/bin/env python3
"""CLIP pairwise score regression baseline for ReFrameJudge.

The model predicts a continuous vote-margin score:

    target = (edited_votes - source_votes) / (edited_votes + source_votes)

Then two thresholds are calibrated on validation data to derive lose/tie/win.
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


LABEL_TO_ID = {"lose": 0, "tie": 1, "win": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABELS = [LABEL_TO_ID["lose"], LABEL_TO_ID["tie"], LABEL_TO_ID["win"]]


def read_jsonl(path, max_samples=None):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def unique_image_paths(*record_lists):
    paths = {}
    for records in record_lists:
        for record in records:
            for key in ["source_image", "edited_image"]:
                paths[record[key]] = None
    return sorted(paths)


def load_cache(cache_path):
    if cache_path is None or not cache_path.exists():
        return {}
    data = np.load(cache_path, allow_pickle=True)
    paths = data["paths"].tolist()
    embeddings = data["embeddings"]
    return {path: embeddings[index] for index, path in enumerate(paths)}


def save_cache(cache_path, embedding_map):
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(embedding_map)
    embeddings = np.stack([embedding_map[path] for path in paths])
    np.savez_compressed(cache_path, paths=np.array(paths, dtype=object), embeddings=embeddings)


def open_image(path):
    return Image.open(path).convert("RGB")


@torch.no_grad()
def extract_embeddings(paths, project_root, model, processor, device, batch_size, embedding_map):
    missing = [path for path in paths if path not in embedding_map]
    if not missing:
        return embedding_map

    model.eval()
    for start in tqdm(range(0, len(missing), batch_size), desc="Extract CLIP embeddings"):
        batch_paths = missing[start : start + batch_size]
        images = [open_image(project_root / path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        features = features.detach().cpu().numpy().astype("float32")
        for path, feature in zip(batch_paths, features):
            embedding_map[path] = feature
    return embedding_map


def target_score(record):
    if "vote_margin" in record:
        margin = float(record["vote_margin"])
    elif "source_votes" in record and "edited_votes" in record:
        margin = float(record["edited_votes"] - record["source_votes"])
    else:
        return float(record.get("improvement_score", 0.0)) / 2.0

    total_votes = float(record.get("source_votes", 0) + record.get("edited_votes", 0))
    if total_votes <= 0:
        total_votes = 5.0
    return margin / total_votes


def label_id(record):
    return LABEL_TO_ID[record["overall_label"]]


def pair_features_and_targets(records, embedding_map):
    features = []
    scores = []
    labels = []
    for record in records:
        src = embedding_map[record["source_image"]]
        edit = embedding_map[record["edited_image"]]
        feature = np.concatenate([src, edit, edit - src, src * edit], axis=0)
        features.append(feature)
        scores.append(target_score(record))
        labels.append(label_id(record))
    x = np.stack(features).astype("float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, np.array(scores, dtype="float32"), np.array(labels, dtype=np.int64)


def safe_correlation(fn, y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        value = fn(y_true, y_pred).statistic
    if math.isnan(value):
        return None
    return float(value)


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "pearson": safe_correlation(pearsonr, y_true, y_pred),
        "spearman": safe_correlation(spearmanr, y_true, y_pred),
        "target_summary": score_summary(y_true),
        "prediction_summary": score_summary(y_pred),
    }


def score_summary(scores):
    return {
        "min": float(np.min(scores)),
        "p05": float(np.quantile(scores, 0.05)),
        "p25": float(np.quantile(scores, 0.25)),
        "median": float(np.quantile(scores, 0.5)),
        "p75": float(np.quantile(scores, 0.75)),
        "p95": float(np.quantile(scores, 0.95)),
        "max": float(np.max(scores)),
    }


def predict_labels(scores, low_threshold, high_threshold):
    pred = np.full(scores.shape, LABEL_TO_ID["tie"], dtype=np.int64)
    pred[scores < low_threshold] = LABEL_TO_ID["lose"]
    pred[scores > high_threshold] = LABEL_TO_ID["win"]
    return pred


def threshold_candidates(scores, max_candidates):
    values = np.unique(scores)
    if len(values) == 0:
        return np.array([0.0])
    candidates = np.concatenate([[float(np.min(values)) - 1e-6], values, [float(np.max(values)) + 1e-6]])
    if len(candidates) > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        candidates = np.unique(np.quantile(candidates, quantiles))
    return candidates


def classification_metric(y_true, y_pred, metric):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "macro_f1":
        return precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )[2]
    raise ValueError(f"Unsupported metric: {metric}")


def calibrate_thresholds(scores, labels, metric, max_candidates):
    candidates = threshold_candidates(scores, max_candidates)
    best = {
        "low_threshold": 0.0,
        "high_threshold": 0.0,
        "metric": -1.0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
    }
    for low in candidates:
        for high in candidates:
            if low > high:
                continue
            pred = predict_labels(scores, float(low), float(high))
            accuracy = accuracy_score(labels, pred)
            macro_f1 = precision_recall_fscore_support(
                labels,
                pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )[2]
            value = macro_f1 if metric == "macro_f1" else accuracy
            if value > best["metric"]:
                best = {
                    "low_threshold": float(low),
                    "high_threshold": float(high),
                    "metric": float(value),
                    "accuracy": float(accuracy),
                    "macro_f1": float(macro_f1),
                }
    return best


def classification_metrics(scores, labels, low_threshold, high_threshold):
    pred = predict_labels(scores, low_threshold, high_threshold)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        labels,
        pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_labels": [ID_TO_LABEL[label_id] for label_id in LABELS],
        "confusion_matrix": confusion_matrix(labels, pred, labels=LABELS).tolist(),
        "per_class": {},
    }
    for label_id, precision, recall, f1, count in zip(
        LABELS,
        per_class_precision,
        per_class_recall,
        per_class_f1,
        support,
    ):
        metrics["per_class"][ID_TO_LABEL[label_id]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(count),
        }
    return metrics


def class_distribution(records):
    counts = {label: 0 for label in LABEL_TO_ID}
    for record in records:
        counts[record["overall_label"]] += 1
    return counts


def score_distribution(records):
    values = [target_score(record) for record in records]
    return {
        str(score): values.count(score)
        for score in sorted(set(values))
    }


def predict_records(records, true_scores, labels, pred_scores, low_threshold, high_threshold, split):
    pred_labels = predict_labels(pred_scores, low_threshold, high_threshold)
    predictions = []
    for record, true_score, true_label, pred_score, pred_label in zip(
        records,
        true_scores,
        labels,
        pred_scores,
        pred_labels,
    ):
        predictions.append(
            {
                "id": record["id"],
                "split": split,
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "true_label": ID_TO_LABEL[int(true_label)],
                "pred_label": ID_TO_LABEL[int(pred_label)],
                "target_score": float(true_score),
                "predicted_score": float(pred_score),
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
                "score_error": float(pred_score - true_score),
                "correct": bool(int(true_label) == int(pred_label)),
                "edit_type": record.get("edit_type", ""),
                "improvement_score": record.get("improvement_score"),
                "issue_tags": record.get("issue_tags", []),
                "preference_strength": record.get("preference_strength"),
                "vote_margin": record.get("vote_margin"),
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
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/clip_embeddings_fcdb_5k.npz"))
    parser.add_argument(
        "--model-name",
        default="data/cache/clip-vit-base-patch32",
        help="Hugging Face model id or local directory containing CLIP weights/config/tokenizer files.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("data/cache/huggingface"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--threshold-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--max-threshold-candidates", type=int, default=300)
    args = parser.parse_args()

    train_records = read_jsonl(args.train_jsonl, args.max_train)
    val_records = read_jsonl(args.val_jsonl, args.max_val)
    test_records = read_jsonl(args.test_jsonl, args.max_test)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading CLIP: {args.model_name}")
    processor = CLIPProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
    )
    clip_model = CLIPModel.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
    ).to(device)

    paths = unique_image_paths(train_records, val_records, test_records)
    embedding_map = load_cache(args.cache)
    embedding_map = extract_embeddings(
        paths,
        args.project_root,
        clip_model,
        processor,
        device,
        args.batch_size,
        embedding_map,
    )
    save_cache(args.cache, embedding_map)

    x_train, y_train_score, y_train_label = pair_features_and_targets(train_records, embedding_map)
    x_val, y_val_score, y_val_label = pair_features_and_targets(val_records, embedding_map)
    x_test, y_test_score, y_test_label = pair_features_and_targets(test_records, embedding_map)

    regressor = make_pipeline(
        StandardScaler(),
        Ridge(alpha=args.alpha),
    )
    regressor.fit(x_train, y_train_score)

    train_pred_score = regressor.predict(x_train)
    val_pred_score = regressor.predict(x_val)
    test_pred_score = regressor.predict(x_test)
    thresholds = calibrate_thresholds(
        val_pred_score,
        y_val_label,
        args.threshold_metric,
        args.max_threshold_candidates,
    )

    result = {
        "model": "clip_score_regression",
        "clip_model": args.model_name,
        "device": str(device),
        "feature_dim": int(x_train.shape[1]),
        "records": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "label_mapping": LABEL_TO_ID,
        "target": "vote_margin / total_votes",
        "class_distribution": {
            "train": class_distribution(train_records),
            "val": class_distribution(val_records),
            "test": class_distribution(test_records),
        },
        "score_distribution": {
            "train": score_distribution(train_records),
            "val": score_distribution(val_records),
            "test": score_distribution(test_records),
        },
        "regressor": {
            "name": "Ridge",
            "alpha": args.alpha,
        },
        "threshold_calibration": {
            "metric": args.threshold_metric,
            **thresholds,
        },
        "train_regression": regression_metrics(y_train_score, train_pred_score),
        "val_regression": regression_metrics(y_val_score, val_pred_score),
        "test_regression": regression_metrics(y_test_score, test_pred_score),
        "val": classification_metrics(
            val_pred_score,
            y_val_label,
            thresholds["low_threshold"],
            thresholds["high_threshold"],
        ),
        "test": classification_metrics(
            test_pred_score,
            y_test_label,
            thresholds["low_threshold"],
            thresholds["high_threshold"],
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.predictions_jsonl is not None:
        predictions = []
        predictions.extend(
            predict_records(
                val_records,
                y_val_score,
                y_val_label,
                val_pred_score,
                thresholds["low_threshold"],
                thresholds["high_threshold"],
                "val",
            )
        )
        predictions.extend(
            predict_records(
                test_records,
                y_test_score,
                y_test_label,
                test_pred_score,
                thresholds["low_threshold"],
                thresholds["high_threshold"],
                "test",
            )
        )
        write_jsonl(args.predictions_jsonl, predictions)

    print(json.dumps(result, indent=2))
    del clip_model
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
