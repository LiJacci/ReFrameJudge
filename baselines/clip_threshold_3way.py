#!/usr/bin/env python3
"""CLIP pairwise preference score with a calibrated tie threshold.

The model learns a directional win/lose preference from strong FCDB labels, then
calibrates a symmetric threshold on 3-way validation data:

    score > tau      -> win
    abs(score) <= tau -> tie
    score < -tau     -> lose
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


BINARY_LABEL_TO_ID = {"lose": 0, "win": 1}
THREEWAY_LABEL_TO_ID = {"lose": 0, "tie": 1, "win": 2}
ID_TO_THREEWAY_LABEL = {value: key for key, value in THREEWAY_LABEL_TO_ID.items()}
THREEWAY_LABELS = [
    THREEWAY_LABEL_TO_ID["lose"],
    THREEWAY_LABEL_TO_ID["tie"],
    THREEWAY_LABEL_TO_ID["win"],
]


def read_jsonl(path, allowed_labels, max_samples=None):
    records = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["overall_label"] in allowed_labels:
                records.append(record)
            else:
                skipped += 1
            if max_samples is not None and len(records) >= max_samples:
                break
    if skipped:
        print(f"Skipped {skipped} records with labels outside {sorted(allowed_labels)}")
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


def pair_features(records, embedding_map, label_map):
    features = []
    labels = []
    for record in records:
        src = embedding_map[record["source_image"]]
        edit = embedding_map[record["edited_image"]]
        feature = np.concatenate([src, edit, edit - src, src * edit], axis=0)
        features.append(feature)
        labels.append(label_map[record["overall_label"]])
    x = np.stack(features).astype("float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, np.array(labels)


def preference_scores(model, x):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if hasattr(model, "decision_function"):
            return model.decision_function(x)
        probabilities = model.predict_proba(x)[:, 1]
    return probabilities - 0.5


def predict_from_scores(scores, threshold):
    predictions = np.full(scores.shape, THREEWAY_LABEL_TO_ID["tie"], dtype=np.int64)
    predictions[scores > threshold] = THREEWAY_LABEL_TO_ID["win"]
    predictions[scores < -threshold] = THREEWAY_LABEL_TO_ID["lose"]
    return predictions


def metric_value(y_true, y_pred, metric):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "macro_f1":
        return precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=THREEWAY_LABELS,
            average="macro",
            zero_division=0,
        )[2]
    raise ValueError(f"Unsupported metric: {metric}")


def candidate_thresholds(scores, max_candidates):
    abs_scores = np.unique(np.abs(scores))
    if len(abs_scores) == 0:
        return np.array([0.0])
    candidates = np.concatenate([[0.0], abs_scores])
    if len(candidates) > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        candidates = np.quantile(candidates, quantiles)
        candidates = np.unique(candidates)
    return candidates


def calibrate_threshold(scores, labels, metric, max_candidates):
    best = {
        "threshold": 0.0,
        "metric": -1.0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
    }
    for threshold in candidate_thresholds(scores, max_candidates):
        pred = predict_from_scores(scores, float(threshold))
        accuracy = accuracy_score(labels, pred)
        macro_f1 = precision_recall_fscore_support(
            labels,
            pred,
            labels=THREEWAY_LABELS,
            average="macro",
            zero_division=0,
        )[2]
        score = macro_f1 if metric == "macro_f1" else accuracy
        if score > best["metric"]:
            best = {
                "threshold": float(threshold),
                "metric": float(score),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
            }
    return best


def class_distribution(records):
    counts = {label: 0 for label in THREEWAY_LABEL_TO_ID}
    for record in records:
        counts[record["overall_label"]] += 1
    return counts


def evaluate_scores(scores, labels, threshold):
    pred = predict_from_scores(scores, threshold)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        labels=THREEWAY_LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        labels=THREEWAY_LABELS,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        labels,
        pred,
        labels=THREEWAY_LABELS,
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
        "confusion_matrix_labels": [ID_TO_THREEWAY_LABEL[label_id] for label_id in THREEWAY_LABELS],
        "confusion_matrix": confusion_matrix(labels, pred, labels=THREEWAY_LABELS).tolist(),
        "score_summary": {
            "min": float(np.min(scores)),
            "p05": float(np.quantile(scores, 0.05)),
            "p25": float(np.quantile(scores, 0.25)),
            "median": float(np.quantile(scores, 0.5)),
            "p75": float(np.quantile(scores, 0.75)),
            "p95": float(np.quantile(scores, 0.95)),
            "max": float(np.max(scores)),
        },
        "per_class": {},
    }
    for label_id, precision, recall, f1, count in zip(
        THREEWAY_LABELS,
        per_class_precision,
        per_class_recall,
        per_class_f1,
        support,
    ):
        metrics["per_class"][ID_TO_THREEWAY_LABEL[label_id]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(count),
        }
    return metrics


def predict_records(records, labels, scores, threshold, split):
    pred = predict_from_scores(scores, threshold)
    predictions = []
    for record, true_id, pred_id, score in zip(records, labels, pred, scores):
        predictions.append(
            {
                "id": record["id"],
                "split": split,
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "true_label": ID_TO_THREEWAY_LABEL[int(true_id)],
                "pred_label": ID_TO_THREEWAY_LABEL[int(pred_id)],
                "preference_score": float(score),
                "tie_threshold": float(threshold),
                "confidence": float(abs(score)),
                "correct": bool(int(true_id) == int(pred_id)),
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
    parser.add_argument("--strong-train-jsonl", type=Path, required=True)
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
    parser.add_argument("--max-strong-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--logreg-c", type=float, default=0.1)
    parser.add_argument("--threshold-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--max-threshold-candidates", type=int, default=1000)
    args = parser.parse_args()

    strong_train_records = read_jsonl(
        args.strong_train_jsonl,
        BINARY_LABEL_TO_ID,
        args.max_strong_train,
    )
    val_records = read_jsonl(args.val_jsonl, THREEWAY_LABEL_TO_ID, args.max_val)
    test_records = read_jsonl(args.test_jsonl, THREEWAY_LABEL_TO_ID, args.max_test)

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

    paths = unique_image_paths(strong_train_records, val_records, test_records)
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

    x_train, y_train = pair_features(strong_train_records, embedding_map, BINARY_LABEL_TO_ID)
    x_val, y_val = pair_features(val_records, embedding_map, THREEWAY_LABEL_TO_ID)
    x_test, y_test = pair_features(test_records, embedding_map, THREEWAY_LABEL_TO_ID)

    scorer = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.logreg_c,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight="balanced",
            solver="liblinear",
        ),
    )
    scorer.fit(x_train, y_train)

    val_scores = preference_scores(scorer, x_val)
    test_scores = preference_scores(scorer, x_test)
    threshold = calibrate_threshold(
        val_scores,
        y_val,
        args.threshold_metric,
        args.max_threshold_candidates,
    )

    result = {
        "model": "clip_threshold_3way",
        "clip_model": args.model_name,
        "device": str(device),
        "feature_dim": int(x_train.shape[1]),
        "records": {
            "strong_train": len(strong_train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "label_mapping": THREEWAY_LABEL_TO_ID,
        "class_distribution": {
            "strong_train": class_distribution(strong_train_records),
            "val": class_distribution(val_records),
            "test": class_distribution(test_records),
        },
        "preference_model": {
            "binary_label_mapping": BINARY_LABEL_TO_ID,
            "C": args.logreg_c,
            "solver": "liblinear",
        },
        "threshold_calibration": {
            "metric": args.threshold_metric,
            **threshold,
        },
        "val": evaluate_scores(val_scores, y_val, threshold["threshold"]),
        "test": evaluate_scores(test_scores, y_test, threshold["threshold"]),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.predictions_jsonl is not None:
        predictions = []
        predictions.extend(
            predict_records(val_records, y_val, val_scores, threshold["threshold"], "val")
        )
        predictions.extend(
            predict_records(test_records, y_test, test_scores, threshold["threshold"], "test")
        )
        write_jsonl(args.predictions_jsonl, predictions)

    print(json.dumps(result, indent=2))
    del clip_model
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
