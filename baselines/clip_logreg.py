#!/usr/bin/env python3
"""CLIP embedding + Logistic Regression baseline for ReFrameJudge.

This baseline freezes a CLIP image encoder, extracts embeddings for source and
edited images, builds pairwise features, and trains a binary logistic regression
classifier for win/lose prediction.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


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
                path = record[key]
                paths[path] = None
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


def pair_features(records, embedding_map):
    features = []
    labels = []
    for record in records:
        src = embedding_map[record["source_image"]]
        edit = embedding_map[record["edited_image"]]
        feature = np.concatenate([src, edit, edit - src, src * edit], axis=0)
        features.append(feature)
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
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    return metrics


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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--logreg-c", type=float, default=0.1)
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

    x_train, y_train = pair_features(train_records, embedding_map)
    x_val, y_val = pair_features(val_records, embedding_map)
    x_test, y_test = pair_features(test_records, embedding_map)

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
        "model": "clip_logreg",
        "clip_model": args.model_name,
        "device": str(device),
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
    del clip_model
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
