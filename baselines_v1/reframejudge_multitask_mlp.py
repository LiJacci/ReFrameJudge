#!/usr/bin/env python3
"""Frozen CLIP encoder + multi-task MLP baseline for ReFrameJudge-v1.

The model predicts the overall 3-way preference label and numeric judgement
scores from a pair of source/edited image embeddings.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


LABEL_TO_ID = {"lose": 0, "tie": 1, "win": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_IDS = [LABEL_TO_ID["lose"], LABEL_TO_ID["tie"], LABEL_TO_ID["win"]]
REGRESSION_TARGETS = [
    "improvement_score",
    "composition_gain",
    "content_preservation",
    "visual_naturalness",
]


def read_jsonl(path, max_samples=None):
    records = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("overall_label") in LABEL_TO_ID:
                records.append(record)
            else:
                skipped += 1
            if max_samples is not None and len(records) >= max_samples:
                break
    if skipped:
        print(f"Skipped {skipped} records with labels outside {sorted(LABEL_TO_ID)}")
    return records


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(project_root, image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def filter_missing_images(records, project_root, policy):
    if policy == "error":
        missing = []
        for record in records:
            for key in ["source_image", "edited_image"]:
                if not resolve_image_path(project_root, record[key]).exists():
                    missing.append(record[key])
        if missing:
            preview = "\n".join(sorted(set(missing))[:10])
            raise FileNotFoundError(f"Missing {len(set(missing))} image files. First missing paths:\n{preview}")
        return records

    kept = []
    skipped = 0
    for record in records:
        if all(resolve_image_path(project_root, record[key]).exists() for key in ["source_image", "edited_image"]):
            kept.append(record)
        else:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} records with missing image files")
    return kept


def unique_image_paths(*record_lists):
    paths = {}
    for records in record_lists:
        for record in records:
            paths[record["source_image"]] = None
            paths[record["edited_image"]] = None
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
def extract_clip_embeddings(paths, project_root, model, processor, device, batch_size, embedding_map):
    missing = [path for path in paths if path not in embedding_map]
    if not missing:
        return embedding_map

    model.eval()
    for start in tqdm(range(0, len(missing), batch_size), desc="Extract CLIP embeddings"):
        batch_paths = missing[start : start + batch_size]
        images = [open_image(resolve_image_path(project_root, image_path)) for image_path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        features = features.detach().cpu().numpy().astype("float32")
        for image_path, feature in zip(batch_paths, features):
            embedding_map[image_path] = feature
    return embedding_map


def pair_features(records, embedding_map):
    features = []
    label_ids = []
    regression_values = []
    regression_masks = []
    for record in records:
        src = embedding_map[record["source_image"]]
        edit = embedding_map[record["edited_image"]]
        feature = np.concatenate([src, edit, edit - src, src * edit], axis=0)
        features.append(feature)
        label_ids.append(LABEL_TO_ID[record["overall_label"]])

        values = []
        masks = []
        for target in REGRESSION_TARGETS:
            value = record.get(target)
            if value is None:
                values.append(0.0)
                masks.append(0.0)
            else:
                values.append(float(value))
                masks.append(1.0)
        regression_values.append(values)
        regression_masks.append(masks)

    x = np.stack(features).astype("float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "x": x,
        "y_label": np.array(label_ids, dtype="int64"),
        "y_reg": np.array(regression_values, dtype="float32"),
        "reg_mask": np.array(regression_masks, dtype="float32"),
    }


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True).astype("float32")
        self.std = x.std(axis=0, keepdims=True).astype("float32")
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype("float32")


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.label_head = nn.Linear(hidden_dim, len(LABEL_TO_ID))
        self.regression_head = nn.Linear(hidden_dim, len(REGRESSION_TARGETS))

    def forward(self, x):
        hidden = self.backbone(x)
        return {
            "label_logits": self.label_head(hidden),
            "regression": self.regression_head(hidden),
        }


def make_loader(arrays, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(arrays["x"]),
        torch.from_numpy(arrays["y_label"]),
        torch.from_numpy(arrays["y_reg"]),
        torch.from_numpy(arrays["reg_mask"]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_weights(labels, device):
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    weights = []
    for label_id in LABEL_IDS:
        count = max(counts.get(label_id, 0), 1)
        weights.append(total / (len(LABEL_IDS) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(model, train_arrays, val_arrays, args, device):
    train_loader = make_loader(train_arrays, args.train_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    label_loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_arrays["y_label"], device))

    best_state = None
    best_score = -math.inf
    best_epoch = 0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y_label, y_reg, reg_mask in train_loader:
            x = x.to(device)
            y_label = y_label.to(device)
            y_reg = y_reg.to(device)
            reg_mask = reg_mask.to(device)

            outputs = model(x)
            label_loss = label_loss_fn(outputs["label_logits"], y_label)
            reg_error = (outputs["regression"] - y_reg) ** 2
            reg_loss = (reg_error * reg_mask).sum() / reg_mask.sum().clamp_min(1.0)
            loss = label_loss + args.regression_loss_weight * reg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(x)

        val_metrics, _ = evaluate_model(model, val_arrays, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(total_loss / len(train_arrays["x"])),
            "val_accuracy": val_metrics["overall_label"]["accuracy"],
            "val_macro_f1": val_metrics["overall_label"]["macro_f1"],
            "val_weighted_f1": val_metrics["overall_label"]["weighted_f1"],
        }
        history.append(epoch_summary)
        score = val_metrics["overall_label"]["macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        print(
            f"epoch={epoch:03d} train_loss={epoch_summary['train_loss']:.4f} "
            f"val_macro_f1={score:.4f} val_acc={val_metrics['overall_label']['accuracy']:.4f}"
        )
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_score),
        "history": history,
    }


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def safe_corr(x, y, spearman=False):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if len(x) < 2:
        return None
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def predict_arrays(model, arrays, device, batch_size=256):
    loader = make_loader(arrays, batch_size, shuffle=False)
    logits = []
    regressions = []
    model.eval()
    for x, _, _, _ in loader:
        outputs = model(x.to(device))
        logits.append(outputs["label_logits"].detach().cpu())
        regressions.append(outputs["regression"].detach().cpu())
    logits = torch.cat(logits, dim=0)
    probabilities = torch.softmax(logits, dim=-1).numpy()
    pred_label_ids = probabilities.argmax(axis=1)
    reg_pred = torch.cat(regressions, dim=0).numpy()
    return pred_label_ids, probabilities, reg_pred


def label_metrics(y_true, y_pred):
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
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        average=None,
        zero_division=0,
    )
    per_class = {}
    for label_id, precision, recall, f1, count in zip(LABEL_IDS, per_precision, per_recall, per_f1, support):
        per_class[ID_TO_LABEL[label_id]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(count),
        }
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
        "per_class": per_class,
    }


def regression_metrics(y_true, y_pred, mask):
    metrics = {}
    for index, target in enumerate(REGRESSION_TARGETS):
        target_mask = mask[:, index] > 0
        if not target_mask.any():
            metrics[target] = {"count": 0, "mae": None, "pearson": None, "spearman": None}
            continue
        true_values = y_true[target_mask, index]
        pred_values = y_pred[target_mask, index]
        metrics[target] = {
            "count": int(target_mask.sum()),
            "mae": float(mean_absolute_error(true_values, pred_values)),
            "pearson": safe_corr(true_values, pred_values, spearman=False),
            "spearman": safe_corr(true_values, pred_values, spearman=True),
        }
    return metrics


def evaluate_model(model, arrays, device):
    y_pred, probabilities, reg_pred = predict_arrays(model, arrays, device)
    metrics = {
        "overall_label": label_metrics(arrays["y_label"], y_pred),
        "regression": regression_metrics(arrays["y_reg"], reg_pred, arrays["reg_mask"]),
    }
    predictions = {
        "label_ids": y_pred,
        "probabilities": probabilities,
        "regression": reg_pred,
    }
    return metrics, predictions


def subset_metrics(records, arrays, predictions):
    by_subset = defaultdict(list)
    for index, record in enumerate(records):
        by_subset[record.get("subset", record.get("data_source", "unknown"))].append(index)

    results = {}
    for subset, indices in sorted(by_subset.items()):
        idx = np.array(indices, dtype="int64")
        results[subset] = {
            "count": int(len(indices)),
            "overall_label": label_metrics(arrays["y_label"][idx], predictions["label_ids"][idx]),
            "regression": regression_metrics(
                arrays["y_reg"][idx],
                predictions["regression"][idx],
                arrays["reg_mask"][idx],
            ),
        }
    return results


def class_distribution(records):
    return dict(Counter(record["overall_label"] for record in records))


def score_to_0_100(improvement_score):
    return float(np.clip((improvement_score + 2.0) / 4.0 * 100.0, 0.0, 100.0))


def write_predictions(path, records, arrays, predictions, split):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            pred_id = int(predictions["label_ids"][index])
            true_id = int(arrays["y_label"][index])
            probs = {
                ID_TO_LABEL[label_id]: float(predictions["probabilities"][index, label_id])
                for label_id in LABEL_IDS
            }
            pred_scores = {
                target: float(predictions["regression"][index, target_index])
                for target_index, target in enumerate(REGRESSION_TARGETS)
            }
            pred_judgement = {
                "overall_score": score_to_0_100(pred_scores["improvement_score"]),
                "overall_label": ID_TO_LABEL[pred_id],
                "composition_score": pred_scores["composition_gain"],
                "content_preservation": pred_scores["content_preservation"],
                "visual_naturalness": pred_scores["visual_naturalness"],
                "issue_tags": [],
                "reason": "",
            }
            true_scores = {
                target: record.get(target)
                for target in REGRESSION_TARGETS
            }
            output = {
                "id": record["id"],
                "split": split,
                "source_image": record["source_image"],
                "edited_image": record["edited_image"],
                "true_label": ID_TO_LABEL[true_id],
                "pred_label": ID_TO_LABEL[pred_id],
                "label_probabilities": probs,
                "correct": bool(true_id == pred_id),
                "pred_judgement": pred_judgement,
                "true_scores": true_scores,
                "pred_scores": pred_scores,
                "subset": record.get("subset"),
                "data_source": record.get("data_source"),
                "pair_type": record.get("pair_type"),
                "edit_type": record.get("edit_type"),
                "reference_issue_tags": record.get("issue_tags", []),
                "reference_reason": record.get("reason", ""),
            }
            handle.write(json.dumps(output, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/clip_embeddings_reframejudge_v1.npz"))
    parser.add_argument(
        "--model-name",
        default="data/cache/clip-vit-base-patch32",
        help="CLIP model id or local model directory.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("data/cache/huggingface"))
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--regression-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--missing-image-policy", choices=["error", "skip"], default="error")
    args = parser.parse_args()

    seed_everything(args.random_state)
    project_root = args.project_root.resolve()

    train_records = filter_missing_images(read_jsonl(args.train_jsonl, args.max_train), project_root, args.missing_image_policy)
    val_records = filter_missing_images(read_jsonl(args.val_jsonl, args.max_val), project_root, args.missing_image_policy)
    test_records = filter_missing_images(read_jsonl(args.test_jsonl, args.max_test), project_root, args.missing_image_policy)
    if not train_records or not val_records or not test_records:
        raise ValueError("Train/val/test records must be non-empty after filtering.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading CLIP encoder: {args.model_name}")
    processor = CLIPProcessor.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    encoder = CLIPModel.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir).to(device)

    paths = unique_image_paths(train_records, val_records, test_records)
    embedding_map = load_cache(args.cache)
    embedding_map = extract_clip_embeddings(
        paths,
        project_root,
        encoder,
        processor,
        device,
        args.extract_batch_size,
        embedding_map,
    )
    save_cache(args.cache, embedding_map)
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    train_arrays = pair_features(train_records, embedding_map)
    val_arrays = pair_features(val_records, embedding_map)
    test_arrays = pair_features(test_records, embedding_map)

    standardizer = Standardizer().fit(train_arrays["x"])
    for arrays in [train_arrays, val_arrays, test_arrays]:
        arrays["x"] = standardizer.transform(arrays["x"])

    model = MultiTaskMLP(
        input_dim=train_arrays["x"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    train_summary = train_model(model, train_arrays, val_arrays, args, device)

    val_metrics, val_predictions = evaluate_model(model, val_arrays, device)
    test_metrics, test_predictions = evaluate_model(model, test_arrays, device)

    result = {
        "model": "clip_multitask_mlp",
        "vision_model": str(args.model_name),
        "device": str(device),
        "feature": "[source, edited, edited-source, source*edited]",
        "feature_dim": int(train_arrays["x"].shape[1]),
        "records": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "class_distribution": {
            "train": class_distribution(train_records),
            "val": class_distribution(val_records),
            "test": class_distribution(test_records),
        },
        "label_mapping": LABEL_TO_ID,
        "regression_targets": REGRESSION_TARGETS,
        "training": {
            "epochs": args.epochs,
            "best_epoch": train_summary["best_epoch"],
            "best_val_macro_f1": train_summary["best_val_macro_f1"],
            "history": train_summary["history"],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "regression_loss_weight": args.regression_loss_weight,
        },
        "val": val_metrics,
        "test": test_metrics,
        "test_by_subset": subset_metrics(test_records, test_arrays, test_predictions),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.predictions_jsonl is not None:
        if args.predictions_jsonl.exists():
            args.predictions_jsonl.unlink()
        write_predictions(args.predictions_jsonl, val_records, val_arrays, val_predictions, "val")
        write_predictions(args.predictions_jsonl, test_records, test_arrays, test_predictions, "test")

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "standardizer_mean": standardizer.mean,
                "standardizer_std": standardizer.std,
                "label_mapping": LABEL_TO_ID,
                "regression_targets": REGRESSION_TARGETS,
                "vision_model": str(args.model_name),
                "feature_dim": int(train_arrays["x"].shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "args": vars(args),
            },
            args.checkpoint,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
