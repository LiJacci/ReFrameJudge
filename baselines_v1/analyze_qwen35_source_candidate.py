#!/usr/bin/env python3
"""Analyze Qwen3.5 source/candidate ReFrameJudge predictions."""

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LABEL_TO_ID = {"lose": 0, "tie": 1, "win": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_IDS = [LABEL_TO_ID["lose"], LABEL_TO_ID["tie"], LABEL_TO_ID["win"]]
REGRESSION_TARGETS = [
    "improvement_score",
    "composition_gain",
    "content_preservation",
    "visual_naturalness",
]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


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


def label_metrics(records):
    y_true = [LABEL_TO_ID[item["true_label"]] for item in records]
    y_pred = [LABEL_TO_ID[item["pred_label"]] for item in records]
    matrix = [[0 for _ in LABEL_IDS] for _ in LABEL_IDS]
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[true_id][pred_id] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for label_id in LABEL_IDS:
        tp = matrix[label_id][label_id]
        fp = sum(matrix[row][label_id] for row in LABEL_IDS if row != label_id)
        fn = sum(matrix[label_id][col] for col in LABEL_IDS if col != label_id)
        support = sum(matrix[label_id])
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        per_class[ID_TO_LABEL[label_id]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    total = max(1, len(records))
    accuracy = sum(1 for true_id, pred_id in zip(y_true, y_pred) if true_id == pred_id) / total
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1s) / len(f1s)
    weighted_precision = sum(value * support for value, support in zip(precisions, supports)) / total
    weighted_recall = sum(value * support for value, support in zip(recalls, supports)) / total
    weighted_f1 = sum(value * support for value, support in zip(f1s, supports)) / total
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_labels": [ID_TO_LABEL[label_id] for label_id in LABEL_IDS],
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def score_value(record, source, target):
    container = record.get(source, {})
    if not isinstance(container, dict):
        return None
    return safe_float(container.get(target))


def histogram(values, bins):
    values = [value for value in values if value is not None]
    if not values:
        return []
    counts, edges = np.histogram(np.asarray(values, dtype="float64"), bins=bins)
    return [
        {
            "low": float(edges[index]),
            "high": float(edges[index + 1]),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def regression_metrics(records):
    metrics = {}
    for target in REGRESSION_TARGETS:
        true_values = []
        pred_values = []
        for record in records:
            true_value = score_value(record, "true_scores", target)
            pred_value = score_value(record, "pred_scores", target)
            if true_value is None or pred_value is None:
                continue
            true_values.append(true_value)
            pred_values.append(pred_value)
        if not true_values:
            metrics[target] = {"count": 0, "mae": None, "pearson": None, "spearman": None}
            continue
        metrics[target] = {
            "count": len(true_values),
            "mae": float(np.mean(np.abs(np.asarray(pred_values) - np.asarray(true_values)))),
            "pearson": safe_corr(true_values, pred_values, spearman=False),
            "spearman": safe_corr(true_values, pred_values, spearman=True),
            "true_mean": float(np.mean(true_values)),
            "pred_mean": float(np.mean(pred_values)),
            "true_std": float(np.std(true_values)),
            "pred_std": float(np.std(pred_values)),
            "true_distribution": dict(Counter(true_values)),
            "pred_distribution": dict(Counter(pred_values)),
            "true_histogram": histogram(true_values, bins=8),
            "pred_histogram": histogram(pred_values, bins=8),
        }
    return metrics


def subset_metrics(records):
    by_subset = defaultdict(list)
    for record in records:
        by_subset[record.get("subset") or record.get("data_source") or "unknown"].append(record)
    return {
        subset: {
            "count": len(items),
            "overall_label": label_metrics(items),
            "regression": regression_metrics(items),
            "prediction_distribution": dict(Counter(item["pred_label"] for item in items)),
            "target_distribution": dict(Counter(item["true_label"] for item in items)),
        }
        for subset, items in sorted(by_subset.items())
    }


def score_error(record, target):
    true_value = score_value(record, "true_scores", target)
    pred_value = score_value(record, "pred_scores", target)
    if true_value is None or pred_value is None:
        return None
    return abs(pred_value - true_value)


def wrongness_score(record):
    label_gap = abs(LABEL_TO_ID[record["pred_label"]] - LABEL_TO_ID[record["true_label"]])
    improvement_error = score_error(record, "improvement_score") or 0.0
    composition_error = score_error(record, "composition_gain") or 0.0
    return label_gap * 10.0 + improvement_error * 2.0 + composition_error


def select_cases(records, max_cases):
    wrong = [record for record in records if record["pred_label"] != record["true_label"]]
    correct = [record for record in records if record["pred_label"] == record["true_label"]]
    tie_errors = [
        record
        for record in wrong
        if record["true_label"] == "tie" or record["pred_label"] == "tie"
    ]
    severe = sorted(wrong, key=wrongness_score, reverse=True)
    score_outliers = sorted(
        wrong,
        key=lambda record: (
            score_error(record, "improvement_score") or 0.0,
            score_error(record, "composition_gain") or 0.0,
        ),
        reverse=True,
    )
    representative = []
    by_pair = defaultdict(list)
    for record in wrong:
        by_pair[(record["true_label"], record["pred_label"])].append(record)
    for key in sorted(by_pair):
        representative.extend(sorted(by_pair[key], key=wrongness_score, reverse=True)[: max(1, max_cases // 12)])
    ordered = []
    seen = set()
    for group in (severe, tie_errors, score_outliers, representative, correct):
        for record in group:
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            ordered.append(record)
            if len(ordered) >= max_cases:
                return ordered
    return ordered


def relative_path(target, start):
    try:
        return target.relative_to(start)
    except ValueError:
        import os

        return Path(os.path.relpath(target, start))


def rel_image_path(image_path, project_root, output_path):
    absolute_image = (project_root / image_path).resolve()
    output_dir = output_path.resolve().parent
    return relative_path(absolute_image, output_dir).as_posix()


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def score_table(record):
    rows = []
    for target in REGRESSION_TARGETS:
        true_value = score_value(record, "true_scores", target)
        pred_value = score_value(record, "pred_scores", target)
        rows.append(
            f"""
            <tr>
              <td>{html.escape(target)}</td>
              <td>{html.escape(fmt(true_value))}</td>
              <td>{html.escape(fmt(pred_value))}</td>
              <td>{html.escape(fmt(None if true_value is None or pred_value is None else pred_value - true_value))}</td>
            </tr>
            """
        )
    return "<table class=\"scores\"><thead><tr><th>score</th><th>true</th><th>pred</th><th>delta</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def confusion_table(metrics):
    labels = metrics["confusion_matrix_labels"]
    matrix = metrics["confusion_matrix"]
    header = "".join(f"<th>pred {html.escape(label)}</th>" for label in labels)
    rows = []
    for label, values in zip(labels, matrix):
        cells = "".join(f"<td>{value}</td>" for value in values)
        rows.append(f"<tr><th>true {html.escape(label)}</th>{cells}</tr>")
    return f"<table class=\"matrix\"><thead><tr><th></th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def metric_cards(summary):
    overall = summary["overall_label"]
    cards = [
        ("records", summary["records"]),
        ("accuracy", fmt(overall["accuracy"])),
        ("macro-F1", fmt(overall["macro_f1"])),
        ("weighted-F1", fmt(overall["weighted_f1"])),
        ("errors", summary["error_count"]),
        ("tie errors", summary["tie_error_count"]),
    ]
    return "".join(
        f"<div class=\"card\"><div class=\"card-label\">{html.escape(label)}</div><div class=\"card-value\">{html.escape(str(value))}</div></div>"
        for label, value in cards
    )


def regression_table(regression):
    rows = []
    for target in REGRESSION_TARGETS:
        item = regression[target]
        rows.append(
            f"""
            <tr>
              <td>{html.escape(target)}</td>
              <td>{html.escape(fmt(item.get('mae')))}</td>
              <td>{html.escape(fmt(item.get('pearson')))}</td>
              <td>{html.escape(fmt(item.get('spearman')))}</td>
              <td>{html.escape(fmt(item.get('true_mean')))}</td>
              <td>{html.escape(fmt(item.get('pred_mean')))}</td>
              <td>{html.escape(fmt(item.get('true_std')))}</td>
              <td>{html.escape(fmt(item.get('pred_std')))}</td>
            </tr>
            """
        )
    return "<table><thead><tr><th>score</th><th>MAE</th><th>Pearson</th><th>Spearman</th><th>true mean</th><th>pred mean</th><th>true std</th><th>pred std</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def subset_table(by_subset):
    rows = []
    for subset, item in by_subset.items():
        overall = item["overall_label"]
        rows.append(
            f"""
            <tr>
              <td>{html.escape(subset)}</td>
              <td>{item['count']}</td>
              <td>{html.escape(fmt(overall['accuracy']))}</td>
              <td>{html.escape(fmt(overall['macro_f1']))}</td>
              <td>{html.escape(json.dumps(item['target_distribution'], ensure_ascii=False))}</td>
              <td>{html.escape(json.dumps(item['prediction_distribution'], ensure_ascii=False))}</td>
            </tr>
            """
        )
    return "<table><thead><tr><th>subset</th><th>n</th><th>accuracy</th><th>macro-F1</th><th>target</th><th>pred</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def case_section(record, index, project_root, output_path):
    source = rel_image_path(record["source_image"], project_root, output_path)
    edited = rel_image_path(record["edited_image"], project_root, output_path)
    is_wrong = record["pred_label"] != record["true_label"]
    tags = ", ".join(record.get("issue_tags", []))
    reason = record.get("reason", "")
    raw_response = record.get("raw_response", "")
    return f"""
    <section class="sample {'wrong' if is_wrong else 'correct'}">
      <div class="sample-head">
        <div>
          <strong>{index}. {html.escape(record['id'])}</strong>
          <span>{html.escape(record.get('subset') or record.get('data_source') or 'unknown')}</span>
        </div>
        <div>true <strong>{html.escape(record['true_label'])}</strong> / pred <strong>{html.escape(record['pred_label'])}</strong></div>
      </div>
      <div class="images">
        <figure>
          <img src="{html.escape(source)}" alt="source image">
          <figcaption>source</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(edited)}" alt="candidate image">
          <figcaption>candidate</figcaption>
        </figure>
      </div>
      <div class="details">
        {score_table(record)}
        <p><strong>tags:</strong> {html.escape(tags)}</p>
        <p><strong>reason:</strong> {html.escape(reason)}</p>
        <details>
          <summary>raw response</summary>
          <pre>{html.escape(raw_response)}</pre>
        </details>
      </div>
    </section>
    """


def build_html(records, selected, summary, project_root, output_path, title):
    sample_html = "".join(
        case_section(record, index, project_root, output_path)
        for index, record in enumerate(selected, 1)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --border: #d8dde6;
      --ok: #e7f8ef;
      --bad: #fff0e8;
      --ink: #172554;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 3;
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.97);
    }}
    h1 {{ margin: 0; font-size: 21px; }}
    h2 {{ margin: 26px 0 10px; font-size: 17px; }}
    .sub {{ margin-top: 5px; color: var(--muted); font-size: 13px; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 20px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }}
    .card-label {{ color: var(--muted); font-size: 12px; }}
    .card-value {{ margin-top: 4px; font-size: 22px; font-weight: 750; color: var(--ink); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef1f5; }}
    .sample {{
      margin: 18px 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 11px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--ok);
      font-size: 14px;
    }}
    .sample.wrong .sample-head {{ background: var(--bad); }}
    .sample-head span {{ margin-left: 8px; color: var(--muted); }}
    .images {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: var(--border);
    }}
    figure {{ margin: 0; padding: 10px; background: #fff; }}
    img {{
      display: block;
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: #f1f3f5;
    }}
    figcaption {{
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }}
    .details {{ padding: 12px 14px; font-size: 13px; color: var(--muted); }}
    .scores {{ margin-bottom: 10px; }}
    pre {{
      white-space: pre-wrap;
      background: #f6f8fb;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      color: var(--text);
    }}
    @media (max-width: 760px) {{
      .images {{ grid-template-columns: 1fr; }}
      .sample-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="sub">{len(records)} predictions, {len(selected)} selected review cases</div>
  </header>
  <main>
    <div class="cards">{metric_cards(summary)}</div>
    <h2>Confusion Matrix</h2>
    {confusion_table(summary["overall_label"])}
    <h2>Regression Scores</h2>
    {regression_table(summary["regression"])}
    <h2>Subset Metrics</h2>
    {subset_table(summary["by_subset"])}
    <h2>Review Cases</h2>
    {sample_html}
  </main>
</body>
</html>
"""


def build_summary(records):
    wrong = [record for record in records if record["pred_label"] != record["true_label"]]
    tie_errors = [
        record
        for record in wrong
        if record["true_label"] == "tie" or record["pred_label"] == "tie"
    ]
    return {
        "records": len(records),
        "unique_ids": len({record["id"] for record in records}),
        "error_count": len(wrong),
        "tie_error_count": len(tie_errors),
        "overall_label": label_metrics(records),
        "regression": regression_metrics(records),
        "by_subset": subset_metrics(records),
        "prediction_distribution": dict(Counter(record["pred_label"] for record in records)),
        "target_distribution": dict(Counter(record["true_label"] for record in records)),
        "confusion_pairs": dict(Counter(f"{record['true_label']}->{record['pred_label']}" for record in records)),
        "top_wrong_cases": [
            {
                "id": record["id"],
                "subset": record.get("subset") or record.get("data_source"),
                "true_label": record["true_label"],
                "pred_label": record["pred_label"],
                "wrongness_score": wrongness_score(record),
                "true_scores": record.get("true_scores", {}),
                "pred_scores": record.get("pred_scores", {}),
                "reason": record.get("reason", ""),
            }
            for record in sorted(wrong, key=wrongness_score, reverse=True)[:20]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--title", default="Qwen3.5 Source-Candidate Error Review")
    args = parser.parse_args()

    records = read_jsonl(args.predictions)
    summary = build_summary(records)
    selected = select_cases(records, args.max_cases)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    args.html_output.write_text(
        build_html(records, selected, summary, args.project_root.resolve(), args.html_output, args.title),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "selected": len(selected),
                "summary_json": str(args.summary_json),
                "html_output": str(args.html_output),
                "accuracy": summary["overall_label"]["accuracy"],
                "macro_f1": summary["overall_label"]["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
