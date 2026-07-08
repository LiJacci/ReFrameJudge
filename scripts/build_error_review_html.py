#!/usr/bin/env python3
"""Build an HTML page for inspecting baseline prediction errors."""

import argparse
import html
import json
import random
from pathlib import Path


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def confusion_bucket(record):
    true_label = record["true_label"]
    pred_label = record["pred_label"]
    if true_label == "win" and pred_label == "win":
        return "TP"
    if true_label == "lose" and pred_label == "lose":
        return "TN"
    if true_label == "lose" and pred_label == "win":
        return "FP"
    if true_label == "win" and pred_label == "lose":
        return "FN"
    if true_label == pred_label:
        return f"CORRECT_{true_label.upper()}"
    return "ERROR"


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


def select_records(records, buckets, sample_size, seed):
    filtered = [record for record in records if confusion_bucket(record) in buckets]
    rng = random.Random(seed)
    filtered.sort(key=lambda record: (confusion_bucket(record), record["id"]))
    if sample_size is not None and sample_size < len(filtered):
        filtered = rng.sample(filtered, sample_size)
        filtered.sort(key=lambda record: (confusion_bucket(record), record["id"]))
    return filtered


def css_class(bucket):
    return {
        "TP": "tp",
        "TN": "tn",
        "FP": "fp",
        "FN": "fn",
        "ERROR": "err",
    }.get(bucket, "")


def probability_text(record):
    if "predicted_score" in record:
        return (
            f"target: {record.get('target_score', 0.0):.3f}, "
            f"pred: {record['predicted_score']:.3f}, "
            f"thr: [{record.get('low_threshold', 0.0):.3f}, "
            f"{record.get('high_threshold', 0.0):.3f}]"
        )
    if "preference_score" in record:
        return (
            f"score: {record['preference_score']:.4f}, "
            f"tau: {record.get('tie_threshold', 0.0):.4f}"
        )
    if "prob_win" in record:
        return f"prob_win: {record['prob_win']:.4f}"
    if "class_probabilities" in record:
        parts = [
            f"{label}: {probability:.4f}"
            for label, probability in sorted(record["class_probabilities"].items())
        ]
        return "probs: " + ", ".join(parts)
    if "confidence" in record:
        return f"confidence: {record['confidence']:.4f}"
    return "confidence: n/a"


def build_html(records, project_root, output_path, title):
    counts = {}
    for record in records:
        bucket = confusion_bucket(record)
        counts[bucket] = counts.get(bucket, 0) + 1

    sections = []
    for index, record in enumerate(records, 1):
        bucket = confusion_bucket(record)
        source = rel_image_path(record["source_image"], project_root, output_path)
        edited = rel_image_path(record["edited_image"], project_root, output_path)
        tags = ", ".join(record.get("issue_tags", []))
        reason = record.get("reason", "")
        sections.append(
            f"""
      <section class="sample {css_class(bucket)}">
        <div class="meta">
          <div><strong>{index}. {html.escape(record['id'])}</strong></div>
          <div class="bucket">{bucket}</div>
          <div>true: <strong>{html.escape(record['true_label'])}</strong></div>
          <div>pred: <strong>{html.escape(record['pred_label'])}</strong></div>
          <div>{html.escape(probability_text(record))}</div>
        </div>
        <div class="images">
          <figure>
            <img src="{html.escape(source)}" alt="source image">
            <figcaption>source</figcaption>
          </figure>
          <figure>
            <img src="{html.escape(edited)}" alt="edited image">
            <figcaption>edited</figcaption>
          </figure>
        </div>
        <div class="details">
          <div><strong>tags:</strong> {html.escape(tags)}</div>
          <div><strong>reason:</strong> {html.escape(reason)}</div>
          <div><strong>notes:</strong> {html.escape(record.get('notes', ''))}</div>
        </div>
      </section>
"""
        )

    summary = " | ".join(f"{key}: {counts[key]}" for key in sorted(counts))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d9dde5;
      --text: #1f2937;
      --muted: #667085;
      --tp: #d8f3dc;
      --tn: #dbeafe;
      --fp: #fee2e2;
      --fn: #fef3c7;
      --err: #ffe4cc;
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
      z-index: 2;
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.96);
    }}
    h1 {{ margin: 0; font-size: 20px; }}
    .sub {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
    .sample {{
      margin-bottom: 18px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .sample.tp .meta {{ background: var(--tp); }}
    .sample.tn .meta {{ background: var(--tn); }}
    .sample.fp .meta {{ background: var(--fp); }}
    .sample.fn .meta {{ background: var(--fn); }}
    .sample.err .meta {{ background: var(--err); }}
    .meta {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 90px 120px 120px minmax(220px, 1fr);
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}
    .bucket {{ font-weight: 800; }}
    .images {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: var(--border);
    }}
    figure {{ margin: 0; padding: 12px; background: #fff; }}
    img {{
      display: block;
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: #f1f3f5;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }}
    .details {{
      padding: 12px 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    @media (max-width: 760px) {{
      .meta {{ grid-template-columns: 1fr 64px; }}
      .images {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="sub">{len(records)} sampled predictions. {html.escape(summary)}</div>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--buckets", nargs="+", default=["FP", "FN"])
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="ReFrameJudge Error Review")
    args = parser.parse_args()

    records = read_jsonl(args.predictions)
    selected = select_records(records, set(args.buckets), args.sample_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_html(selected, args.project_root, args.output, args.title),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(records), "selected": len(selected), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
