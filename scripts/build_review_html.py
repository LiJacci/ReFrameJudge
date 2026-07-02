#!/usr/bin/env python3
"""Build a lightweight HTML review page for ReFrameJudge pairs."""

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


def rel_image_path(image_path, project_root, output_path):
    absolute_image = (project_root / image_path).resolve()
    output_dir = output_path.resolve().parent
    return Path(relative_path(absolute_image, output_dir)).as_posix()


def relative_path(target, start):
    try:
        return target.relative_to(start)
    except ValueError:
        return Path(__import__("os").path.relpath(target, start))


def sample_records(records, sample_size, seed):
    if sample_size is None or sample_size >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, sample_size)


def label_class(label):
    return {
        "win": "label-win",
        "tie": "label-tie",
        "lose": "label-lose",
    }.get(label, "")


def build_html(records, project_root, output_path, title):
    rows = []
    for index, record in enumerate(records, 1):
        source = rel_image_path(record["source_image"], project_root, output_path)
        edited = rel_image_path(record["edited_image"], project_root, output_path)
        label = html.escape(record["overall_label"])
        score = html.escape(str(record["improvement_score"]))
        tags = ", ".join(record.get("issue_tags", []))
        notes = record.get("notes", "")
        rows.append(
            f"""
      <section class="sample">
        <div class="meta">
          <div><strong>{index}. {html.escape(record['id'])}</strong></div>
          <div class="{label_class(record['overall_label'])}">{label}</div>
          <div>score: {score}</div>
          <div>type: {html.escape(record.get('edit_type', ''))}</div>
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
          <div><strong>notes:</strong> {html.escape(notes)}</div>
        </div>
      </section>
"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #d9dde5;
      --text: #1f2937;
      --muted: #667085;
      --win: #0f7b4f;
      --tie: #8a5b00;
      --lose: #b42318;
      --bg: #f6f7f9;
      --panel: #ffffff;
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
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }}
    .sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px;
    }}
    .sample {{
      margin-bottom: 18px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .meta {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 80px 100px 100px;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}
    .label-win {{ color: var(--win); font-weight: 700; }}
    .label-tie {{ color: var(--tie); font-weight: 700; }}
    .label-lose {{ color: var(--lose); font-weight: 700; }}
    .images {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: var(--border);
    }}
    figure {{
      margin: 0;
      padding: 12px;
      background: #fff;
    }}
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
      .meta {{ grid-template-columns: 1fr 70px; }}
      .images {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="sub">{len(records)} sampled pairs. Judge whether edited improves composition over source.</div>
  </header>
  <main>
    {''.join(rows)}
  </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="ReFrameJudge Pair Review")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    sampled = sample_records(records, args.sample_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_html(sampled, args.project_root, args.output, args.title),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(records), "sampled": len(sampled), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
