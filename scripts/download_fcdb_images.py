#!/usr/bin/env python3
"""Download FCDB images listed in annotation JSON files."""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_records(annotation_path):
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def filename_from_record(record):
    url_name = record["url"].rstrip("/").split("/")[-1]
    if url_name:
        return url_name
    return f"{record['flickr_photo_id']}.jpg"


def download_one(record, output_dir, timeout, retries):
    url = record["url"]
    output_path = output_dir / filename_from_record(record)
    if output_path.exists() and output_path.stat().st_size > 0:
        return "exists", url, str(output_path)

    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ReFrameJudge/0.1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty response")
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(output_path)
            return "downloaded", url, str(output_path)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1 + attempt)

    return "failed", url, str(last_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    if not args.annotation.exists():
        raise SystemExit(f"Annotation file not found: {args.annotation}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.annotation)
    if args.max_images is not None:
        records = records[: args.max_images]

    summary = {"downloaded": 0, "exists": 0, "failed": 0}
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                record,
                args.output_dir,
                args.timeout,
                args.retries,
            )
            for record in records
        ]
        for index, future in enumerate(as_completed(futures), 1):
            status, url, detail = future.result()
            summary[status] += 1
            if status == "failed":
                failures.append({"url": url, "error": detail})
            if index % 100 == 0 or index == len(futures):
                print(
                    f"{index}/{len(futures)} "
                    f"downloaded={summary['downloaded']} "
                    f"exists={summary['exists']} "
                    f"failed={summary['failed']}"
                )

    if failures:
        failure_path = args.output_dir.parent / "download_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Wrote failures to {failure_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
