#!/usr/bin/env python3
"""Convert FCDB ranking annotations into ReFrameJudge pair annotations.

FCDB ranking annotations compare two crops from the same image. The crop with
more votes is treated as the better composition.
"""

import argparse
import json
import shutil
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def image_filename_from_url(url):
    return url.rstrip("/").split("/")[-1]


def find_image(record, image_dir):
    if image_dir is None:
        return None

    filename = image_filename_from_url(record["url"])
    direct_path = image_dir / filename
    if direct_path.exists():
        return direct_path

    photo_id = str(record.get("flickr_photo_id", ""))
    if photo_id:
        matches = []
        for extension in VALID_IMAGE_EXTENSIONS:
            matches.extend(image_dir.glob(f"*{photo_id}*{extension}"))
        if matches:
            return sorted(matches)[0]

    return None


def crop_box_xywh(crop):
    x, y, width, height = crop
    return int(x), int(y), int(x + width), int(y + height)


def save_crop(source_image, crop, output_path):
    from PIL import Image

    ensure_dir(output_path.parent)
    with Image.open(source_image) as image:
        image.crop(crop_box_xywh(crop)).save(output_path)


def copy_or_crop_image(source_image, crop, output_path, metadata_only):
    if metadata_only:
        return
    if source_image is None:
        raise FileNotFoundError("source image is required unless --metadata-only is set")
    save_crop(source_image, crop, output_path)


def label_from_votes(vote_a, vote_b, tie_margin):
    diff = vote_b - vote_a
    if abs(diff) <= tie_margin:
        return "tie", 0
    if diff > 0:
        return "win", 1
    return "lose", -1


def score_from_votes(vote_a, vote_b, max_abs_score=2):
    total = max(vote_a + vote_b, 1)
    normalized = (vote_b - vote_a) / total
    return round(max(-max_abs_score, min(max_abs_score, normalized * max_abs_score)), 3)


def make_record(
    sample_id,
    source_rel,
    edit_rel,
    label,
    improvement_score,
    vote_source,
    vote_edit,
    photo_id,
    pair_index,
):
    if label == "win":
        composition_gain = 4 if improvement_score < 1.5 else 5
        issue_tags = ["better_crop"]
    elif label == "lose":
        composition_gain = 2 if improvement_score > -1.5 else 1
        issue_tags = ["composition_worse"]
    else:
        composition_gain = 3
        issue_tags = ["composition_not_improved"]

    return {
        "id": sample_id,
        "source_image": source_rel,
        "edited_image": edit_rel,
        "edit_type": "crop",
        "data_source": "FCDB",
        "overall_label": label,
        "improvement_score": improvement_score,
        "composition_gain": composition_gain,
        "content_preservation": 5,
        "visual_naturalness": 5,
        "issue_tags": issue_tags,
        "notes": (
            f"FCDB ranking pair. photo_id={photo_id}, pair_index={pair_index}, "
            f"source_votes={vote_source}, edit_votes={vote_edit}."
        ),
    }


def build_pairs(args):
    data = load_json(args.annotation)
    output_jsonl = args.output_jsonl
    ensure_dir(output_jsonl.parent)

    source_out_dir = args.output_image_dir / "source"
    edit_out_dir = args.output_image_dir / "edit"
    ensure_dir(source_out_dir)
    ensure_dir(edit_out_dir)

    records = []
    missing_images = 0
    created = 0

    for image_index, image_record in enumerate(data):
        if args.max_images is not None and image_index >= args.max_images:
            break

        source_image = find_image(image_record, args.image_dir)
        if source_image is None and not args.metadata_only:
            missing_images += 1
            continue

        photo_id = str(image_record.get("flickr_photo_id", image_index))
        for pair_index, crop_record in enumerate(image_record.get("crops", [])):
            if args.max_pairs is not None and created >= args.max_pairs:
                break

            vote_0 = int(crop_record["vote_for_0"])
            vote_1 = int(crop_record["vote_for_1"])
            crop_0 = crop_record["crop_0"]
            crop_1 = crop_record["crop_1"]

            if abs(vote_1 - vote_0) < args.min_vote_margin:
                continue

            pairs_to_emit = [
                ("crop_0", crop_0, vote_0, "crop_1", crop_1, vote_1),
            ]
            if args.include_reverse:
                pairs_to_emit.append(("crop_1", crop_1, vote_1, "crop_0", crop_0, vote_0))

            for source_name, source_crop, source_votes, edit_name, edit_crop, edit_votes in pairs_to_emit:
                if args.max_pairs is not None and created >= args.max_pairs:
                    break

                label, _ = label_from_votes(source_votes, edit_votes, args.tie_margin)
                improvement_score = score_from_votes(source_votes, edit_votes)
                sample_id = f"rfj_fcdb_{created + 1:06d}"

                source_filename = f"{sample_id}_source.jpg"
                edit_filename = f"{sample_id}_edit.jpg"
                source_path = source_out_dir / source_filename
                edit_path = edit_out_dir / edit_filename

                copy_or_crop_image(source_image, source_crop, source_path, args.metadata_only)
                copy_or_crop_image(source_image, edit_crop, edit_path, args.metadata_only)

                source_rel = str(Path("data/pairs/images/source") / source_filename)
                edit_rel = str(Path("data/pairs/images/edit") / edit_filename)
                record = make_record(
                    sample_id,
                    source_rel,
                    edit_rel,
                    label,
                    improvement_score,
                    source_votes,
                    edit_votes,
                    photo_id,
                    pair_index,
                )
                record["notes"] += f" source={source_name}, edit={edit_name}."
                records.append(record)
                created += 1

        if args.max_pairs is not None and created >= args.max_pairs:
            break

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return {
        "records": len(records),
        "missing_images": missing_images,
        "output_jsonl": str(output_jsonl),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path, default=Path("data/pairs/images"))
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--min-vote-margin", type=int, default=1)
    parser.add_argument("--tie-margin", type=int, default=0)
    parser.add_argument("--include-reverse", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write JSONL without reading or cropping images.",
    )
    args = parser.parse_args()

    if not args.annotation.exists():
        raise SystemExit(f"Annotation file not found: {args.annotation}")
    if not args.metadata_only and args.image_dir is None:
        raise SystemExit("--image-dir is required unless --metadata-only is set")

    if not args.metadata_only:
        try:
            import PIL  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Pillow is required for cropping images: pip install Pillow") from exc

    result = build_pairs(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
