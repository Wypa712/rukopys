import argparse
import random
from pathlib import Path

from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


DEFAULT_CLASSES = [
    "handwritten",
    "printed",
    "formula",
    "table",
    "annotation",
    "image",
    "graph",
]


def yolo_bbox_from_xyxy(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(float(x1), image_width))
    x2 = max(0, min(float(x2), image_width))
    y1 = max(0, min(float(y1), image_height))
    y2 = max(0, min(float(y2), image_height))

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2

    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def region_to_yolo_row(region, image_width, image_height, classes):
    region_type = region.get("type")
    if region_type not in classes:
        return None

    x_center, y_center, width, height = yolo_bbox_from_xyxy(
        region["bbox"],
        image_width,
        image_height,
    )
    if width <= 0 or height <= 0:
        return None

    class_id = classes.index(region_type)
    return (
        f"{class_id} "
        f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    )


def split_indices(num_items, val_ratio, seed):
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(num_items * val_ratio))) if num_items > 1 else 0
    val_indices = set(indices[:val_count])
    train_indices = [idx for idx in indices if idx not in val_indices]
    val_indices = [idx for idx in indices if idx in val_indices]
    return train_indices, val_indices


def write_data_yaml(output_dir, classes):
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(classes))
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_split(dataset, indices, split_name, output_dir, classes):
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped_empty = 0

    for page_idx in tqdm(indices, desc=f"export {split_name}"):
        page = dataset[int(page_idx)]
        image = page["image"]
        image_width, image_height = image.size

        rows = [
            row
            for row in (
                region_to_yolo_row(region, image_width, image_height, classes)
                for region in page["regions"]
            )
            if row is not None
        ]

        if not rows:
            skipped_empty += 1
            continue

        image_path = images_dir / f"page_{int(page_idx):06d}.jpg"
        label_path = labels_dir / f"page_{int(page_idx):06d}.txt"

        image.convert("RGB").save(image_path, quality=95)
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        exported += 1

    return {"exported": exported, "skipped_empty": skipped_empty}


def export_detector_dataset(
    dataset,
    output_dir,
    classes,
    val_ratio=0.2,
    seed=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_indices, val_indices = split_indices(len(dataset), val_ratio, seed)
    train_stats = export_split(dataset, train_indices, "train", output_dir, classes)
    val_stats = export_split(dataset, val_indices, "val", output_dir, classes)
    write_data_yaml(output_dir, classes)

    return {"train": train_stats, "val": val_stats, "classes": classes}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RUKOPYS page annotations to YOLO detector format.",
    )
    parser.add_argument("--output-dir", default="detector_data")
    parser.add_argument("--dataset-name", default="UkrainianCatholicUniversity/rukopys")
    parser.add_argument("--config", default="gt_only")
    parser.add_argument("--split", default="train")
    parser.add_argument("--classes", nargs="+", default=["handwritten"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    from datasets import load_dataset

    Image.MAX_IMAGE_PIXELS = 200_000_000
    args = parse_args()
    dataset = load_dataset(args.dataset_name, args.config, split=args.split)
    stats = export_detector_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        classes=args.classes,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(stats)


if __name__ == "__main__":
    main()
