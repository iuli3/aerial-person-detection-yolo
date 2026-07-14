from pathlib import Path
import argparse
import yaml
from collections import defaultdict, Counter
import csv
import statistics

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # Does not include .npy.


def load_dataset_path(args):
    if args.data:
        data_yaml = Path(args.data)
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        root = Path(data["path"])
        splits = {
            "train": root / data.get("train", "train/images"),
            "val": root / data.get("val", "val/images"),
            "test": root / data.get("test", "test/images"),
        }
        return root, splits

    if args.dataset:
        root = Path(args.dataset)
        splits = {
            "train": root / "train/images",
            "val": root / "val/images",
            "test": root / "test/images",
        }
        return root, splits

    raise ValueError("Use either --data path/to/data.yaml or --dataset path/to/dataset")


def source_from_name(filename: str):
    name = filename.lower()

    if name.startswith("visdrone"):
        return "VisDrone"
    if name.startswith("smallobject") or name.startswith("small_object") or name.startswith("small"):
        return "SmallObject"
    if name.startswith("hdtop") or name.startswith("stanford") or name.startswith("sdd"):
        return "Stanford/HDTopView"
    if name.startswith("suas"):
        return "SUAS"
    if "visdrone" in name:
        return "VisDrone"
    if "small" in name:
        return "SmallObject"
    if "hdtop" in name or "stanford" in name:
        return "Stanford/HDTopView"
    if "suas" in name:
        return "SUAS"

    return "Unknown"


def label_path_for_image(root: Path, split: str, img_path: Path):
    labels_dir = root / split / "labels"
    return labels_dir / f"{img_path.stem}.txt"


def read_yolo_label(label_path: Path):
    """
    Returns list of boxes:
    (class_id, x_center, y_center, width, height)
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append((cls, x, y, w, h))
            except ValueError:
                continue

    return boxes


def scale_category(avg_h):
    if avg_h is None:
        return "empty"
    if avg_h < 0.02:
        return "small_h_lt_0.02"
    if avg_h < 0.05:
        return "medium_0.02_0.05"
    return "large_h_ge_0.05"


def density_category(n):
    if n <= 0:
        return "empty"
    if n <= 5:
        return "sparse_1_5"
    if n <= 15:
        return "medium_6_15"
    return "crowded_gt_15"


def collect_split_stats(root: Path, split: str, img_dir: Path):
    stats = {
        "images": 0,
        "labels": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "instances": 0,
        "sources": defaultdict(lambda: {"images": 0, "instances": 0}),
        "scale": defaultdict(lambda: {"images": 0, "instances": 0}),
        "density": defaultdict(lambda: {"images": 0, "instances": 0}),
        "classes": Counter(),
        "bbox_heights": [],
        "bbox_widths": [],
        "instances_per_image": [],
    }

    if not img_dir.exists():
        print(f"[WARN] Missing split image dir: {img_dir}")
        return stats, []

    images = [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    per_image_rows = []

    for img_path in sorted(images):
        stats["images"] += 1

        source = source_from_name(img_path.name)
        label_path = label_path_for_image(root, split, img_path)

        boxes = read_yolo_label(label_path)

        if not label_path.exists():
            stats["missing_labels"] += 1
        else:
            stats["labels"] += 1

        if label_path.exists() and len(boxes) == 0:
            stats["empty_labels"] += 1

        n_inst = len(boxes)
        stats["instances"] += n_inst
        stats["instances_per_image"].append(n_inst)

        stats["sources"][source]["images"] += 1
        stats["sources"][source]["instances"] += n_inst

        for cls, x, y, w, h in boxes:
            stats["classes"][cls] += 1
            stats["bbox_widths"].append(w)
            stats["bbox_heights"].append(h)

        avg_h = statistics.mean([b[4] for b in boxes]) if boxes else None
        sc = scale_category(avg_h)
        den = density_category(n_inst)

        stats["scale"][sc]["images"] += 1
        stats["scale"][sc]["instances"] += n_inst

        stats["density"][den]["images"] += 1
        stats["density"][den]["instances"] += n_inst

        per_image_rows.append({
            "split": split,
            "image": str(img_path),
            "label": str(label_path),
            "source": source,
            "instances": n_inst,
            "avg_bbox_h": avg_h if avg_h is not None else "",
            "scale_category": sc,
            "density_category": den,
        })

    return stats, per_image_rows


def pct(part, total):
    if total == 0:
        return 0.0
    return 100.0 * part / total


def print_stats(split, stats):
    print(f"\n================ {split.upper()} ================")
    print(f"Images:         {stats['images']}")
    print(f"Label files:    {stats['labels']}")
    print(f"Missing labels: {stats['missing_labels']}")
    print(f"Empty labels:   {stats['empty_labels']}")
    print(f"Instances:      {stats['instances']}")

    if stats["images"] > 0:
        print(f"Inst/img:       {stats['instances'] / stats['images']:.2f}")

    print("\n--- Sources ---")
    for source, d in sorted(stats["sources"].items()):
        print(
            f"{source:18s} | "
            f"images={d['images']:6d} ({pct(d['images'], stats['images']):5.1f}%) | "
            f"instances={d['instances']:8d} ({pct(d['instances'], stats['instances']):5.1f}%)"
        )

    print("\n--- Scale subsets, by average bbox height per image ---")
    for sc, d in sorted(stats["scale"].items()):
        print(
            f"{sc:18s} | "
            f"images={d['images']:6d} ({pct(d['images'], stats['images']):5.1f}%) | "
            f"instances={d['instances']:8d} ({pct(d['instances'], stats['instances']):5.1f}%)"
        )

    print("\n--- Density subsets, by persons/image ---")
    for den, d in sorted(stats["density"].items()):
        print(
            f"{den:18s} | "
            f"images={d['images']:6d} ({pct(d['images'], stats['images']):5.1f}%) | "
            f"instances={d['instances']:8d} ({pct(d['instances'], stats['instances']):5.1f}%)"
        )

    if stats["bbox_heights"]:
        h = stats["bbox_heights"]
        w = stats["bbox_widths"]
        print("\n--- Bounding box normalized size ---")
        print(f"bbox h mean:    {statistics.mean(h):.5f}")
        print(f"bbox h median:  {statistics.median(h):.5f}")
        print(f"bbox h min/max: {min(h):.5f} / {max(h):.5f}")
        print(f"bbox w mean:    {statistics.mean(w):.5f}")
        print(f"bbox w median:  {statistics.median(w):.5f}")
        print(f"bbox w min/max: {min(w):.5f} / {max(w):.5f}")

    print("\n--- Classes ---")
    for cls, count in sorted(stats["classes"].items()):
        print(f"class {cls}: {count}")


def save_csv(rows, out_path: Path):
    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-image CSV: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to YOLO data.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--out", type=str, default="dataset_stats_per_image.csv")
    args = parser.parse_args()

    root, splits = load_dataset_path(args)

    print(f"Dataset root: {root}")
    print("Image extensions used:", sorted(IMG_EXTS))
    print("NOTE: .npy files are ignored.")

    all_rows = []
    all_stats = {}

    for split, img_dir in splits.items():
        stats, rows = collect_split_stats(root, split, img_dir)
        all_stats[split] = stats
        all_rows.extend(rows)
        print_stats(split, stats)

    total_images = sum(s["images"] for s in all_stats.values())
    total_labels = sum(s["labels"] for s in all_stats.values())
    total_missing = sum(s["missing_labels"] for s in all_stats.values())
    total_instances = sum(s["instances"] for s in all_stats.values())

    print("\n================ TOTAL ================")
    print(f"Images:         {total_images}")
    print(f"Label files:    {total_labels}")
    print(f"Missing labels: {total_missing}")
    print(f"Instances:      {total_instances}")

    save_csv(all_rows, Path(args.out))


if __name__ == "__main__":
    main()
