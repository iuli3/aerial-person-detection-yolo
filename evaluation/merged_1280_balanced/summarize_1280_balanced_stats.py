from pathlib import Path
import argparse
import yaml
from collections import defaultdict, Counter
import csv
import statistics

# IMPORTANT: .npy files are not included here.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(data_yaml_path: str):
    data_yaml = Path(data_yaml_path)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data["path"])

    splits = {
        "train": root / data.get("train", "images/train"),
        "val": root / data.get("val", "images/val"),
        "test": root / data.get("test", "images/test"),
    }

    return root, splits


def source_from_filename(filename: str):
    name = filename.lower()

    if "visdrone" in name:
        return "VisDrone2019-DET"

    # In the manuscript, this source is referred to as Stanford Drone Dataset,
    # Even if files use the hdtop prefix.
    if "hdtop" in name or "stanford" in name:
        return "Stanford Drone Dataset"

    if "suas" in name:
        return "SUAS Mannequin"

    return "Unknown"


def label_path_for_image(root: Path, split: str, img_path: Path):
    """
    Pentru structura:
    images/train/*.jpg
    labels/train/*.txt
    """

    candidates = [
        root / "labels" / split / f"{img_path.stem}.txt",
        root / split / "labels" / f"{img_path.stem}.txt",
    ]

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def read_yolo_label(label_path: Path):
    """
    YOLO format:
    class x_center y_center width height
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
    if n == 0:
        return "empty"

    if n <= 5:
        return "sparse_1_5"

    if n <= 15:
        return "medium_6_15"

    return "crowded_gt_15"


def pct(part, total):
    if total == 0:
        return 0.0
    return 100.0 * part / total


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

    rows = []

    if not img_dir.exists():
        print(f"[WARN] Missing image dir for split {split}: {img_dir}")
        return stats, rows

    images = [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    for img_path in sorted(images):
        stats["images"] += 1

        source = source_from_filename(img_path.name)
        label_path = label_path_for_image(root, split, img_path)

        if label_path.exists():
            stats["labels"] += 1
        else:
            stats["missing_labels"] += 1

        boxes = read_yolo_label(label_path)

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

        rows.append({
            "split": split,
            "image": str(img_path),
            "label": str(label_path),
            "source": source,
            "instances": n_inst,
            "avg_bbox_h": avg_h if avg_h is not None else "",
            "scale_category": sc,
            "density_category": den,
        })

    return stats, rows


def print_stats(split: str, stats):
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
            f"{source:24s} | "
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

    print(f"\nSaved CSV: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO data.yaml"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="merged_1280_balanced_stats.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    root, splits = load_data_yaml(args.data)

    print(f"Dataset root: {root}")
    print(f"Image extensions used: {sorted(IMG_EXTS)}")
    print("NOTE: .npy files are ignored.")

    all_stats = {}
    all_rows = []

    for split, img_dir in splits.items():
        stats, rows = collect_split_stats(root, split, img_dir)
        all_stats[split] = stats
        all_rows.extend(rows)
        print_stats(split, stats)

    total_images = sum(s["images"] for s in all_stats.values())
    total_labels = sum(s["labels"] for s in all_stats.values())
    total_missing = sum(s["missing_labels"] for s in all_stats.values())
    total_empty = sum(s["empty_labels"] for s in all_stats.values())
    total_instances = sum(s["instances"] for s in all_stats.values())

    print("\n================ TOTAL ================")
    print(f"Images:         {total_images}")
    print(f"Label files:    {total_labels}")
    print(f"Missing labels: {total_missing}")
    print(f"Empty labels:   {total_empty}")
    print(f"Instances:      {total_instances}")

    save_csv(all_rows, Path(args.out))


if __name__ == "__main__":
    main()
