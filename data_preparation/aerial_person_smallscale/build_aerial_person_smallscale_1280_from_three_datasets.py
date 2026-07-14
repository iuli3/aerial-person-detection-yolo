from pathlib import Path
import random
import shutil
import yaml

random.seed(42)

BASE = Path("data")
OUT = BASE / "aerial_person_smallscale_1280"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# VISDRONE DET - official splits
# ============================================================

VIS_TRAIN_IMG = BASE / "VisDrone2019-DET-train-extracted/VisDrone2019-DET-train/images"
VIS_TRAIN_LBL = BASE / "VisDrone2019-DET-train-extracted/VisDrone2019-DET-train/labels_yolo"

VIS_VAL_IMG = BASE / "VisDrone2019-DET-val-extracted/VisDrone2019-DET-val/images"
VIS_VAL_LBL = BASE / "VisDrone2019-DET-val-extracted/VisDrone2019-DET-val/labels_yolo"

VIS_TEST_IMG = BASE / "VisDrone2019-DET-test-dev-extracted/images"
VIS_TEST_LBL = BASE / "VisDrone2019-DET-test-dev-extracted/labels_yolo"

# ============================================================
# SMALL OBJECT AERIAL PERSON DETECTION
# ============================================================

SMALL_TRAIN_IMG = BASE / "small_object_aerial_person_detection/Images/Train"
SMALL_TRAIN_LBL = BASE / "small_object_aerial_person_detection/Annotations/Yolo/Train"

SMALL_VAL_IMG = BASE / "small_object_aerial_person_detection/Images/Valid"
SMALL_VAL_LBL = BASE / "small_object_aerial_person_detection/Annotations/Yolo/Valid"

SMALL_TEST_IMG = BASE / "small_object_aerial_person_detection/Images/Test"
SMALL_TEST_LBL = BASE / "small_object_aerial_person_detection/Annotations/Yolo/Test"

# ============================================================
# HDTOPVIEW / STANFORD DRONE SUBSET
# ============================================================

HDTOP_TRAIN_IMG = BASE / "hdtopviw/Yolov5PersonDetector-1/train/images"
HDTOP_TRAIN_LBL = BASE / "hdtopviw/Yolov5PersonDetector-1/train/labels"

HDTOP_VAL_IMG = BASE / "hdtopviw/Yolov5PersonDetector-1/valid/images"
HDTOP_VAL_LBL = BASE / "hdtopviw/Yolov5PersonDetector-1/valid/labels"

HDTOP_TEST_IMG = BASE / "hdtopviw/Yolov5PersonDetector-1/test/images"
HDTOP_TEST_LBL = BASE / "hdtopviw/Yolov5PersonDetector-1/test/labels"

# ============================================================
# SUAS - auxiliary training source only
# ============================================================

SUAS_TRAIN_IMG = BASE / "suas-mannequin/train/images"
SUAS_TRAIN_LBL = BASE / "suas-mannequin/train/labels"

# ============================================================
# FINAL SPLIT CONFIG
# ============================================================

CONFIG = {
    "train": [
        ("visdrone", VIS_TRAIN_IMG, VIS_TRAIN_LBL, None),          # all official VisDrone train
        ("smallobject", SMALL_TRAIN_IMG, SMALL_TRAIN_LBL, None),   # all Small Object train
        ("hdtop", HDTOP_TRAIN_IMG, HDTOP_TRAIN_LBL, 900),          # reduced top-view subset
        ("suas", SUAS_TRAIN_IMG, SUAS_TRAIN_LBL, 1200),            # auxiliary only
    ],
    "val": [
        ("visdrone", VIS_VAL_IMG, VIS_VAL_LBL, None),              # all official VisDrone val
        ("smallobject", SMALL_VAL_IMG, SMALL_VAL_LBL, None),       # all Small Object val
        ("hdtop", HDTOP_VAL_IMG, HDTOP_VAL_LBL, 250),              # reduced top-view val
    ],
    "test": [
        ("visdrone", VIS_TEST_IMG, VIS_TEST_LBL, None),
        ("smallobject", SMALL_TEST_IMG, SMALL_TEST_LBL, None),
    ],
}


def label_for_image(label_dir: Path, img_path: Path) -> Path:
    return label_dir / f"{img_path.stem}.txt"


def collect_pairs(img_dir: Path, label_dir: Path):
    if not img_dir.exists():
        print(f"[WARN] image dir missing: {img_dir}")
        return []

    if not label_dir.exists():
        print(f"[WARN] label dir missing: {label_dir}")
        return []

    pairs = []
    missing = 0

    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue

        lbl = label_for_image(label_dir, img)
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing += 1

    if missing:
        print(f"[WARN] {img_dir}: {missing} images without labels")

    return pairs


def normalize_label_to_class0(src_label: Path, dst_label: Path):
    """
    Force all labels to class 0 = person.
    Expected YOLO format:
    class x_center y_center width height
    """

    lines_out = []

    with open(src_label, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            parts[0] = "0"
            lines_out.append(" ".join(parts[:5]))

    with open(dst_label, "w", encoding="utf-8") as f:
        if lines_out:
            f.write("\n".join(lines_out) + "\n")


def copy_subset(split: str, sources):
    out_img = OUT / split / "images"
    out_lbl = OUT / split / "labels"

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    print(f"\n================ {split.upper()} ================")

    split_total = 0

    for source_name, img_dir, lbl_dir, max_count in sources:
        pairs = collect_pairs(img_dir, lbl_dir)

        if not pairs:
            print(f"[WARN] {source_name}: 0 valid image-label pairs")
            continue

        random.shuffle(pairs)

        if max_count is not None:
            pairs = pairs[:max_count]

        print(f"{source_name}: copying {len(pairs)} images")

        for img, lbl in pairs:
            new_stem = f"{source_name}_{img.stem}"

            dst_img = out_img / f"{new_stem}{img.suffix.lower()}"
            dst_lbl = out_lbl / f"{new_stem}.txt"

            shutil.copy2(img, dst_img)
            normalize_label_to_class0(lbl, dst_lbl)

        split_total += len(pairs)

    print(f"TOTAL {split}: {split_total} images")


def count_instances(label_dir: Path):
    total = 0

    for lbl in label_dir.rglob("*.txt"):
        with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
            total += sum(1 for line in f if line.strip())

    return total


def count_split(split: str):
    img_dir = OUT / split / "images"
    lbl_dir = OUT / split / "labels"

    n_img = len([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    n_lbl = len(list(lbl_dir.rglob("*.txt")))
    n_inst = count_instances(lbl_dir)

    return n_img, n_lbl, n_inst


def main():
    if OUT.exists():
        print(f"[ERROR] Output already exists: {OUT}")
        print("Delete it first with:")
        print(f"rm -rf {OUT}")
        return

    for split, sources in CONFIG.items():
        copy_subset(split, sources)

    data_yaml = {
        "path": str(OUT),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "person"},
    }

    with open(OUT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("\n================ FINAL COUNTS ================")

    for split in ["train", "val", "test"]:
        n_img, n_lbl, n_inst = count_split(split)
        print(f"{split}: {n_img} images | {n_lbl} labels | {n_inst} instances")

    print(f"\nDONE: {OUT}")
    print(f"DATA YAML: {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()