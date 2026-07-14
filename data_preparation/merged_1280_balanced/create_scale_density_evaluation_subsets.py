from pathlib import Path
import shutil

SRC = Path("data/merged_1280_balanced")
OUT = Path("data/eval_scale_density_balanced")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if OUT.exists():
    shutil.rmtree(OUT)

scale_subsets = {
    "small_high_altitude": [],
    "medium_scale": [],
    "large_low_altitude": [],
}

density_subsets = {
    "sparse": [],
    "medium_density": [],
    "crowded": [],
}

def read_heights(lbl_path):
    heights = []
    if not lbl_path.exists():
        return heights

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                heights.append(float(parts[4]))
    return heights

def copy_subset(items, subset_name):
    img_out = OUT / subset_name / "images" / "test"
    lbl_out = OUT / subset_name / "labels" / "test"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_inst = 0

    for img, lbl in items:
        shutil.copy2(img, img_out / img.name)

        dst_lbl = lbl_out / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
            with open(lbl, "r") as f:
                total_inst += sum(1 for line in f if line.strip())
        else:
            dst_lbl.touch()

    yaml_text = f"""path: {OUT / subset_name}
train: images/test
val: images/test
test: images/test
nc: 1
names: ['Person']
"""
    (OUT / subset_name / "data.yaml").write_text(yaml_text)

    print(f"{subset_name}: {len(items)} images, {total_inst} instances")

images_dir = SRC / "images" / "test"
labels_dir = SRC / "labels" / "test"

images = sorted([
    p for p in images_dir.iterdir()
    if p.is_file() and p.suffix.lower() in EXTS
])

for img in images:
    lbl = labels_dir / f"{img.stem}.txt"
    heights = read_heights(lbl)

    if not heights:
        continue

    avg_h = sum(heights) / len(heights)
    n_inst = len(heights)

    # Scale-based grouping: proxy for altitude.
    if avg_h < 0.02:
        scale_subsets["small_high_altitude"].append((img, lbl))
    elif avg_h < 0.05:
        scale_subsets["medium_scale"].append((img, lbl))
    else:
        scale_subsets["large_low_altitude"].append((img, lbl))

    # Density-based grouping
    if n_inst <= 5:
        density_subsets["sparse"].append((img, lbl))
    elif n_inst <= 15:
        density_subsets["medium_density"].append((img, lbl))
    else:
        density_subsets["crowded"].append((img, lbl))

print("=" * 80)
print("SCALE SUBSETS")
print("=" * 80)
for name, items in scale_subsets.items():
    copy_subset(items, f"scale_{name}")

print("\n" + "=" * 80)
print("DENSITY SUBSETS")
print("=" * 80)
for name, items in density_subsets.items():
    copy_subset(items, f"density_{name}")

print("\nDone.")
