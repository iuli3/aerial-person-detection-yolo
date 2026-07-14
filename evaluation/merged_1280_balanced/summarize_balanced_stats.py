"""
stats_merged_1280_balanced.py   Detailed statistics for merged_1280_balanced
"""

from pathlib import Path

BASE = Path('data/merged_1280_balanced')
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

print("=" * 80)
print("MERGED_1280_BALANCED STATISTICS")
print("=" * 80)


def get_images(imgs_dir):
    return sorted([
        f for f in imgs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTS
    ])


def count_instances(lbl_path):
    if not lbl_path.exists():
        return 0
    with open(lbl_path, "r") as f:
        return sum(1 for line in f if line.strip())


def get_source_name(filename):
    if filename.startswith("visdrone_"):
        return "VisDrone DET"
    if filename.startswith("hdtop_"):
        return "Stanford Drone"
    if filename.startswith("suas_"):
        return "SUAS Mannequin"
    return "Unknown"


#    General statistics                                        

total_imgs = 0
total_instances = 0

for split in ["train", "val", "test"]:
    imgs_dir = BASE / "images" / split
    lbls_dir = BASE / "labels" / split

    imgs = get_images(imgs_dir)
    n_imgs = len(imgs)

    n_instances = 0
    for img in imgs:
        lbl = lbls_dir / f"{img.stem}.txt"
        n_instances += count_instances(lbl)

    total_imgs += n_imgs
    total_instances += n_instances

    print(f"\n[{split.upper()}]")
    print(f"  Imagini:   {n_imgs}")
    print(f"  Instante:  {n_instances}")
    if n_imgs > 0:
        print(f"  Medie inst/img: {n_instances / n_imgs:.2f}")

print(f"\n{'=' * 80}")
print(f"TOTAL: {total_imgs} images, {total_instances} instances")


#    Distribu ie per surs                                       

print(f"\n{'=' * 80}")
print("DISTRIBUTIE PER SURSA")
print(f"{'=' * 80}")

for split in ["train", "val", "test"]:
    imgs_dir = BASE / "images" / split
    lbls_dir = BASE / "labels" / split

    imgs = get_images(imgs_dir)
    n_split = len(imgs)

    source_data = {}

    for img in imgs:
        source = get_source_name(img.name)
        lbl = lbls_dir / f"{img.stem}.txt"
        inst = count_instances(lbl)

        if source not in source_data:
            source_data[source] = {
                "images": 0,
                "instances": 0,
            }

        source_data[source]["images"] += 1
        source_data[source]["instances"] += inst

    print(f"\n  [{split.upper()}]   {n_split} images total")
    print(f"  {'Sursa':<25} {'Imagini':>8} {'%':>7} {'Instante':>10} {'Inst/img':>9}")
    print(f"  {'-' * 68}")

    for source, data in sorted(source_data.items()):
        n_imgs = data["images"]
        n_inst = data["instances"]
        pct = n_imgs / n_split * 100 if n_split > 0 else 0
        avg = n_inst / n_imgs if n_imgs > 0 else 0

        print(f"  {source:<25} {n_imgs:>8} {pct:>6.1f}% {n_inst:>10} {avg:>9.2f}")


#    Bounding-box sizes                                            

print(f"\n{'=' * 80}")
print("DIMENSIUNI BBOX, pe baza inaltimii normalizate h")
print("  h < 0.02        = very small people")
print("  0.02 <= h < .05 = medium people")
print("  h >= 0.05       = large people")
print(f"{'=' * 80}")

for split in ["train", "val", "test"]:
    lbls_dir = BASE / "labels" / split

    small = 0
    medium = 0
    large = 0

    for lbl in lbls_dir.glob("*.txt"):
        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    h = float(parts[4])

                    if h < 0.02:
                        small += 1
                    elif h < 0.05:
                        medium += 1
                    else:
                        large += 1

    total = small + medium + large

    print(f"\n  [{split.upper()}]   {total} instances total")
    if total > 0:
        print(f"  Mici  (h < 0.02):        {small:>8} ({small / total * 100:.1f}%)")
        print(f"  Medii (0.02 <= h < .05): {medium:>8} ({medium / total * 100:.1f}%)")
        print(f"  Mari  (h >= 0.05):       {large:>8} ({large / total * 100:.1f}%)")

print("\nDone!")
