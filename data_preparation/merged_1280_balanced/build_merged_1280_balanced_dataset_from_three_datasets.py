"""
Builds the balanced 1280px multi-source YOLO dataset.

Sources:
    - VisDrone DET official 2019 archives.
    - HDTopView / Yolov5PersonDetector-1.
    - SUAS Mannequin.

Target split sizes are chosen to keep source proportions balanced for a fair
multi-source generalization study. A fixed seed of 42 is used for reproducible
sampling.
"""

import random
import shutil
from pathlib import Path

random.seed(42)

BASE    = Path('data')
OUT     = BASE / 'merged_1280_balanced'

# Official VisDrone sources
VIS_TRAIN = BASE / 'VisDrone2019-DET-train-extracted/VisDrone2019-DET-train'
VIS_VAL   = BASE / 'VisDrone2019-DET-val-extracted/VisDrone2019-DET-val'
VIS_TEST  = BASE / 'VisDrone2019-DET-test-dev-extracted'

# HDTopView and SUAS sources
HDTOP = BASE / 'hdtopviw/Yolov5PersonDetector-1'
SUAS  = BASE / 'suas-mannequin'

EXTS = {'.jpg', '.jpeg', '.png'}

#    Output structure                                           
if OUT.exists():
    shutil.rmtree(OUT)
for split in ['train', 'val', 'test']:
    (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)


def get_imgs(folder):
    return sorted([f for f in Path(folder).glob('*.*') if f.suffix.lower() in EXTS])


def copy_imgs(img_list, dst_split, src_labels_dir, prefix=None):
    """Copies a list of images into the destination split with an optional prefix."""
    src_labels_dir = Path(src_labels_dir)
    n = 0
    for img in img_list:
        if prefix:
            dst_name = f"{prefix}_{img.name}"
            dst_lbl_name = f"{prefix}_{img.stem}.txt"
        else:
            dst_name = img.name
            dst_lbl_name = img.stem + '.txt'

        shutil.copy2(img, OUT / 'images' / dst_split / dst_name)
        lbl = src_labels_dir / (img.stem + '.txt')
        dst_lbl = OUT / 'labels' / dst_split / dst_lbl_name
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.touch()
        n += 1
    return n


def copy_visdrone(src, dst_split, prefix, limit=None):
    """Copies VisDrone images from the labels_yolo structure."""
    src = Path(src)
    imgs = sorted(src.glob('images/*.jpg'))
    if limit:
        random.shuffle(imgs)
        imgs = imgs[:limit]
    n = 0
    for img in imgs:
        lbl = src / 'labels_yolo' / (img.stem + '.txt')
        dst_img = OUT / 'images' / dst_split / f'{prefix}_{img.name}'
        dst_lbl = OUT / 'labels' / dst_split / f'{prefix}_{img.stem}.txt'
        shutil.copy2(img, dst_img)
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.touch()
        n += 1
    return n


def sample_and_copy(src_images, src_labels, dst_split, prefix, limit):
    """Samples up to <limit> images from a source folder."""
    imgs = get_imgs(src_images)
    random.shuffle(imgs)
    imgs = imgs[:limit]
    return copy_imgs(imgs, dst_split, src_labels, prefix)


print("=" * 60)
print("MERGE   merged_1280_balanced")
print("=" * 60)

#                                                               
# TRAIN
#                                                               
print("\n[TRAIN]")

# VisDrone - 5500 from the official train split (6471 available)
n = copy_visdrone(VIS_TRAIN, 'train', 'visdrone', limit=5500)
print(f"  VisDrone:   {n}")

# HDTopView - 2500 from train (2759 available)
n = sample_and_copy(HDTOP/'train/images', HDTOP/'train/labels', 'train', 'hdtop', 2500)
print(f"  HDTopView:  {n}")

# SUAS - 3500 from train (8369 available)
n = sample_and_copy(SUAS/'train/images', SUAS/'train/labels', 'train', 'suas', 3500)
print(f"  SUAS:       {n}")

#                                                               
# VAL
#                                                               
print("\n[VAL]")

# VisDrone - all official validation images (548 images)
n = copy_visdrone(VIS_VAL, 'val', 'visdrone', limit=None)
print(f"  VisDrone:   {n}")

# HDTopView - 350 from validation (787 available)
n = sample_and_copy(HDTOP/'valid/images', HDTOP/'valid/labels', 'val', 'hdtop', 350)
print(f"  HDTopView:  {n}")

# SUAS - 350 from validation (1230 available)
n = sample_and_copy(SUAS/'valid/images', SUAS/'valid/labels', 'val', 'suas', 350)
print(f"  SUAS:       {n}")

#                                                               
# TEST
#                                                               
print("\n[TEST]")

# VisDrone - 500 from test-dev (751 available)
n = copy_visdrone(VIS_TEST, 'test', 'visdrone', limit=500)
print(f"  VisDrone:   {n}")

# HDTopView - all original test images (394 images)
n = sample_and_copy(HDTOP/'test/images', HDTOP/'test/labels', 'test', 'hdtop', 394)
print(f"  HDTopView:  {n}")

# SUAS - 400 from validation after reserving 350 samples for validation
# Luam din test original (63) + completam din val
suas_test_orig = get_imgs(SUAS/'test/images')
suas_val_all   = get_imgs(SUAS/'valid/images')
random.shuffle(suas_val_all)

# Exclude images already used for validation (first 350 after shuffling with seed 42)
# Repeat the shuffle with the same seed for consistency
random.seed(42)
suas_val_shuffled = get_imgs(SUAS/'valid/images')
random.shuffle(suas_val_shuffled)
suas_used_for_val  = set(f.name for f in suas_val_shuffled[:350])
suas_remaining_val = [f for f in suas_val_shuffled if f.name not in suas_used_for_val]

suas_test_pool = suas_test_orig + suas_remaining_val
random.shuffle(suas_test_pool)
suas_test_final = suas_test_pool[:400]

# Copy SUAS test samples
n = 0
for img in suas_test_final:
    # determine the label source
    if img in suas_test_orig or img.parent.name == 'images' and (SUAS/'test/images') in img.parents:
        lbl_dir = SUAS / 'test/labels'
    else:
        lbl_dir = SUAS / 'valid/labels'
    lbl = lbl_dir / (img.stem + '.txt')
    dst_img = OUT / 'images' / 'test' / f'suas_{img.name}'
    dst_lbl = OUT / 'labels' / 'test' / f'suas_{img.stem}.txt'
    shutil.copy2(img, dst_img)
    if lbl.exists():
        shutil.copy2(lbl, dst_lbl)
    else:
        dst_lbl.touch()
    n += 1
print(f"  SUAS:       {n}")

#                                                               
# data.yaml
#                                                               
yaml = f"""path: {OUT}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['Person']
"""
(OUT / 'data.yaml').write_text(yaml)

#                                                               
# Final statistics
#                                                               
print("\n" + "=" * 60)
print("FINAL STATISTICS")
print("=" * 60)
total = 0
counts = {}
for split in ['train', 'val', 'test']:
    imgs = [f for f in (OUT / 'images' / split).iterdir() if f.suffix.lower() in EXTS]
    n = len(imgs)
    counts[split] = n
    total += n
    print(f"\n  {split.upper()}: {n} images")
    for prefix in ['visdrone', 'hdtop', 'suas']:
        k = len([f for f in imgs if f.name.startswith(prefix)])
        pct = 100 * k / n if n > 0 else 0
        print(f"    {prefix:<12}: {k:>5} ({pct:.1f}%)")

print(f"\n  TOTAL: {total} images")
print(f"\nSplit percentages:")
for split in ['train', 'val', 'test']:
    print(f"  {split}: {counts[split]/total*100:.1f}%")

print(f"\ndata.yaml written to: {OUT}")
print("Done!")