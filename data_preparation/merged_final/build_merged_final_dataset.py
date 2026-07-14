"""
Builds the unified merged_final YOLO dataset for person detection.

Sources:
    1. VisDrone DET: original train/validation/test splits are preserved.
    2. VisDrone VID train: sampled video frames are assigned to train.
    3. VisDrone VID validation: sampled video frames are assigned to validation.
    4. HDTopView: validation images are redistributed 50/50 between validation and test.
    5. SUAS Mannequin: validation images are redistributed 50/50 between validation and test.

The split policy preserves the standard VisDrone benchmark while increasing the
final test set size for robust manuscript evaluation. A fixed seed of 42 is used
for reproducibility.
"""

import random
import shutil
from pathlib import Path

random.seed(42)  # fixed seed for reproducibility

BASE = Path('data')
OUT  = BASE / 'merged_final'

EXTS = {'.jpg', '.jpeg', '.png'}

# Create output structure
for split in ['train', 'val', 'test']:
    (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)


def copy_images(src_images, src_labels, dst_split, prefix):
    """
    Copy all images and labels from a source folder
    into a destination split in merged_final.
    The prefix identifies the source in the file name (ex: 'suas_img001.jpg')
    so the origin of each image can be traced if needed.
    """
    copied = 0
    src_images = Path(src_images)
    src_labels = Path(src_labels)
    for img in sorted(src_images.glob('*.*')):
        if img.suffix.lower() not in EXTS:
            continue
        lbl = src_labels / (img.stem + '.txt')
        dst_img = OUT / 'images' / dst_split / f"{prefix}_{img.name}"
        dst_lbl = OUT / 'labels' / dst_split / f"{prefix}_{img.stem}.txt"
        shutil.copy2(img, dst_img)
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            # Images without detectable people use an empty YOLO label file
            dst_lbl.touch()
        copied += 1
    return copied


def copy_split_half_to_test(src_images, src_labels, prefix):
    """
    Redistributes a dataset validation split: half to validation and half to test.

    Rationale: some datasets have a large validation split and a small test split, which would lead to
    an insufficient final test set for robust evaluation in the manuscript.
    The 50/50 split with fixed seed 42 ensures reproducibility.
    """
    src_images = Path(src_images)
    src_labels = Path(src_labels)
    imgs = sorted([f for f in src_images.glob('*.*') if f.suffix.lower() in EXTS])
    random.shuffle(imgs)  # random.seed(42) is set globally above

    half = len(imgs) // 2
    val_imgs  = imgs[:half]
    test_imgs = imgs[half:]

    for img in val_imgs:
        lbl = src_labels / (img.stem + '.txt')
        shutil.copy2(img, OUT / 'images' / 'val' / f"{prefix}_{img.name}")
        dst_lbl = OUT / 'labels' / 'val' / f"{prefix}_{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.touch()

    for img in test_imgs:
        lbl = src_labels / (img.stem + '.txt')
        shutil.copy2(img, OUT / 'images' / 'test' / f"{prefix}_{img.name}")
        dst_lbl = OUT / 'labels' / 'test' / f"{prefix}_{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.touch()

    return len(val_imgs), len(test_imgs)


print("=" * 60)
print("MERGE DATASETS   merged_final")
print("=" * 60)

#    1. VisDrone DET (Roboflow)                                 
# The original splits are preserved for compatibility with
# the standard VisDrone benchmark used in the literature
print("\n[1] VisDrone DET (Roboflow)   original splits preserved")
vd = BASE / 'VisDrone'
n = copy_images(vd/'train/images', vd/'train/labels', 'train', 'visdrone_det')
print(f"  train: {n}")
n = copy_images(vd/'valid/images', vd/'valid/labels', 'val', 'visdrone_det')
print(f"  val:   {n}")
n = copy_images(vd/'test/images',  vd/'test/labels',  'test', 'visdrone_det')
print(f"  test:  {n}")

#    2. VisDrone VID train                                      
# Video frames add temporal diversity; the same scene
# is captured from different angles and altitudes during a flight.
# Assigned entirely to train because no predefined split exists.
print("\n[2] VisDrone VID train   assigned entirely to train because no predefined split exists")
vid = BASE / 'VisDrone_VID_YOLO'
n = copy_images(vid/'images', vid/'labels', 'train', 'visdrone_vid')
print(f"  train: {n}")

#    3. VisDrone VID val                                        
# Original VisDrone VID validation sequences are assigned to validation
# Preserve the semantics of the original dataset split.
print("\n[3] VisDrone VID val   assigned entirely to validation")
vid_val = BASE / 'VisDrone_VID_val_YOLO'
n = copy_images(vid_val/'images', vid_val/'labels', 'val', 'visdrone_vid_val')
print(f"  val: {n}")

#    4. HDTopView                                               
# Validation images (787) are redistributed 50/50 between validation and test
# The original test split (394 images) is already large enough,
# because the larger validation split would skew the final proportions.
print("\n[4] HDTopView   validation redistributed 50/50 between validation and test")
hd = BASE / 'hdtopviw/Yolov5PersonDetector-1'
n = copy_images(hd/'train/images', hd/'train/labels', 'train', 'hdtop')
print(f"  train: {n}")
n_val, n_test = copy_split_half_to_test(hd/'valid/images', hd/'valid/labels', 'hdtop')
n_test_orig = copy_images(hd/'test/images', hd/'test/labels', 'test', 'hdtop_orig')
print(f"  val:   {n_val}")
print(f"  test:  {n_test + n_test_orig} ({n_test} din val + {n_test_orig} test original)")

#    5. SUAS Mannequin                                          
# Validation images (1230) are redistributed 50/50 between validation and test
# because the original test split has only 63 images, which is insufficient
# for robust final evaluation reported in the manuscript.
print("\n[5] SUAS Mannequin   validation redistributed 50/50 between validation and test")
suas = BASE / 'suas-mannequin'
n = copy_images(suas/'train/images', suas/'train/labels', 'train', 'suas')
print(f"  train: {n}")
n_val, n_test = copy_split_half_to_test(suas/'valid/images', suas/'valid/labels', 'suas')
n_test_orig = copy_images(suas/'test/images', suas/'test/labels', 'test', 'suas_orig')
print(f"  val:   {n_val}")
print(f"  test:  {n_test + n_test_orig} ({n_test} din val + {n_test_orig} test original)")

#    Final statistics                                          
print("\n" + "=" * 60)
print("FINAL STATISTICS merged_final")
print("=" * 60)
total = 0
counts = {}
for split in ['train', 'val', 'test']:
    n = len(list((OUT / 'images' / split).glob('*')))
    counts[split] = n
    total += n
    print(f"  {split}: {n} images")
print(f"  total: {total}")
print(f"\nProcentaje:")
for split in ['train', 'val', 'test']:
    print(f"  {split}: {counts[split]/total*100:.1f}%")

#    Statistics by source                                       
print("\nSource distribution in train:")
for prefix in ['visdrone_det', 'visdrone_vid', 'hdtop', 'suas']:
    n = len(list((OUT / 'images' / 'train').glob(f'{prefix}_*')))
    print(f"  {prefix}: {n} ({n/counts['train']*100:.1f}%)")

#    data.yaml                                                  
yaml = f"""path: {OUT}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['Person']
"""
(OUT / 'data.yaml').write_text(yaml)
print(f"\ndata.yaml written to: {OUT}")
print("Done!")
