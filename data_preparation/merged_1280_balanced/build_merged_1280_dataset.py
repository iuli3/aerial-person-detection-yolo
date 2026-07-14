from pathlib import Path
import shutil
import random

random.seed(42)

VIS_TRAIN = Path('data/VisDrone2019-DET-train-extracted/VisDrone2019-DET-train')
VIS_VAL   = Path('data/VisDrone2019-DET-val-extracted/VisDrone2019-DET-val')
VIS_TEST  = Path('data/VisDrone2019-DET-test-dev-extracted')
MERGED    = Path('data/merged_final')
OUT       = Path('data/merged_1280')

if OUT.exists():
    shutil.rmtree(OUT)

for split in ['train', 'val', 'test']:
    (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)

def copy_visdrone(src, dst_split, prefix):
    n = 0
    for img in sorted(src.glob('images/*.jpg')):
        lbl = src / 'labels_yolo' / (img.stem + '.txt')
        shutil.copy2(img, OUT / 'images' / dst_split / f'{prefix}_{img.name}')
        dst_lbl = OUT / 'labels' / dst_split / f'{prefix}_{img.stem}.txt'
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.touch()
        n += 1
    return n

def copy_imgs(img_list, dst_split, src_labels_dir):
    for img in img_list:
        lbl = src_labels_dir / (img.stem + '.txt')
        shutil.copy2(img, OUT / 'images' / dst_split / img.name)
        if lbl.exists():
            shutil.copy2(lbl, OUT / 'labels' / dst_split / lbl.name)
        else:
            (OUT / 'labels' / dst_split / (img.stem + '.txt')).touch()

def copy_from_merged(src_split, dst_split, prefix, limit=None):
    imgs = sorted((MERGED / 'images' / src_split).glob(f'{prefix}_*'))
    if limit:
        random.shuffle(imgs)
        imgs = imgs[:limit]
    copy_imgs(imgs, dst_split, MERGED / 'labels' / src_split)
    return len(imgs)

# SUAS split
suas_val_all = sorted((MERGED / 'images' / 'val').glob('suas_*'))
random.shuffle(suas_val_all)
suas_for_val  = suas_val_all[:500]
suas_for_test = suas_val_all[500:800]

# HDTopView split
hdtop_val_all = sorted((MERGED / 'images' / 'val').glob('hdtop_*'))
random.shuffle(hdtop_val_all)
hdtop_for_test = hdtop_val_all[:350]
hdtop_for_val  = hdtop_val_all[350:]

print("TRAIN:")
n = copy_visdrone(VIS_TRAIN, 'train', 'visdrone_train')
print(f"  VisDrone: {n}")
n = copy_from_merged('train', 'train', 'hdtop')
print(f"  HDTopView: {n}")
n = copy_from_merged('train', 'train', 'suas', 2500)
print(f"  SUAS: {n}")

print("\nVAL:")
n = copy_visdrone(VIS_VAL, 'val', 'visdrone_val')
print(f"  VisDrone: {n}")
copy_imgs(hdtop_for_val, 'val', MERGED / 'labels' / 'val')
print(f"  HDTopView: {len(hdtop_for_val)}")
copy_imgs(suas_for_val, 'val', MERGED / 'labels' / 'val')
print(f"  SUAS: {len(suas_for_val)}")

print("\nTEST:")
n = copy_visdrone(VIS_TEST, 'test', 'visdrone_test')
print(f"  VisDrone: {n}")
copy_imgs(hdtop_for_test, 'test', MERGED / 'labels' / 'val')
print(f"  HDTopView: {len(hdtop_for_test)}")
copy_imgs(suas_for_test, 'test', MERGED / 'labels' / 'val')
print(f"  SUAS: {len(suas_for_test)}")

yaml = f"""path: {OUT}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['Person']
"""
(OUT / 'data.yaml').write_text(yaml)

print("\nFINAL STATISTICS:")
total = 0
counts = {}
for split in ['train', 'val', 'test']:
    n = len(list((OUT / 'images' / split).glob('*')))
    counts[split] = n
    total += n
    print(f"  {split}: {n}")
print(f"  TOTAL: {total}")
for split in ['train', 'val', 'test']:
    print(f"  {split}: {counts[split]/total*100:.1f}%")
print("\nDone!")
