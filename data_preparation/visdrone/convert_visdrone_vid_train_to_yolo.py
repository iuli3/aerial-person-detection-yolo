"""
Converts VisDrone VID annotations to YOLO format.
FRAME_STEP = 3   # Extract one out of every three frames.
Format VID: frame_id, target_id, x, y, w, h, score, category, truncation, occlusion
"""

from pathlib import Path
from PIL import Image
import shutil

FRAME_STEP  = 3
PERSON_CATS = {1, 2}  # pedestrian + people

BASE = Path('data/VisDrone2019-VID-train')
SEQS_DIR  = BASE / 'sequences'
ANNS_DIR  = BASE / 'annotations'
OUT       = Path('data/VisDrone_VID_YOLO')

OUT_IMGS   = OUT / 'images'
OUT_LABELS = OUT / 'labels'
OUT_IMGS.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

total_copied  = 0
total_skipped = 0

for ann_file in sorted(ANNS_DIR.glob('*.txt')):
    seq_name = ann_file.stem
    seq_dir  = SEQS_DIR / seq_name

    if not seq_dir.exists():
        print(f"[SKIP] {seq_name}")
        continue

    # Citeste adnotarile grupate pe frame_id
    frame_anns = {}
    with open(ann_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            category  = int(parts[7])
            if category not in PERSON_CATS:
                continue
            if w <= 0 or h <= 0:
                continue
            if frame_id not in frame_anns:
                frame_anns[frame_id] = []
            frame_anns[frame_id].append((x, y, w, h))

    # Process frames.
    img_files = sorted(seq_dir.glob('*.jpg'))
    seq_count = 0
    for img_file in img_files:
        frame_id = int(img_file.stem)
        if frame_id % FRAME_STEP != 0:
            continue

        anns = frame_anns.get(frame_id, [])

        try:
            with Image.open(img_file) as img:
                iw, ih = img.size
        except Exception:
            total_skipped += 1
            continue

        yolo_lines = []
        for (x, y, w, h) in anns:
            x_center = max(0, min(1, (x + w / 2) / iw))
            y_center = max(0, min(1, (y + h / 2) / ih))
            w_norm   = max(0, min(1, w / iw))
            h_norm   = max(0, min(1, h / ih))
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        out_name = f"vid_{seq_name}_{img_file.stem}"
        shutil.copy2(img_file, OUT_IMGS / (out_name + '.jpg'))
        with open(OUT_LABELS / (out_name + '.txt'), 'w') as f:
            f.write('\n'.join(yolo_lines))

        total_copied += 1
        seq_count    += 1

    print(f"[OK] {seq_name}: {seq_count} frames")

print(f"\nTotal extracted: {total_copied}")
print(f"Total sarite:  {total_skipped}")
print(f"Output: {OUT}")
