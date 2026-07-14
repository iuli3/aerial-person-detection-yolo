from pathlib import Path
from PIL import Image
import imagehash

DATASET = Path(".")

splits = {
    "train": DATASET / "train/images",
    "valid": DATASET / "valid/images",
    "test": DATASET / "test/images",
}

items = []

for split, folder in splits.items():
    if not folder.exists():
        continue
    for img in folder.rglob("*"):
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            try:
                h = imagehash.phash(Image.open(img).convert("RGB"))
                items.append((split, img, h))
            except Exception as e:
                print("Error:", img, e)

train_items = [(p, h) for split, p, h in items if split == "train"]
test_items = [(p, h) for split, p, h in items if split == "test"]

print(f"Train images: {len(train_items)}")
print(f"Test images: {len(test_items)}")

count = 0

for test_path, test_hash in test_items:
    for train_path, train_hash in train_items:
        dist = test_hash - train_hash
        if dist <= 3:
            count += 1
            print(f"\nNEAR DUPLICATE dist={dist}")
            print("test :", test_path)
            print("train:", train_path)
            if count >= 50:
                print("\nStopped after 50 near-duplicates.")
                raise SystemExit

print(f"\nTotal near-duplicates found: {count}")
