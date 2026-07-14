from pathlib import Path
import hashlib
from collections import defaultdict

DATASET = Path(".")

splits = {
    "train": DATASET / "train/images",
    "valid": DATASET / "valid/images",
    "test": DATASET / "test/images",
}

print("Found splits:")
for name, path in splits.items():
    print(f"  {name}: {path} -> exists={path.exists()}")

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

hash_map = defaultdict(list)

for split, folder in splits.items():
    if not folder.exists():
        continue
    for img in folder.rglob("*"):
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            hash_map[file_hash(img)].append((split, img))

duplicates = []

for h, items in hash_map.items():
    used_splits = {x[0] for x in items}
    if len(used_splits) > 1:
        duplicates.append((h, items))

print(f"\nTotal duplicate groups across splits: {len(duplicates)}")

for h, items in duplicates[:50]:
    print("\nDUPLICATE GROUP:")
    for split, path in items:
        print(f"  {split}: {path}")

if len(duplicates) > 50:
    print(f"\n...  nc  {len(duplicates) - 50} grupuri duplicate neafi ate.")
