import shutil
from pathlib import Path

DS1 = Path("data/VisDrone")
DS2 = Path("data/hdtopviw/Yolov5PersonDetector-1")
OUT = Path("data/merged_person")

splits = [("train", "train"), ("valid", "val"), ("test", "test")]

for folder, split_out in splits:
    for dtype in ["images", "labels"]:
        out_dir = OUT / dtype / split_out
        out_dir.mkdir(parents=True, exist_ok=True)

        for ds_name, ds_path in [("ds1", DS1), ("ds2", DS2)]:
            src = ds_path / folder / dtype
            if not src.exists():
                print(f"    Nu exist : {src}")
                continue
            copied = 0
            for f in src.iterdir():
                dst = out_dir / f"{ds_name}_{f.name}"
                shutil.copy2(f, dst)
                copied += 1
            print(f"  {ds_name} {folder}/{dtype}: {copied} fi iere copiate")

yaml_content = """path: data/merged_person
train: images/train
val: images/val
test: images/test

nc: 1
names: ['Person']
"""
(OUT / "data.yaml").write_text(yaml_content)
print("\n  data.yaml written!")
print(f"   Dataset merged in: {OUT}")
