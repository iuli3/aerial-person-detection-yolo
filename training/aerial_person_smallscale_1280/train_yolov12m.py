from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path("data/aerial_person_smallscale_1280")

DATA = BASE_DIR / "aerial_person_smallscale_1280/data.yaml"
PROJECT = BASE_DIR / "runs"

model = YOLO("yolo12m.pt")

model.train(
    data=str(DATA),
    imgsz=1280,
    epochs=50,
    batch=16,
    device="0",
    workers=8,
    amp=True,

    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,

    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    degrees=10,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    close_mosaic=20,

    patience=20,
    cache="disk",
    val=True,
    plots=True,
    exist_ok=True,
    save=True,
    save_period=10,

    project=str(PROJECT),
    name="yolo12m_aerial_person_smallscale_1280"
)