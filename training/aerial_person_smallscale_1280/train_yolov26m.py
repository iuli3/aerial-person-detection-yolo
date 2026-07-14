from ultralytics import YOLO

DATA = "data/aerial_person_smallscale_1280/data.yaml"
PROJECT = "experiments/runs/aerial_person_smallscale_1280"

COMMON = dict(
    data=DATA,
    imgsz=1280,
    epochs=50,
    batch=12,
    device="1,2,3",
    workers=8,
    amp=True,
    optimizer="AdamW",
    lrf=0.01,
    warmup_epochs=5,
    cos_lr=True,
    cache="disk",
    val=True,
    plots=True,
    exist_ok=True,
    save=True,
    save_period=10,
)

# YOLOv26m regularized
YOLO("yolo26m.pt").train(
    name="yolo26m_aerial_person_smallscale_1280_reg",
    lr0=0.0005,
    weight_decay=0.001,
    patience=12,
    mosaic=0.7,
    mixup=0.05,
    copy_paste=0.0,
    degrees=5,
    scale=0.4,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    close_mosaic=15,
    project=PROJECT,
    **COMMON
)