from ultralytics import YOLO

DATA_PATH = "data/merged_final/data.yaml"

print("   Start training: YOLOv26m on merged_final at 640px...")

model = YOLO("experiments/yolo26m/yolo26m.pt")

model.train(
    data=DATA_PATH,
    epochs=100,
    imgsz=640,
    batch=32,
    amp=True,
    device=3,
    project="experiments/yolo26m/runs",
    name="yolo26m_merged_final_640px",
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    degrees=10.0,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    close_mosaic=20,
    patience=50,
    val=True,
    deterministic=False,
    plots=True,
    exist_ok=True,
    weight_decay=0.0005,
    dropout=0.0,
    workers=2,
    save=True,
    save_period=10,
    cache='disk',
)

print("  Training completed!")