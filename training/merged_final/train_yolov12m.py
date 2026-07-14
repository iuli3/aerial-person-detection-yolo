from ultralytics import YOLO

DATA_PATH = "data/merged_final/data.yaml"
MODEL_PATH = "experiments/yolo12m/yolo12m.pt"
PROJECT_PATH = "experiments/yolo12m/runs"

print("   Start training: YOLOv12m on merged_final at 640px for 200 epochs...")

model = YOLO(MODEL_PATH)

results = model.train(
    data=DATA_PATH,
    epochs=200,
    imgsz=640,
    batch=32,
    device='0,1,2,3',
    workers=8,
    amp=True,
    project=PROJECT_PATH,
    name="yolo12m_merged_final_640_200ep",
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
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
    val=True,
    patience=50,
    cache="disk",
    deterministic=False,
    plots=True,
    exist_ok=True,
    dropout=0.0,
    save=True,
    save_period=10,
)

print("  Training completed!")
print("Results saved in:", results.save_dir)