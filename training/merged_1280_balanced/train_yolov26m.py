import os
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'

from ultralytics import YOLO

DATA    = './datasets/merged_1280_balanced/data.yaml'
PROJECT = './runs/balanced'
NAME    = 'yolo26m_balanced_1280_reg'

print("Start training: YOLOv26m on merged_1280_balanced at 1280px...")
print(f"Working from processing_server root directory")

print("Start training: YOLOv26m on merged_1280_balanced at 1280px...")

model = YOLO('yolo26m.pt')

results = model.train(
    data          = DATA,
    epochs        = 50,
    imgsz         = 1280,
    batch         = -1,
    device        = '0',
    workers       = 8,
    amp           = True,

    project       = PROJECT,
    name          = NAME,

    optimizer     = 'AdamW',
    lr0           = 0.0005,      # Lower than YOLOv12.
    lrf           = 0.01,
    weight_decay  = 0.001,       # regularizare mai puternica
    warmup_epochs = 5,
    cos_lr        = True,

    # augmentari mai moderate decat la YOLOv12
    mosaic        = 0.7,
    mixup         = 0.05,
    copy_paste    = 0.0,
    degrees       = 5.0,
    scale         = 0.4,
    fliplr        = 0.5,
    hsv_h         = 0.015,
    hsv_s         = 0.5,
    hsv_v         = 0.3,
    close_mosaic  = 15,

    val           = True,
    patience      = 12,          # Stop early if overfitting starts.
    cache         = 'disk',
    deterministic = False,
    plots         = True,
    exist_ok      = True,

    dropout       = 0.0,
    save          = True,
    save_period   = 10,
)

print("Training completed!")
print("Results saved in:", results.save_dir)
