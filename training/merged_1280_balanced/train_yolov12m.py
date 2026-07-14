import os
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'

from ultralytics import YOLO

DATA    = 'data/merged_1280_balanced/data.yaml'
PROJECT = 'runs/balanced'
NAME    = 'yolo12m_balanced_1280'

print("Start training: YOLOv12m on merged_1280_balanced at 1280px...")

model = YOLO('yolo12m.pt')

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
    lr0           = 0.001,
    lrf           = 0.01,
    weight_decay  = 0.0005,
    warmup_epochs = 5,
    cos_lr        = True,
    mosaic        = 1.0,
    mixup         = 0.15,
    copy_paste    = 0.1,
    degrees       = 10.0,
    scale         = 0.5,
    fliplr        = 0.5,
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    close_mosaic  = 20,
    val           = True,
    patience      = 20,
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
