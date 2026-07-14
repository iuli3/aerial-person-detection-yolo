"""
evaluate_all_models.py   Complete model evaluation on all datasets
Calculeaza: mAP50, mAP50-95, Precision, Recall, F1
"""

from ultralytics import YOLO
import json
import os

#    Models to evaluate                                                          
MODELS = {
    'YOLOv11m_merged_all3': 'runs/yolo11/runs/detect/yolo11m_merged_all3/run1_200ep/weights/best.pt',
    'YOLOv12m_merged_all3': 'runs/yolo12/runs/detect/yolo12m_merged_all3/run1_200ep/weights/best.pt',
    'YOLOv26m_merged_all3': 'runs/yolo26/runs/detect/yolo26m_merged_all3/run1_200ep/weights/best.pt',
}

#    Datasets to evaluate                                                        
DATASETS = {
    'merged_all3': 'data/merged_all3/data.yaml',
    'VisDrone_DET': 'data/VisDrone/data.yaml',
    'HDTopView': 'data/hdtopviw/Yolov5PersonDetector-1/data.yaml',
    'SUAS_Mannequin': 'data/suas-mannequin/data.yaml',
}

IMGSZ = 640
DEVICE = 0  # Change if GPU 0 is busy.

results_all = {}

for model_name, model_path in MODELS.items():
    if not os.path.exists(model_path):
        print(f"[SKIP] Model negasit: {model_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = YOLO(model_path)
    results_all[model_name] = {}

    for dataset_name, yaml_path in DATASETS.items():
        if not os.path.exists(yaml_path):
            print(f"  [SKIP] {dataset_name}   YAML not found")
            continue

        print(f"\n  Evaluation pe {dataset_name}...")

        try:
            metrics = model.val(
                data=yaml_path,
                split='val',
                imgsz=IMGSZ,
                batch=8,
                device=DEVICE,
                verbose=False,
                plots=True,
                save_json=False,
                project=f'experiments/eval_results_640px/{model_name}',
                name=dataset_name,
                exist_ok=True,
            )

            mp      = float(metrics.box.mp)
            mr      = float(metrics.box.mr)
            map50   = float(metrics.box.map50)
            map5095 = float(metrics.box.map)
            f1      = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

            results_all[model_name][dataset_name] = {
                'mAP50':     round(map50,   4),
                'mAP50_95':  round(map5095, 4),
                'Precision': round(mp,      4),
                'Recall':    round(mr,      4),
                'F1':        round(f1,      4),
            }

            print(f"    mAP50:     {map50:.4f}")
            print(f"    mAP50-95:  {map5095:.4f}")
            print(f"    Precision: {mp:.4f}")
            print(f"    Recall:    {mr:.4f}")
            print(f"    F1:        {f1:.4f}")

        except Exception as e:
            print(f"  [EROARE] {dataset_name}: {e}")
            results_all[model_name][dataset_name] = {'eroare': str(e)}

#    Save JSON                                                               
out_path = 'experiments/eval_results/results_summary.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results_all, f, indent=2, ensure_ascii=False)
print(f"\nResults saved in: {out_path}")

#    Print final table                                                          
print(f"\n{'='*95}")
print("FINAL RESULTS TABLE")
print(f"{'='*95}")
print(f"{'Model':<30} {'Dataset':<20} {'mAP50':>7} {'mAP50-95':>9} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print(f"{'-'*95}")

for model_name, datasets in results_all.items():
    for dataset_name, m in datasets.items():
        if 'eroare' in m:
            print(f"{model_name:<30} {dataset_name:<20} {'EROARE':>40}")
            continue
        print(
            f"{model_name:<30} {dataset_name:<20} "
            f"{m['mAP50']:>7.4f} {m['mAP50_95']:>9.4f} "
            f"{m['Precision']:>7.4f} {m['Recall']:>7.4f} "
            f"{m['F1']:>7.4f}"
        )

print(f"{'='*95}")
print(f"\nPlots and results in: experiments/eval_results/")
