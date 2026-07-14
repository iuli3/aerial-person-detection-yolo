from ultralytics import YOLO
from pathlib import Path
import csv
import traceback

PROJECT = Path("workspace_path_placeholder")
PROJECT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = PROJECT / "summary_tinyperson_1280.csv"

TINYPERSON = "data/tinyperson/data.yaml"

MODELS = {
    "yolo11m": "runs/balanced/yolo11m_balanced_1280/weights/best.pt",
    "yolo12m": "runs/balanced/yolo12m_balanced_1280/weights/best.pt",
    "yolo26m": "runs/balanced/yolo26m_balanced_1280_reg/weights/best.pt",
}

rows = []

for model_name, model_path in MODELS.items():
    run_name = f"{model_name}_tinyperson_test"

    print("\n" + "=" * 80)
    print(f"Running TinyPerson evaluation: {run_name}")
    print(f"Model: {model_path}")
    print(f"Data:  {TINYPERSON}")
    print("=" * 80)

    try:
        model = YOLO(model_path)

        metrics = model.val(
            data=TINYPERSON,
            imgsz=1280,
            device=0,
            split="test",
            project=str(PROJECT),
            name=run_name,
            plots=True,
            save_json=True,
            exist_ok=True,
            verbose=True,
        )

        row = {
            "model": model_name,
            "dataset": "tinyperson",
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "save_dir": str(metrics.save_dir),
        }

        rows.append(row)

        print(f"\nRESULT {run_name}:")
        print(f"  Precision:  {row['precision']:.5f}")
        print(f"  Recall:     {row['recall']:.5f}")
        print(f"  mAP50:      {row['mAP50']:.5f}")
        print(f"  mAP50-95:   {row['mAP50-95']:.5f}")
        print(f"  Saved in:   {row['save_dir']}")

    except Exception as e:
        print(f"\nERROR while running {run_name}: {e}")
        traceback.print_exc()

        rows.append({
            "model": model_name,
            "dataset": "tinyperson",
            "precision": "ERROR",
            "recall": "ERROR",
            "mAP50": "ERROR",
            "mAP50-95": "ERROR",
            "save_dir": "",
        })

with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "dataset", "precision", "recall", "mAP50", "mAP50-95", "save_dir"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 80)
print("TINYPERSON EVALUATION FINISHED")
print(f"Summary saved to: {SUMMARY_CSV}")
print("=" * 80)

print("\nSUMMARY:")
for r in rows:
    print(
        f"{r['model']:<8} | {r['dataset']:<10} | "
        f"P={r['precision']} | R={r['recall']} | "
        f"mAP50={r['mAP50']} | mAP50-95={r['mAP50-95']}"
    )