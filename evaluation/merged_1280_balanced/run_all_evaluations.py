from ultralytics import YOLO
from pathlib import Path
import csv
import traceback

# =========================
# Paths
# =========================

PROJECT = Path("workspace_path_placeholder")
PROJECT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = PROJECT / "summary_results.csv"

DATASETS = {
    "global": "data/merged_1280_balanced/data.yaml",
    "visdrone": "data/eval_subsets_balanced/visdrone_test/data.yaml",
    "stanford": "data/eval_subsets_balanced/stanford_test/data.yaml",
    "suas": "data/eval_subsets_balanced/suas_test/data.yaml",
}

MODELS = {
    "yolo11m": "runs/balanced/yolo11m_balanced_1280/weights/best.pt",
    "yolo12m": "runs/balanced/yolo12m_balanced_1280/weights/best.pt",
    "yolo26m": "runs/balanced/yolo26m_balanced_1280_reg/weights/best.pt",
}

# =========================
# Evaluation
# =========================

rows = []

for model_name, model_path in MODELS.items():
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print(f"PATH:  {model_path}")
    print("=" * 80)

    model = YOLO(model_path)

    for dataset_name, data_path in DATASETS.items():
        run_name = f"{model_name}_{dataset_name}_test"

        print("\n" + "-" * 80)
        print(f"Running evaluation: {run_name}")
        print(f"Data: {data_path}")
        print("-" * 80)

        try:
            metrics = model.val(
                data=data_path,
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

            # Ultralytics metrics
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)

            row = {
                "model": model_name,
                "dataset": dataset_name,
                "precision": precision,
                "recall": recall,
                "mAP50": map50,
                "mAP50-95": map5095,
                "save_dir": str(metrics.save_dir),
            }

            rows.append(row)

            print(f"\nRESULT {run_name}:")
            print(f"  Precision:  {precision:.5f}")
            print(f"  Recall:     {recall:.5f}")
            print(f"  mAP50:      {map50:.5f}")
            print(f"  mAP50-95:   {map5095:.5f}")
            print(f"  Saved in:   {metrics.save_dir}")

        except Exception as e:
            print(f"\nERROR while running {run_name}: {e}")
            traceback.print_exc()

            rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "precision": "ERROR",
                "recall": "ERROR",
                "mAP50": "ERROR",
                "mAP50-95": "ERROR",
                "save_dir": "",
            })

# =========================
# Save summary CSV
# =========================

with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "dataset", "precision", "recall", "mAP50", "mAP50-95", "save_dir"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 80)
print("ALL EVALUATIONS FINISHED")
print(f"Summary saved to: {SUMMARY_CSV}")
print("=" * 80)

print("\nSUMMARY:")
for r in rows:
    print(
        f"{r['model']:<8} | {r['dataset']:<9} | "
        f"P={r['precision']} | R={r['recall']} | "
        f"mAP50={r['mAP50']} | mAP50-95={r['mAP50-95']}"
    )