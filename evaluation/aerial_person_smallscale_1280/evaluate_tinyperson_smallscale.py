from ultralytics import YOLO
from pathlib import Path
import csv
import json
from datetime import datetime
import yaml

BASE_DIR = Path("workspace_path_placeholder")

# Change this path if TinyPerson is stored elsewhere.
TINYPERSON_DATA = BASE_DIR / "datasets/tinyperson/data.yaml"

RUNS = {
    "YOLOv11m_smallscale": Path("data/aerial_person_smallscale_1280/runs/yolo11m_aerial_person_smallscale_1280/weights/best.pt"),
    "YOLOv12m_smallscale": Path("data/aerial_person_smallscale_1280/runs/yolo12m_aerial_person_smallscale_1280/weights/best.pt"),
    "YOLOv26m_smallscale": Path("data/new_train_small/runs/aerial_person_smallscale_1280/yolo26m_aerial_person_smallscale_1280_reg/weights/best.pt"),
}

OUTPUT_DIR = BASE_DIR / "results_tinyperson/smallscale_models"
VAL_PROJECT = OUTPUT_DIR / "ultralytics_val_runs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_PROJECT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 1280
BATCH = 8
DEVICE = "0"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def choose_split(data_yaml: Path) -> str:
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    if "test" in data and data["test"]:
        return "test"
    if "val" in data and data["val"]:
        return "val"

    raise ValueError("data.yaml does not contain a valid 'test' or 'val' split.")


def extract_metrics(metrics):
    box = metrics.box

    try:
        precision = safe_float(box.p[0])
    except Exception:
        precision = safe_float(box.mp)

    try:
        recall = safe_float(box.r[0])
    except Exception:
        recall = safe_float(box.mr)

    return {
        "precision": precision,
        "recall": recall,
        "mAP50": safe_float(box.map50),
        "mAP50_95": safe_float(box.map),
    }


def main():
    if not TINYPERSON_DATA.exists():
        raise FileNotFoundError(
            f"TinyPerson data.yaml not found: {TINYPERSON_DATA}\n"
            f"Modify TINYPERSON_DATA in the script."
        )

    split = choose_split(TINYPERSON_DATA)

    print("=" * 90)
    print("TinyPerson evaluation for models trained on aerial_person_smallscale_1280")
    print(f"Data YAML: {TINYPERSON_DATA}")
    print(f"Split: {split}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 90)

    results = []

    for model_name, weights_path in RUNS.items():
        print("\n" + "=" * 90)
        print(f"Evaluating {model_name}")
        print(f"Weights: {weights_path}")
        print("=" * 90)

        if not weights_path.exists():
            print(f"[SKIP] Missing weights: {weights_path}")
            continue

        model = YOLO(str(weights_path))

        metrics = model.val(
            data=str(TINYPERSON_DATA),
            split=split,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            project=str(VAL_PROJECT),
            name=f"{model_name}_TinyPerson",
            exist_ok=True,
            plots=True,
            verbose=True,
        )

        m = extract_metrics(metrics)

        row = {
            "model": model_name,
            "training_dataset": "aerial_person_smallscale_1280",
            "test_set": "TinyPerson",
            "split": split,
            "weights": str(weights_path),
            "precision": m["precision"],
            "recall": m["recall"],
            "mAP50": m["mAP50"],
            "mAP50_95": m["mAP50_95"],
            "imgsz": IMG_SIZE,
            "batch": BATCH,
            "device": DEVICE,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        results.append(row)

        print(f"\n[RESULT] {model_name}")
        print(f"Precision : {row['precision']}")
        print(f"Recall    : {row['recall']}")
        print(f"mAP@50    : {row['mAP50']}")
        print(f"mAP@50:95 : {row['mAP50_95']}")

    if not results:
        print("No results generated. Check model paths.")
        return

    csv_path = OUTPUT_DIR / "tinyperson_results_smallscale_models.csv"
    json_path = OUTPUT_DIR / "tinyperson_results_smallscale_models.json"
    latex_path = OUTPUT_DIR / "tinyperson_results_smallscale_models_latex_rows.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(latex_path, "w") as f:
        f.write("% LaTeX rows: TinyPerson, models trained on aerial_person_smallscale_1280\n")
        f.write("% Model & Training dataset & Precision & Recall & mAP@50 & mAP@50:95 \\\\\n")
        for r in results:
            f.write(
                f"{r['model']} & "
                f"\\textbf{{aerial\\_person\\_smallscale\\_1280}} & "
                f"{r['precision']:.3f} & "
                f"{r['recall']:.3f} & "
                f"{r['mAP50']:.3f} & "
                f"{r['mAP50_95']:.3f} \\\\\n"
            )

    print("\n" + "=" * 90)
    print("Saved:")
    print(csv_path)
    print(json_path)
    print(latex_path)
    print("=" * 90)


if __name__ == "__main__":
    main()
