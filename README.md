# Aerial Person Detection with YOLO

This repository contains the scripts used to prepare aerial and high-mounted person-detection datasets, train YOLO-based detectors, and evaluate the resulting models across standard, merged, and balanced test sets.

The code is organized as reproducibility material for research experiments. Dataset files, trained weights, and large experiment outputs are not included.

## Repository Structure

```text
configs/
  datasets/                 # YOLO dataset YAML files
  training_args/            # Training argument snapshots by dataset/model

data_preparation/
  aerial_person_smallscale/ # Small-scale aerial person dataset builders
  merged_final/             # Final merged dataset builders
  merged_1280_balanced/     # Balanced 1280px merged dataset builders
  quality_checks/           # Dataset/subset validation scripts
  suas_mannequin/           # SUAS duplicate and near-duplicate checks
  visdrone/                 # VisDrone VID-to-YOLO conversion scripts

training/
  aerial_person_smallscale_1280/
  merged_final/
  merged_1280_balanced/

evaluation/
  aerial_person_smallscale_1280/
  merged_final/
  merged_1280_balanced/
  standard_benchmarks/
  statistics/

docs/
  SCRIPT_INVENTORY.md
  DATASET_SOURCES.md
  CITATIONS.bib
```

## Datasets

The scripts reference several dataset sources used in the experiments:

- VisDrone DET
- VisDrone VID
- HDTopView / Stanford Drone-derived top-view person data
- SUAS Mannequin
- TinyPerson / small-person benchmark data

The repository does not redistribute images or annotations from third-party datasets. Before sharing any dataset copy, check the license and usage terms of the original provider.

Known dataset sources and citation entries are documented in:

```text
docs/DATASET_SOURCES.md
docs/CITATIONS.bib
```

## Constructed Datasets

Three derived YOLO-format datasets are built by the scripts in this repository.

| Dataset | Purpose | Images | Person instances | Notes |
| --- | --- | ---: | ---: | --- |
| `merged_final` | Initial 640x640 experiments | 30,073 | 358,657 | Large and diverse dataset, but less balanced across sources. |
| `merged_1280_balanced` | Main 1280x1280 comparison dataset | 14,042 | 169,195 | More controlled source distribution across VisDrone, Stanford/HDTopView, and SUAS. |
| `aerial_person_smallscale_1280` | Small-person-focused dataset | 14,115 | 227,863 | Includes the Small Object Aerial Person Detection Dataset and emphasizes small-scale people. |

### Dataset Splits

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `merged_final` | 24,427 images / 275,722 instances | 3,429 images / 54,423 instances | 2,217 images / 28,512 instances |
| `merged_1280_balanced` | 11,500 images / 134,467 instances | 1,248 images / 19,614 instances | 1,294 images / 15,114 instances |
| `aerial_person_smallscale_1280` | 10,663 images / 162,331 instances | 1,321 images / 27,718 instances | 2,131 images / 37,814 instances |

### Average Density

| Dataset | Person instances per image |
| --- | ---: |
| `merged_final` | 11.93 |
| `merged_1280_balanced` | 12.05 |
| `aerial_person_smallscale_1280` | 16.14 |

### Dataset Roles

`merged_final` is the largest dataset by image count and was used for the initial experiments. It exposes the models to diverse aerial and high-mounted viewpoints, but its source distribution is less uniform, so it is not the best option for controlled model comparisons.

`merged_1280_balanced` is the main reference dataset for high-resolution experiments. It contains fewer images than `merged_final`, but its source distribution is more controlled, which makes comparisons between models more consistent.

`aerial_person_smallscale_1280` is the most relevant dataset for the small-person detection objective. It has a similar number of images to `merged_1280_balanced`, but contains more annotated person instances, making the scenes denser and more challenging.

## Path Convention

Machine-specific paths were removed. The scripts use portable placeholders:

- `data/...` for datasets and dataset YAML files
- `weights/...` for model weights
- `runs/...` for training runs
- `experiments/...` for evaluation outputs and experiment artifacts

Update these paths to match your local workspace before running the scripts.

## Data Preparation

Data preparation scripts are grouped by the dataset or derived dataset they contribute to.

Examples:

```bash
python data_preparation/visdrone/convert_visdrone_vid_train_to_yolo.py
python data_preparation/merged_final/build_merged_final_dataset.py
python data_preparation/merged_1280_balanced/build_merged_1280_balanced_dataset.py
python data_preparation/aerial_person_smallscale/build_aerial_person_smallscale_1280.py
```

Quality checks:

```bash
python data_preparation/quality_checks/check_scale_density_subset_counts.py
python data_preparation/suas_mannequin/check_suas_duplicate_images.py
python data_preparation/suas_mannequin/check_suas_near_duplicate_images.py
```

## Training

Training scripts are organized by dataset. Each dataset folder contains one script for each YOLO model family:

```text
training/<dataset>/
  train_yolov11m.py
  train_yolov12m.py
  train_yolov26m.py
```

Examples:

```bash
python training/merged_final/train_yolov11m.py
python training/merged_final/train_yolov12m.py
python training/merged_final/train_yolov26m.py

python training/merged_1280_balanced/train_yolov11m.py
python training/aerial_person_smallscale_1280/train_yolov26m.py
```

Training argument snapshots are available under:

```text
configs/training_args/
```

## Model Sources

The experiments use medium-scale YOLO detection models from the Ultralytics model family:

| Model family | Medium model used | Official documentation |
| --- | --- | --- |
| YOLO11 | `yolo11m.pt` | <https://docs.ultralytics.com/models/yolo11/> |
| YOLO12 | `yolo12m.pt` | <https://docs.ultralytics.com/models/yolo12/> |
| YOLO26 | `yolo26m.pt` | <https://docs.ultralytics.com/models/yolo26/> |

The medium variants were selected to keep the comparison consistent across model families while maintaining a practical balance between accuracy and computational cost.

## Evaluation

Evaluation scripts are grouped by target dataset or benchmark.

Examples:

```bash
python evaluation/merged_final/evaluate_all_models.py
python evaluation/merged_final/summarize_merged_final_stats.py

python evaluation/merged_1280_balanced/run_all_evaluations.py
python evaluation/standard_benchmarks/evaluate_tinpeople.py
```

Some scripts expect trained weights to be available under `weights/`, `runs/`, or `experiments/`. Adjust the paths inside each script if your local layout differs.

## Reproducibility Notes

- Dataset split logic is encoded in the data preparation scripts.
- Fixed seeds are used where sampling or redistribution is performed.
- Training scripts specify model family, image size, epochs, optimizer, augmentation, device, and output project path.
- Dataset YAML files are available under `configs/datasets/`.
- Evaluation scripts include model/dataset paths and output locations.
