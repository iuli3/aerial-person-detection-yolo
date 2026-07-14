# Script Inventory

This file summarizes the public scripts included for reproducibility.

| Public script | Purpose |
| --- | --- |
| `convert_visdrone_vid_train_to_yolo.py` | Converts VisDrone VID train annotations to YOLO format. |
| `convert_visdrone_vid_val_to_yolo.py` | Converts VisDrone VID validation annotations to YOLO format. |
| `build_merged_final_dataset.py` | Builds the final merged dataset from VisDrone DET, VisDrone VID, HDTopView, and SUAS. |
| `build_merged_1280_balanced_dataset.py` | Builds the balanced 1280px merged dataset. |
| `build_aerial_person_smallscale_1280.py` | Builds the small-scale aerial person dataset. |
| `create_scale_density_evaluation_subsets.py` | Creates scale and density evaluation subsets. |
| `check_scale_density_subset_counts.py` | Checks subset counts and source distributions. |

Training scripts are organized by dataset under `training/`, with YOLOv11, YOLOv12, and YOLOv26 variants in each dataset folder.
