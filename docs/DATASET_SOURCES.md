# Dataset Sources

This file documents the public sources used to build the datasets referenced by the preparation, training, and evaluation scripts. Image data is not redistributed in this repository.

## VisDrone

- Repository name used in this project: `VisDrone DET` and `VisDrone VID`.
- Reference publication: Zhu et al., "Detection and tracking meet drones challenge", IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.
- Usage in this project: used as an aerial drone-view person-detection source and as a source for VID frame conversion to YOLO format.
- Redistribution status: images are not included in this repository; users should obtain the dataset from the official source and follow the provider terms.

## HDTopView / Stanford Drone Subset

- Repository name used in this project: `HDTopView` / Stanford Drone subset.
- Source dataset name: `stanford_deathCircle Dataset`.
- Provider: Roboflow Universe.
- Author listed by source: Deneme.
- Publisher: Roboflow.
- Source URL: <https://universe.roboflow.com/deneme-eo7pz/stanford_deathcircle>
- Source year/month: May 2025.
- Access date: 2026-07-14.
- Usage in this project: used as a high-mounted/top-view person-detection source in the merged datasets.
- Redistribution status: images are not included in this repository; users should download the dataset from the source and follow the provider terms.

## SUAS Mannequin

- Repository name used in this project: `SUAS Mannequin`.
- Source dataset name: `suas-mannequin-detection Dataset`.
- Provider: Roboflow Universe.
- Author listed by source: Alex Eagles 5 M.
- Publisher: Roboflow.
- Source URL: <https://universe.roboflow.com/alex-eagles-5-m/suas-mannequin-detection>
- Source year/month: October 2023.
- Access date: 2026-07-14.
- Usage in this project: used as an aerial/drone-view mannequin detection source in merged training and evaluation datasets.
- Redistribution status: images are not included in this repository; users should download the dataset from the source and follow the provider terms.

## Small Object Aerial Person Detection

- Repository name used in this project: `aerial_person_smallscale_1280` / small-scale aerial person data.
- Source dataset name: `Small Object Aerial Person Detection Dataset`.
- Authors: Rafael Makrigiorgis, Christos Kyrkou, and Panayiotis Kolios.
- Version: 1.0.
- Publisher: Zenodo.
- DOI: <https://doi.org/10.5281/zenodo.7740081>
- Source year: 2023.
- Usage in this project: used as a small-object aerial person-detection source in the aerial person small-scale dataset construction.
- Redistribution status: images are not included in this repository; users should download the dataset from Zenodo and follow the dataset terms.

## Notes

Before publishing data or trained weights, verify the license and usage terms of each source dataset. If redistribution is not explicitly allowed, publish only code, configuration files, aggregate statistics, and instructions for reproducing the dataset locally.
