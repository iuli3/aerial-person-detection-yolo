#!/usr/bin/env python3
"""
Plot training/convergence comparison:
YOLOv11m vs YOLOv12m vs YOLOv26m on aerial_person_smallscale_1280
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ===================== PATHS =====================

yolo11m_csv = "data/aerial_person_smallscale_1280/runs/yolo11m_aerial_person_smallscale_1280/results.csv"

yolo12m_csv = "data/aerial_person_smallscale_1280/runs/yolo12m_aerial_person_smallscale_1280/results.csv"

yolo26m_csv = "data/aerial_person_smallscale_1280/yolo26_results.csv"

output_dir = "data/plots"
os.makedirs(output_dir, exist_ok=True)

output_png = os.path.join(output_dir, "aerial_person_smallscale_1280_convergence.png")
output_pdf = os.path.join(output_dir, "aerial_person_smallscale_1280_convergence.pdf")


# ===================== SETTINGS =====================

EPOCH_LIMIT = 50

colors = {
    "YOLOv11m": "#1f77b4",
    "YOLOv12m": "#ff7f0e",
    "YOLOv26m": "#2ca02c",
}

metrics = [
    {
        "column": "metrics/mAP50(B)",
        "title": "Validation mAP@50",
        "ylabel": "Metric value",
        "ylim": (0.35, 0.78),
    },
    {
        "column": "metrics/mAP50-95(B)",
        "title": "Validation mAP@50:95",
        "ylabel": "Metric value",
        "ylim": (0.10, 0.38),
    },
    {
        "column": "val/box_loss",
        "title": "Validation Box Loss",
        "ylabel": "Loss",
        "ylim": None,
    },
]


# ===================== FUNCTIONS =====================

def load_results(csv_path, model_name):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Missing CSV for {model_name}: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if EPOCH_LIMIT is not None:
        df = df.iloc[:EPOCH_LIMIT].copy()

    return df


def get_epochs(df):
    if "epoch" in df.columns:
        return df["epoch"].astype(int)
    return range(1, len(df) + 1)


def check_columns(df, model_name):
    required = [m["column"] for m in metrics]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"\n[ERROR] Missing columns in {model_name}:")
        for col in missing:
            print(f"  - {col}")

        print("\n[INFO] Available columns are:")
        for col in df.columns:
            print(f"  - {col}")

        raise KeyError(f"Missing required columns for {model_name}")


# ===================== LOAD DATA =====================

print("[PLOT] Loading training results...")

dfs = {
    "YOLOv11m": load_results(yolo11m_csv, "YOLOv11m"),
    "YOLOv12m": load_results(yolo12m_csv, "YOLOv12m"),
    "YOLOv26m": load_results(yolo26m_csv, "YOLOv26m"),
}

for model_name, df in dfs.items():
    check_columns(df, model_name)
    print(f"[PLOT] {model_name}: {len(df)} epochs loaded")


# ===================== PLOT STYLE =====================

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


# ===================== CREATE FIGURE =====================

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.1))

legend_handles = []
legend_labels = []

for ax, metric in zip(axes, metrics):
    for model_name, df in dfs.items():
        epochs = get_epochs(df)

        line, = ax.plot(
            epochs,
            df[metric["column"]],
            linewidth=1.3,
            color=colors[model_name],
            label=model_name,
        )

        if metric == metrics[0]:
            legend_handles.append(line)
            legend_labels.append(model_name)

    ax.set_title(metric["title"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric["ylabel"])
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.25)

    if metric["ylim"] is not None:
        ax.set_ylim(metric["ylim"])

    ax.set_xlim(1, EPOCH_LIMIT if EPOCH_LIMIT is not None else max(len(df) for df in dfs.values()))

# Shared legend at the top.
fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, 1.03),
    fontsize=7,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig(output_png, bbox_inches="tight")
fig.savefig(output_pdf, bbox_inches="tight")

plt.close(fig)

print(f"[PLOT] PNG saved: {output_png}")
print(f"[PLOT] PDF saved: {output_pdf}")
print("[PLOT] Done!")
