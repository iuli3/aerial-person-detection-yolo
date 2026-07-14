#!/usr/bin/env python3
"""
Extract final results for YOLOv11m, YOLOv12m, YOLOv26m on aerial_person_smallscale_1280
Create a comparison table.
"""

import pandas as pd
import os

# Paths
yolo11m_csv = "data/aerial_person_smallscale_1280/runs/yolo11m_aerial_person_smallscale_1280/results.csv"
yolo12m_csv = "data/aerial_person_smallscale_1280/runs/yolo12m_aerial_person_smallscale_1280/results.csv"
yolo26m_csv = "data/aerial_person_smallscale_1280/yolo26_results.csv"

output_dir = "data/plots"
os.makedirs(output_dir, exist_ok=True)

# Load data
print("[TABLE] Loading results...")
df_11m = pd.read_csv(yolo11m_csv)
df_12m = pd.read_csv(yolo12m_csv)
df_26m = pd.read_csv(yolo26m_csv)

# Clean column names
df_11m.columns = df_11m.columns.str.strip()
df_12m.columns = df_12m.columns.str.strip()
df_26m.columns = df_26m.columns.str.strip()

# Find best epoch (by mAP50-95)
best_11m = df_11m.loc[df_11m['metrics/mAP50-95(B)'].idxmax()]
best_12m = df_12m.loc[df_12m['metrics/mAP50-95(B)'].idxmax()]
best_26m = df_26m.loc[df_26m['metrics/mAP50-95(B)'].idxmax()]

# Create table
table_data = {
    'Model': ['YOLOv11m', 'YOLOv12m', 'YOLOv26m'],
    'Epoch': [int(best_11m['epoch']), int(best_12m['epoch']), int(best_26m['epoch'])],
    'Precision': [f"{best_11m['metrics/precision(B)']:.4f}", f"{best_12m['metrics/precision(B)']:.4f}", f"{best_26m['metrics/precision(B)']:.4f}"],
    'Recall': [f"{best_11m['metrics/recall(B)']:.4f}", f"{best_12m['metrics/recall(B)']:.4f}", f"{best_26m['metrics/recall(B)']:.4f}"],
    'mAP@50': [f"{best_11m['metrics/mAP50(B)']:.4f}", f"{best_12m['metrics/mAP50(B)']:.4f}", f"{best_26m['metrics/mAP50(B)']:.4f}"],
    'mAP@50:95': [f"{best_11m['metrics/mAP50-95(B)']:.4f}", f"{best_12m['metrics/mAP50-95(B)']:.4f}", f"{best_26m['metrics/mAP50-95(B)']:.4f}"],
    'Box Loss': [f"{best_11m['val/box_loss']:.4f}", f"{best_12m['val/box_loss']:.4f}", f"{best_26m['val/box_loss']:.4f}"],
}

df_table = pd.DataFrame(table_data)

# Display
print("\n" + "="*90)
print("AERIAL PERSON SMALLSCALE 1280 - BEST RESULTS")
print("="*90)
print(df_table.to_string(index=False))
print("="*90 + "\n")

# Save as CSV
csv_path = os.path.join(output_dir, "aerial_person_1280_results_table.csv")
df_table.to_csv(csv_path, index=False)
print(f"[TABLE] Saved to: {csv_path}")

# Save as markdown (manual)
md_path = os.path.join(output_dir, "aerial_person_1280_results_table.md")
with open(md_path, 'w') as f:
    f.write("# Aerial Person Smallscale 1280 - Results Comparison\n\n")
    f.write("| " + " | ".join(df_table.columns) + " |\n")
    f.write("|" + "|".join(["---"] * len(df_table.columns)) + "|\n")
    for _, row in df_table.iterrows():
        f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")
    f.write("\n")
print(f"[TABLE] Saved to: {md_path}")
