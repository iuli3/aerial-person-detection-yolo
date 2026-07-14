#!/usr/bin/env python3
"""
Extract ALL metrics for merged_final: YOLOv12m vs YOLOv26m
"""

import pandas as pd
import os

# Paths
yolo12m_csv = "runs/yolo12m_merged_final_640_200ep/results.csv"
yolo26m_csv = "runs/merged_final_640/yolo26m_merged_final_640_200ep/results.csv"

output_dir = "data/plots"
os.makedirs(output_dir, exist_ok=True)

# Load data
print("[TABLE] Loading merged_final results...")
df_12m = pd.read_csv(yolo12m_csv)
df_26m = pd.read_csv(yolo26m_csv)

# Clean column names
df_12m.columns = df_12m.columns.str.strip()
df_26m.columns = df_26m.columns.str.strip()

# Find best epoch (by mAP50-95)
best_12m = df_12m.loc[df_12m['metrics/mAP50-95(B)'].idxmax()]
best_26m = df_26m.loc[df_26m['metrics/mAP50-95(B)'].idxmax()]

# Create table with ALL metrics
table_data = {
    'Model': ['YOLOv12m', 'YOLOv26m'],
    'Epoch': [int(best_12m['epoch']), int(best_26m['epoch'])],
    'Time (s)': [f"{best_12m['time']:.1f}", f"{best_26m['time']:.1f}"],

    # Training losses
    'train/box_loss': [f"{best_12m['train/box_loss']:.4f}", f"{best_26m['train/box_loss']:.4f}"],
    'train/cls_loss': [f"{best_12m['train/cls_loss']:.4f}", f"{best_26m['train/cls_loss']:.4f}"],
    'train/dfl_loss': [f"{best_12m['train/dfl_loss']:.4f}", f"{best_26m['train/dfl_loss']:.4f}"],

    # Validation metrics
    'Precision': [f"{best_12m['metrics/precision(B)']:.4f}", f"{best_26m['metrics/precision(B)']:.4f}"],
    'Recall': [f"{best_12m['metrics/recall(B)']:.4f}", f"{best_26m['metrics/recall(B)']:.4f}"],
    'mAP@50': [f"{best_12m['metrics/mAP50(B)']:.4f}", f"{best_26m['metrics/mAP50(B)']:.4f}"],
    'mAP@50:95': [f"{best_12m['metrics/mAP50-95(B)']:.4f}", f"{best_26m['metrics/mAP50-95(B)']:.4f}"],

    # Validation losses
    'val/box_loss': [f"{best_12m['val/box_loss']:.4f}", f"{best_26m['val/box_loss']:.4f}"],
    'val/cls_loss': [f"{best_12m['val/cls_loss']:.4f}", f"{best_26m['val/cls_loss']:.4f}"],
    'val/dfl_loss': [f"{best_12m['val/dfl_loss']:.4f}", f"{best_26m['val/dfl_loss']:.4f}"],

    # Learning rates
    'lr/pg0': [f"{best_12m['lr/pg0']:.2e}", f"{best_26m['lr/pg0']:.2e}"],
    'lr/pg1': [f"{best_12m['lr/pg1']:.2e}", f"{best_26m['lr/pg1']:.2e}"],
    'lr/pg2': [f"{best_12m['lr/pg2']:.2e}", f"{best_26m['lr/pg2']:.2e}"],
}

df_table = pd.DataFrame(table_data)

# Display
print("\n" + "="*150)
print("MERGED FINAL 640 - ALL METRICS (BEST EPOCH)")
print("="*150)
print(df_table.to_string(index=False))
print("="*150 + "\n")

# Save as CSV
csv_path = os.path.join(output_dir, "merged_final_all_metrics.csv")
df_table.to_csv(csv_path, index=False)
print(f"[TABLE] Saved to: {csv_path}")

# Save as markdown
md_path = os.path.join(output_dir, "merged_final_all_metrics.md")
with open(md_path, 'w') as f:
    f.write("# Merged Final 640 - All Metrics (Best Epoch)\n\n")
    f.write("| " + " | ".join(df_table.columns) + " |\n")
    f.write("|" + "|".join(["---"] * len(df_table.columns)) + "|\n")
    for _, row in df_table.iterrows():
        f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")
    f.write("\n")
print(f"[TABLE] Saved to: {md_path}")
