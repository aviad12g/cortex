import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

def plot_layer_migration():
    # Find the latest log directory
    log_root = Path("logs/infinite_proof_v3")
    # list dirs
    dirs = [d for d in log_root.iterdir() if d.is_dir()]
    if not dirs:
        print("No log directories found.")
        return
    
    # Sort by creation time (or name if timestamped)
    latest_dir = sorted(dirs, key=lambda x: x.name)[-1]
    print(f"Analyzing logs from: {latest_dir}")
    
    probe_file = latest_dir / "probes.jsonl"
    if not probe_file.exists():
        print(f"No probes.jsonl found in {latest_dir}")
        return

    data = []
    with open(probe_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # We only care about training steps (probe_idx 0 usually implies training sample or first probe)
                # Actually, the script logs one entry per sample.
                # We want 'fast_share_by_layer'
                if "fast_share_by_layer" in entry:
                    data.append({
                        "step": entry["global_step"],
                        "fast_share": entry["fast_share_by_layer"]
                    })
            except json.JSONDecodeError:
                continue
    
    if not data:
        print("No valid data found.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values("step")
    
    # Create matrix: Rows = Layers, Cols = Steps
    # fast_share is a list of floats per step
    matrix = np.array(df["fast_share"].tolist()).T # Transpose so layers are rows
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap="viridis", cbar_kws={'label': 'Fast Weight Contribution'})
    plt.title("Layer Migration: Fast Weight Contribution over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Layer Depth")
    plt.gca().invert_yaxis() # Layer 0 at bottom
    
    output_path = "layer_migration.png"
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    plot_layer_migration()
