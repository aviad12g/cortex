import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True, help="Path to probes.jsonl")
    parser.add_argument("--output", type=str, default="cortex_scaling.png")
    args = parser.parse_args()

    data = []
    print(f"Reading {args.log_file}...")
    with open(args.log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue

    df = pd.DataFrame(data)
    
    # Filter for relevant columns
    # We want to see Loss over Time (Global Step) separated by Gap
    plot_df = df[['global_step', 'gap', 'loss_tok']].copy()
    
    # Calculate rolling average to smooth the noise
    plot_df['loss_smooth'] = plot_df.groupby('gap')['loss_tok'].transform(lambda x: x.rolling(50, 1).mean())

    # Set style
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 8))

    # Create the Line Plot
    sns.lineplot(
        data=plot_df,
        x="global_step",
        y="loss_smooth",
        hue="gap",
        palette="viridis",
        linewidth=2.5
    )

    # Annotations
    plt.title("Cortex v2: Inverse Scaling in Infinite Context", fontsize=20, pad=20)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Cross-Entropy Loss (Lower is Better)", fontsize=14)
    plt.legend(title="Context Gap (Tokens)", title_fontsize=12, fontsize=12)
    
    # Add a benchmark line for Random Guessing
    # Vocab ~150k -> ln(150000) approx 11.9
    plt.axhline(y=11.9, color='r', linestyle='--', alpha=0.5, label='Random Guessing')

    # Save
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()

