import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze():
    # Hardcoded path to the specific run
    log_path = Path("logs/infinite_proof_v3/a1_kv_Qwen1.5-0_cortex_20251124_174053/probes.jsonl")
    
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    df = pd.DataFrame(data)
    
    print(f"--- RUN CONFIGURATION ---")
    print(f"Total Samples: {len(df)}")
    print(f"Gap (Context): {df['gap'].iloc[0]}")
    print(f"Task: {df['task'].iloc[0]}")
    print(f"Run ID: {df['run_id'].iloc[0]}")
    
    print(f"\n--- LOSS STATS (loss_tok) ---")
    print(f"Start Loss (Step 1): {df['loss_tok'].iloc[0]:.4f}")
    print(f"End Loss (Step {len(df)}): {df['loss_tok'].iloc[-1]:.4f}")
    print(f"Min Loss: {df['loss_tok'].min():.4f}")
    print(f"Max Loss: {df['loss_tok'].max():.4f}")
    print(f"Mean Loss: {df['loss_tok'].mean():.4f}")
    
    print(f"\n--- MEMORY MECHANICS ---")
    print(f"Alpha Head Mean (Retention): {df['alpha_head_mean'].mean():.4f} (Max: {df['alpha_head_mean'].max():.4f})")
    print(f"Fast Share Mean (Contribution): {df['fast_share_mean'].mean():.4f}")
    print(f"Mixing Gate Mean (1=Attn, 0=Mem): {df['m_gate_mean'].mean():.4f}")
    
    # Check for "Miracle Loss" (1.06 is the baseline for random guessing on this vocab/task subset)
    baseline = 1.06
    below_baseline = df[df['loss_tok'] < baseline]
    print(f"\n--- PROOF OF LEARNING ---")
    print(f"Samples beating baseline ({baseline}): {len(below_baseline)} / {len(df)}")
    if len(below_baseline) > 0:
        print(f"First beat baseline at Step: {below_baseline['global_step'].iloc[0]}")

if __name__ == "__main__":
    analyze()
