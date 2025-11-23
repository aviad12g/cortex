"""
Interactive Inference Demo for Cortex v2.
Loads a checkpoint and allows manual probing of the memory mechanism.
"""

import argparse
import torch
from transformers import AutoTokenizer
from base.hf_wrap import load_qwen_with_cortex, CortexWrapConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the epoch_1.pt file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--fast_rank", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[Demo] Loading Base Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Match the training config
    config = CortexWrapConfig(
        rank_fast=args.fast_rank,
        alpha_max=0.05, # Legacy param, but we used bias hack
        use_hybrid=True
    )
    
    print(f"[Demo] Injecting Cortex Sidecar (Rank {args.fast_rank})...")
    model = load_qwen_with_cortex(args.model, cortex_cfg=config)
    
    print(f"[Demo] Loading Weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Filter for cortex keys only to be safe, though load_state_dict handles strict=False
    # Actually we saved the whole model state dict in stage_a1
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Demo] Weights Loaded. Missing keys (expected base model keys): {len(missing)}")
    
    model.to(device)
    model.eval()
    
    print("\n" + "="*50)
    print(" CORTEX v2 MEMORY PROBE")
    print("="*50)
    print("Type a KV prompt (e.g., 'Key: 8F3A Value: 9999 ... Query: 8F3A')")
    print("Or type 'exit' to quit.")
    
    while True:
        user_input = input("\nPrompt > ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        if not user_input.strip():
            continue
            
        # Prepare Input
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        
        # Generate
        print("Thinking...", end="", flush=True)
        with torch.no_grad():
            # We use generate with session management implicitly handled by the wrapper if needed,
            # but here we just want standard generation. 
            # Note: The wrapper expects 'session_id' for state persistence. 
            # For a single prompt, we can let it reset or pass a dummy session.
            
            # Important: We must pass use_cache=False to force the model to use the recurrent state 
            # if we want to test 'infinite' context, but for short debug prompts, standard generation is fine.
            # However, HF generate() relies on KV cache. 
            # Cortex is compatible with KV cache for the base model part.
            
            generated_ids = model.base.generate(
                **inputs, 
                max_new_tokens=10, 
                do_sample=False, # Greedy for deterministic debugging
                pad_token_id=tokenizer.eos_token_id,
                session_id="demo_probe", # Activate Cortex
                reset_session=True
            )
            
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = output_text[len(user_input):].strip()
        
        print(f"\rResponse > [{response}]")
        print("-" * 30)
        print(f"Full Output: {output_text}")

if __name__ == "__main__":
    main()

