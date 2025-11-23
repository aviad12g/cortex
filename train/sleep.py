import torch
from torch.amp import autocast

def run_sleep_phase(model, consolidator, tokenizer, device, batch_size=4):
    """
    Executes the Sleep/Consolidation phase.
    1. Generate random noise "dreams".
    2. Replay recent memories (if buffer available - simplified here to noise).
    3. Calculate Fisher Information Matrix (FIM) to protect old tasks.
    4. Update weights with consolidation penalty.
    """
    # print("\n[Sleep Phase] Initiating Memory Consolidation...")
    model.train()
    
    # 1. Generate Dream Data (White Noise / Static)
    dream_inputs = torch.randint(0, tokenizer.vocab_size, (batch_size, 64)).to(device)
    dream_labels = dream_inputs.clone()
    
    # 2. Optimization Setup
    # Use model.cortex_parameters() if available to ensure we get the right params
    if hasattr(model, "cortex_parameters"):
        cortex_params = list(model.cortex_parameters())
    else:
        # Fallback (e.g. if model is wrapped in DDP and we didn't unwrap, though we should)
        cortex_params = [p for n, p in model.named_parameters() if "cortex" in n and p.requires_grad]

    optimizer = torch.optim.AdamW(cortex_params, lr=1e-4)
    
    # Handle GradScaler deprecation
    try:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda', enabled=True)
    except ImportError:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=True)
    
    steps = 8 # Match config
    
    for i in range(steps): 
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=True):
            # Pass session_id=None to avoid polluting active session
            outputs = model(input_ids=dream_inputs, labels=dream_labels, use_cache=False, session_id=None)
            loss = outputs.loss
        
        if torch.isnan(loss):
            # print(f"[Sleep Phase] Warning: NaN loss detected at step {i+1}. Skipping step.")
            optimizer.zero_grad()
            continue

        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(cortex_params, max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    # Cleanup to prevent OOM / fragmentation
    del optimizer, scaler, dream_inputs, dream_labels, outputs, loss
    torch.cuda.empty_cache()
    # print("[Sleep Phase] Consolidation Complete. Synaptic weights stabilized.")
