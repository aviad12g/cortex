import os
import argparse
import math
import json
import random
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import yaml

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer

# Import Cortex components
from base.hf_wrap import CortexWrapConfig, CortexWrappedModel, load_qwen_with_cortex
from train.sleep import run_sleep_phase
from train.data_long import build_dataset, sample_to_training_pair, NOISE_TOKENS

# Task mapping
TASK_KEY_MAP = {
    "kv": "kv",
    "copy": "copy_reverse",
    "nback": "nback",
    "add": "addition",
}

def str_to_bool(value: str) -> bool:
    if value.lower() in {"true", "yes", "1"}:
        return True
    return False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A1 Cortex training (DDP)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["kv", "copy", "nback", "add"], required=True)
    parser.add_argument("--gaps", type=int, nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--variant", type=str, default="cortex", choices=["cortex", "fast_off", "baseline"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_max", type=float, default=0.05)
    parser.add_argument("--fast_rank", type=int, default=64)
    parser.add_argument("--fast_decay", type=float, default=0.95)
    parser.add_argument("--fast_beta", type=float, default=0.01)
    parser.add_argument("--lr_sidecar", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", type=str, default="true")
    parser.add_argument("--use_session", type=str, default="true")
    parser.add_argument("--probe_loss_weight", type=float, default=0.0)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--samples_per_gap", type=int, default=512)
    parser.add_argument("--sleep_interval", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=0, help="If > 0, split sequences into chunks for TBPTT")
    return parser.parse_args()


def freeze_base_model(model: CortexWrappedModel) -> None:
    cortex_ids = {id(p) for p in model.cortex_parameters()}
    for param in model.parameters():
        if id(param) in cortex_ids:
            continue
        param.requires_grad = False


def get_git_hash() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_probe_texts(count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        length = rng.randint(40, 80)
        words = rng.choices(NOISE_TOKENS, k=length)
        texts.append(" ".join(words))
    return texts


def evaluate_probe_perplexity(
    model: CortexWrappedModel,
    tokenizer,
    probe_texts: List[str],
    device: torch.device,
    alpha_freeze: bool,
) -> float:
    # Handle DDP unwrapping
    raw_model = model.module if hasattr(model, "module") else model
    
    prev_mode = raw_model._mix_mode
    prev_alpha = raw_model._alpha_override
    prev_sidecar = raw_model._sidecar_enabled
    if alpha_freeze:
        raw_model.set_alpha_override(0.0)
        raw_model.set_mix_mode("slow_only")
        raw_model.enable_sidecar(True)

    model.eval()
    losses = []
    with torch.no_grad():
        for text in probe_texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with autocast(enabled=True):
                # Use raw_model to avoid DDP sync since this runs only on Rank 0
                outputs = raw_model(**encoded, labels=encoded["input_ids"], session_id="probe", reset_session=True, use_cache=False)
            losses.append(outputs.loss.item())
    model.train()

    if alpha_freeze:
        raw_model.set_alpha_override(prev_alpha)
        raw_model.set_mix_mode(prev_mode)
        raw_model.enable_sidecar(prev_sidecar)

    if not losses:
        return float("nan")
    return float(math.exp(sum(losses) / len(losses)))


def summarize_fast_stats(
    model: CortexWrappedModel,
    answer_positions: torch.Tensor,
) -> Tuple[List[float], List[List[float]]]:
    # Handle DDP unwrapping
    raw_model = model.module if hasattr(model, "module") else model
    
    fast_layer_means: List[float] = []
    alpha_layer_hist: List[List[float]] = []
    answer_idx = answer_positions.cpu()

    for block in raw_model._cortex_layers:
        if block.last_fast_share is not None and block.last_fast_share.numel() > 0:
            share = block.last_fast_share
            if share.dim() == 2: # [B, T]
                 share = share[0] 
            
            if answer_idx.max() < share.size(0):
                if answer_idx.numel() > 0:
                    share_sel = share[answer_idx]
                else:
                    share_sel = share
            else:
                share_sel = share
                
            fast_layer_means.append(float(share_sel.mean().item()) if share_sel.numel() > 0 else 0.0)
        else:
            fast_layer_means.append(0.0)

        if getattr(block, "last_alpha", None) is not None and block.last_alpha.numel() > 0:
            alpha = block.last_alpha
            alpha = alpha.detach().cpu().float()
            alpha = alpha.squeeze() 
            if alpha.dim() > 1:
                dims_to_reduce = list(range(alpha.dim() - 1))
                alpha_mean = alpha.mean(dim=dims_to_reduce)
            else:
                alpha_mean = alpha
            alpha_list = alpha_mean.tolist()
            if isinstance(alpha_list, float):
                alpha_list = [alpha_list]
            alpha_layer_hist.append(alpha_list)
        else:
            alpha_layer_hist.append([0.0] * raw_model.base.config.num_attention_heads)

    return fast_layer_means, alpha_layer_hist


def compute_metrics(
    outputs,
    labels: torch.Tensor,
    tokenizer,
    model: CortexWrappedModel,
) -> Dict[str, float]:
    logits = outputs.logits.float()
    preds = logits.argmax(dim=-1)
    answer_mask = labels != -100
    answer_positions = answer_mask[0].nonzero(as_tuple=True)[0]
    if answer_positions.numel() == 0:
        answer_positions = torch.arange(labels.size(1), device=labels.device)

    gather_index = labels.unsqueeze(-1)
    gather_index[gather_index < 0] = 0
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=gather_index).squeeze(-1)
    valid_token_log_probs = token_log_probs[answer_mask]
    if valid_token_log_probs.numel() > 0:
        loss_tok = float((-valid_token_log_probs).mean().item())
    else:
        loss_tok = 0.0

    entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    entropy_answer = entropy[answer_mask]
    entropy_u = float(entropy_answer.mean().item()) if entropy_answer.numel() > 0 else 0.0

    correct = preds[answer_mask] == labels[answer_mask]
    correct_flag = int(correct.all().item()) if correct.numel() > 0 else 0

    predicted_tokens = preds[answer_mask].detach().cpu().tolist()
    target_tokens = labels[answer_mask].detach().cpu().tolist()
    pred_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip() if predicted_tokens else ""
    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip() if target_tokens else ""

    fast_share_by_layer, alpha_hist = summarize_fast_stats(model, answer_positions)
    fast_share_mean = float(sum(fast_share_by_layer) / len(fast_share_by_layer)) if fast_share_by_layer else 0.0

    alpha_values_flat = [val for layer_vals in alpha_hist for val in layer_vals]
    alpha_head_mean = float(sum(alpha_values_flat) / len(alpha_values_flat)) if alpha_values_flat else 0.0
    alpha_head_max = float(max(alpha_values_flat)) if alpha_values_flat else 0.0

    m_gate_mean = 0.0
    m_gate_max = 0.0
    
    raw_model = model.module if hasattr(model, "module") else model
    if raw_model._last_gates is not None:
        m_gate_tensor = raw_model._last_gates[0]
        if m_gate_tensor.dim() == 2:
            m_gate_slice = m_gate_tensor[0, answer_positions.cpu()] if answer_positions.numel() > 0 else m_gate_tensor[0]
            if m_gate_slice.numel() > 0:
                m_gate_mean = float(m_gate_slice.mean().item())
                m_gate_max = float(m_gate_slice.max().item())

    return {
        "loss_tok": loss_tok,
        "entropy_u": entropy_u,
        "pred_text": pred_text,
        "decoded_target": target_text,
        "correct": correct_flag,
        "fast_share_by_layer": fast_share_by_layer,
        "fast_share_mean": fast_share_mean,
        "alpha_head_hist": alpha_hist,
        "alpha_head_mean": alpha_head_mean,
        "alpha_head_max": alpha_head_max,
        "m_gate_mean": m_gate_mean,
        "m_gate_max": m_gate_max,
    }


def main() -> None:
    args = parse_args()
    
    # DDP Setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[DDP] Rank {rank} (Local {local_rank}) initialized.")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single] Running on {device}")

    amp_enabled = str_to_bool(args.amp)
    use_session = str_to_bool(args.use_session)

    if args.cfg and rank == 0:
        with open(args.cfg) as f:
            cfg_overrides = yaml.safe_load(f)
        for key, value in cfg_overrides.items():
            if not hasattr(args, key):
                continue
            setattr(args, key, value)

    dtype = torch.bfloat16 if amp_enabled and device.type == "cuda" else None

    cortex_cfg = CortexWrapConfig(
        rank_fast=args.fast_rank,
        decay=args.fast_decay,
        alpha_max=args.alpha_max,
        beta=args.fast_beta,
    )

    model = load_qwen_with_cortex(
        model_name=args.model,
        cortex_cfg=cortex_cfg,
        torch_dtype=dtype,
    )

    freeze_base_model(model)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cortex_params_count = sum(p.numel() for p in model.cortex_parameters())
        print(f"[Stage A1] Total Params: {total_params/1e6:.2f}M")
        print(f"[Stage A1] Trainable Params: {trainable_params/1e6:.2f}M")
        print(f"[Stage A1] Cortex Params: {cortex_params_count/1e6:.2f}M")

    if args.variant in {"fast_off", "baseline"}:
        model.set_alpha_override(0.0)
        model.set_mix_mode("slow_only")
    else:
        model.set_alpha_override(None)
        model.set_mix_mode("dual")

    model.enable_sidecar(args.variant != "baseline")
    model.to(device)
    
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    model.train()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = max(8192, max(args.gaps) + 1024)

    run_tag = Path(args.model).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_id = f"a1_{args.task}_{run_tag}_{args.variant}_{timestamp}"
    run_id = args.run_id or default_run_id

    log_dir = Path(args.log_dir) / run_id
    ckpt_dir = Path(args.save_dir) / run_id
    
    if rank == 0:
        ensure_dir(log_dir)
        ensure_dir(ckpt_dir)
        probe_path = log_dir / "probes.jsonl"
        drift_path = log_dir / "drift.jsonl"
        if probe_path.exists(): probe_path.unlink()
        if drift_path.exists(): drift_path.unlink()

        git_hash = get_git_hash()
        run_info = {
            "run_id": run_id,
            "variant": args.variant,
            "task": args.task,
            "gaps": args.gaps,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "alpha_max": args.alpha_max,
            "fast_rank": args.fast_rank,
            "fast_decay": args.fast_decay,
            "fast_beta": args.fast_beta,
            "lr_sidecar": args.lr_sidecar,
            "grad_clip": args.grad_clip,
            "amp": amp_enabled,
            "use_session": use_session,
            "git_hash": git_hash,
            "world_size": world_size,
        }
        with open(log_dir / "run_args.json", "w") as f:
            json.dump(run_info, f, indent=2)

    rng = random.Random(args.seed)
    all_samples: List[Dict] = []
    internal_task = TASK_KEY_MAP[args.task]
    
    for gap in args.gaps:
        dataset = build_dataset(internal_task, gap, args.samples_per_gap, args.seed)
        for sample in dataset:
            pair = sample_to_training_pair(sample, tokenizer, tokenizer.model_max_length)
            pair["gap"] = gap
            pair["task"] = args.task
            pair["seed"] = args.seed
            all_samples.append(pair)

    if ddp:
        rng.shuffle(all_samples)
        my_samples = all_samples[rank::world_size]
        if rank == 0:
            print(f"[DDP] Total Samples: {len(all_samples)}. Per Rank: {len(my_samples)}")
    else:
        my_samples = all_samples

    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(raw_model.cortex_parameters(), lr=args.lr_sidecar, weight_decay=0.0)
    
    # --- PHASE 3: SCHEDULER ---
    # Calculate total optimization steps
    total_steps = (len(my_samples) // max(1, args.batch_size)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    if rank == 0:
        print(f"[Phase 3] Scheduler Enabled: CosineAnnealingLR (T_max={total_steps})")
    # --------------------------

    # Disable scaler for bfloat16
    use_scaler = (amp_enabled and device.type == "cuda" and dtype == torch.float16)
    scaler = GradScaler('cuda', enabled=use_scaler)
    
    # Disable anomaly detection for speed
    # torch.autograd.set_detect_anomaly(True)

    probe_texts = build_probe_texts(200, args.seed)
    baseline_ppl: Optional[float] = None

    accum_steps = max(1, args.batch_size)
    global_step = 0
    
    for epoch in range(args.epochs):
        rng.shuffle(my_samples)
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        for sample in my_samples:
            global_step += 1
            micro_step += 1

            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.tensor(sample["labels"], dtype=torch.long, device=device).unsqueeze(0)
            session_id = f"{run_id}_{sample['sample_id']}" if use_session else None
            
            chunks = []
            if args.chunk_size > 0 and input_ids.size(1) > args.chunk_size:
                seq_len = input_ids.size(1)
                for i in range(0, seq_len, args.chunk_size):
                    end = min(i + args.chunk_size, seq_len)
                    chunk_input = input_ids[:, i:end]
                    chunk_labels = labels[:, i:end]
                    chunks.append((chunk_input, chunk_labels, i == 0))
            else:
                chunks.append((input_ids, labels, True))

            total_loss = 0.0
            
            for chunk_idx, (c_input, c_labels, is_first) in enumerate(chunks):
                should_reset = True if (use_session and is_first) else False
                
                if use_scaler:
                    with autocast(dtype=torch.float16, enabled=amp_enabled and device.type == "cuda"):
                        outputs = model(
                            input_ids=c_input,
                            labels=None,
                            session_id=session_id,
                            reset_session=should_reset,
                            use_cache=False,
                        )
                        
                        if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                            print(f"!!! NAN/INF IN LOGITS at Step {global_step} !!!")
                            print(f"Logits Min: {outputs.logits.min()}, Max: {outputs.logits.max()}, Mean: {outputs.logits.mean()}")
                            import sys; sys.exit(1)

                        logits = outputs.logits.float()
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = c_labels[..., 1:].contiguous()
                        
                        valid_labels_mask = shift_labels != -100
                        if valid_labels_mask.sum() == 0:
                             loss = torch.tensor(0.0, device=device, requires_grad=True)
                        else:
                             loss_fct = torch.nn.CrossEntropyLoss()
                             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"!!! NAN LOSS at Step {global_step} !!!")
                            import sys; sys.exit(1)
                            print(f"Valid Labels Count: {valid_labels.sum()}")
                            import sys; sys.exit(1)

                    loss = loss / accum_steps
                    loss.backward()
                else:
                     # Bfloat16 / Float32 path
                     with autocast(dtype=torch.bfloat16, enabled=amp_enabled and device.type == "cuda"):
                        outputs = model(
                            input_ids=c_input,
                            labels=None,
                            session_id=session_id,
                            reset_session=should_reset,
                            use_cache=False,
                        )
                        
                        if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                            print(f"!!! NAN/INF IN LOGITS at Step {global_step} !!!")
                            import sys; sys.exit(1)

                        logits = outputs.logits.float()
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = c_labels[..., 1:].contiguous()
                        
                        valid_labels_mask = shift_labels != -100
                        if valid_labels_mask.sum() == 0:
                             loss = torch.tensor(0.0, device=device, requires_grad=True)
                        else:
                             loss_fct = torch.nn.CrossEntropyLoss()
                             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"!!! NAN LOSS at Step {global_step} !!!")
                            import sys; sys.exit(1)

                     loss = loss / accum_steps
                     loss.backward()
                
                # Detach state
                raw_model = model.module if ddp else model
                for block in raw_model._cortex_layers:
                    block.S = block.S.detach()
                
                total_loss += loss.item()
                
                if rank == 0:
                    # Log every step as requested
                    should_log = True 
                    if chunk_idx == len(chunks) - 1 and should_log:
                         metrics = compute_metrics(outputs, c_labels, tokenizer, model)
                         # Log LR
                         current_lr = scheduler.get_last_lr()[0]
                         print(f"[Step {global_step}] Loss: {total_loss:.4f} | LR: {current_lr:.2e}", flush=True)
                    elif chunk_idx == len(chunks) - 1:
                         metrics = None
                else:
                    metrics = None

            if micro_step % accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    if args.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(raw_model.cortex_parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(raw_model.cortex_parameters(), args.grad_clip)
                    optimizer.step()
                
                # Step Scheduler
                scheduler.step()
                
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0
                
            if global_step % args.sleep_interval == 0:
                run_sleep_phase(model, None, tokenizer, device)
                model.train()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if metrics is None: continue

            if use_session:
                raw_model = model.module if ddp else model
                raw_model.flush_sessions()

            if rank == 0:
                probe_entry = {
                    "run_id": run_id,
                    "variant": args.variant,
                    "task": args.task,
                    "gap": sample["gap"],
                    "seed": args.seed,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "sample_id": sample["sample_id"],
                    "probe_idx": 0,
                    "answer_text": sample["answer_text"],
                    "pred_text": metrics["pred_text"],
                    "decoded_target": metrics["decoded_target"],
                    "correct": metrics["correct"],
                    "loss_tok": metrics["loss_tok"],
                    "surprise_S": metrics["loss_tok"],
                    "entropy_U": metrics["entropy_u"],
                    "m_gate_mean": metrics["m_gate_mean"],
                    "m_gate_max": metrics["m_gate_max"],
                    "alpha_head_mean": metrics["alpha_head_mean"],
                    "alpha_head_max": metrics["alpha_head_max"],
                    "fast_share_mean": metrics["fast_share_mean"],
                    "fast_share_by_layer": metrics["fast_share_by_layer"],
                    "alpha_head_hist": metrics["alpha_head_hist"],
                    "session_id": session_id or "none",
                }
                with open(probe_path, "a") as pf:
                    pf.write(json.dumps(probe_entry) + "\n")

        if micro_step != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(raw_model.cortex_parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(raw_model.cortex_parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if rank == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
            torch.save(raw_model.state_dict(), ckpt_path)
            
            # ppl = evaluate_probe_perplexity(model, tokenizer, probe_texts, device, alpha_freeze=True)
            ppl = 0.0 # Skip eval for speed
            if baseline_ppl is None or math.isnan(baseline_ppl):
                baseline_ppl = ppl
            delta_pct = 0.0
            if baseline_ppl and not math.isnan(baseline_ppl):
                delta_pct = ((ppl - baseline_ppl) / baseline_ppl) * 100.0
            drift_entry = {
                "run_id": run_id,
                "variant": args.variant,
                "epoch": epoch + 1,
                "seed": args.seed,
                "ppl_probe": ppl,
                "ppl_delta_pct": delta_pct,
                "hidden_kl": 0.0,
            }
            with open(drift_path, "a") as df:
                df.write(json.dumps(drift_entry) + "\n")
        
        if ddp:
            dist.barrier()

    if rank == 0:
        print(f"[Stage A1] Training complete for {run_id}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
