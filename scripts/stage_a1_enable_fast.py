#!/usr/bin/env python3
"""
Stage A1 training harness with Cortex fast weights.

Implements the run book interface:
    - CLI flags for model selection, task, gaps, variant, and logging dirs.
    - Per-sample probe JSONL logging with gate and fast-share telemetry.
    - Drift monitoring per epoch.
    - Variant toggles for cortex / fast_off / baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer

from base.hf_wrap import CortexWrapConfig, CortexWrappedModel, load_qwen_with_cortex
from train.data_long import NOISE_TOKENS, build_dataset, sample_to_training_pair


TASK_KEY_MAP = {
    "kv": "kv",
    "copy": "copy_reverse",
    "nback": "nback",
    "add": "addition",
}


# ---------------------------------------------------------------------------
# CLI parsing utilities
# ---------------------------------------------------------------------------


def str_to_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A1 Cortex training")
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
    parser.add_argument("--fast_rank", type=int, default=16)
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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
    prev_mode = model._mix_mode
    prev_alpha = model._alpha_override
    prev_sidecar = model._sidecar_enabled
    if alpha_freeze:
        model.set_alpha_override(0.0)
        model.set_mix_mode("slow_only")
        model.enable_sidecar(True)

    model.eval()
    losses = []
    with torch.no_grad():
        for text in probe_texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded, labels=encoded["input_ids"], session_id="probe", reset_session=True)
            losses.append(outputs.loss.item())
    model.train()

    if alpha_freeze:
        model.set_alpha_override(prev_alpha)
        model.set_mix_mode(prev_mode)
        model.enable_sidecar(prev_sidecar)

    if not losses:
        return float("nan")
    return float(math.exp(sum(losses) / len(losses)))


def summarize_fast_stats(
    model: CortexWrappedModel,
    answer_positions: torch.Tensor,
) -> Tuple[List[float], List[List[float]]]:
    fast_layer_means: List[float] = []
    alpha_layer_hist: List[List[float]] = []
    answer_idx = answer_positions.cpu()

    for block in model._cortex_layers:
        if block.last_fast_share is not None and block.last_fast_share.numel() > 0:
            share = block.last_fast_share
            if share.dim() == 3:
                share = share[0]
            if answer_idx.numel() > 0:
                share_sel = share[answer_idx]
            else:
                share_sel = share
            fast_layer_means.append(float(share_sel.mean().item()) if share_sel.numel() > 0 else 0.0)
        else:
            fast_layer_means.append(0.0)

        if getattr(block, "last_alpha", None) is not None and block.last_alpha.numel() > 0:
            alpha = block.last_alpha
            if alpha.dim() == 3:
                alpha = alpha[0]
            if answer_idx.numel() > 0:
                alpha_sel = alpha[answer_idx]
            else:
                alpha_sel = alpha
            if alpha_sel.numel() > 0:
                alpha_layer_hist.append(alpha_sel.mean(dim=0).tolist())
            else:
                alpha_layer_hist.append([0.0] * alpha.shape[-1])
        else:
            alpha_layer_hist.append([0.0] * model.base.config.num_attention_heads)

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
    if model._last_gates is not None:
        m_gate_tensor = model._last_gates[0]
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


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    amp_enabled = str_to_bool(args.amp)
    use_session = str_to_bool(args.use_session)

    if args.cfg:
        with open(args.cfg) as f:
            cfg_overrides = yaml.safe_load(f)
        for key, value in cfg_overrides.items():
            if not hasattr(args, key):
                continue
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if amp_enabled and device.type == "cuda" else None

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

    if args.variant in {"fast_off", "baseline"}:
        model.set_alpha_override(0.0)
        model.set_mix_mode("slow_only")
    else:
        model.set_alpha_override(None)
        model.set_mix_mode("dual")

    model.enable_sidecar(args.variant != "baseline")

    model.to(device)
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
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)

    probe_path = log_dir / "probes.jsonl"
    drift_path = log_dir / "drift.jsonl"
    if probe_path.exists():
        probe_path.unlink()
    if drift_path.exists():
        drift_path.unlink()

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

    optimizer = torch.optim.AdamW(model.cortex_parameters(), lr=args.lr_sidecar, weight_decay=0.0)
    scaler = GradScaler('cuda', enabled=amp_enabled and device.type == "cuda")

    probe_texts = build_probe_texts(2000, args.seed)
    baseline_ppl: Optional[float] = None

    accum_steps = max(1, args.batch_size)
    global_step = 0
    for epoch in range(args.epochs):
        rng.shuffle(all_samples)
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        for sample in all_samples:
            global_step += 1
            micro_step += 1

            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.tensor(sample["labels"], dtype=torch.long, device=device).unsqueeze(0)
            session_id = f"{run_id}_{sample['sample_id']}" if use_session else None

            with autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    session_id=session_id,
                    reset_session=True if use_session else False,
                )
                loss = outputs.loss

            loss = loss / accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if micro_step % accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    if args.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.cortex_parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.cortex_parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            metrics = compute_metrics(outputs, labels, tokenizer, model)
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
                    torch.nn.utils.clip_grad_norm_(model.cortex_parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.cortex_parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        ckpt_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), ckpt_path)

        ppl = evaluate_probe_perplexity(model, tokenizer, probe_texts, device, alpha_freeze=True)
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

    print(f"[Stage A1] Training complete for {run_id}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
