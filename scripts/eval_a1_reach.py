#!/usr/bin/env python3
"""
Stage A1 evaluation driver.

Reads training telemetry (probes.jsonl, drift.jsonl) and renders:
    - Reach curves (mean +/- log-level summary).
    - Fast-path usage violins per task/gap.
    - Gate vs surprise scatter plots.
    - Drift lines per variant.
    - Error diagnostics per task.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A1 reach evaluation")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--variants", type=str, nargs="+", required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--gaps", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--plot_dir", type=str, required=True)
    parser.add_argument("--use_session", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_json_lines(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def gather_probe_records(log_dir: Path, variants: List[str], tasks: List[str], gaps: List[int], seeds: List[int]) -> List[Dict]:
    records: List[Dict] = []
    for probes_file in log_dir.rglob("probes.jsonl"):
        for entry in load_json_lines(probes_file):
            if entry.get("variant") not in variants:
                continue
            if entry.get("task") not in tasks:
                continue
            if entry.get("gap") not in gaps:
                continue
            if entry.get("seed") not in seeds:
                continue
            records.append(entry)
    return records


def gather_drift_records(log_dir: Path, variants: List[str], seeds: List[int]) -> List[Dict]:
    records: List[Dict] = []
    for drift_file in log_dir.rglob("drift.jsonl"):
        for entry in load_json_lines(drift_file):
            if entry.get("variant") not in variants:
                continue
            if entry.get("seed") not in seeds:
                continue
            records.append(entry)
    return records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def compute_accuracy_stats(probe_records: List[Dict]) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    bucket: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
    for rec in probe_records:
        key = (rec["variant"], rec["task"], rec["gap"])
        bucket[key].append(int(rec.get("correct", 0)))

    stats: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for key, values in bucket.items():
        if not values:
            continue
        stats[key] = {
            "mean": mean(values),
            "stdev": stdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }
    return stats


def collect_fast_usage(probe_records: List[Dict], task: str, gap: int, variant: str) -> np.ndarray:
    layer_arrays: List[List[float]] = []
    for rec in probe_records:
        if rec["variant"] != variant or rec["task"] != task or rec["gap"] != gap:
            continue
        layer_arrays.append(rec.get("fast_share_by_layer", []))
    if not layer_arrays:
        return np.array([])
    return np.array(layer_arrays)


def collect_gate_pairs(probe_records: List[Dict], task: str, gap: int, variant: str) -> Tuple[List[float], List[float]]:
    surprises, gates = [], []
    for rec in probe_records:
        if rec["variant"] != variant or rec["task"] != task or rec["gap"] != gap:
            continue
        surprises.append(float(rec.get("surprise_S", 0.0)))
        gates.append(float(rec.get("m_gate_mean", 0.0)))
    return surprises, gates


def aggregate_drift(drift_records: List[Dict]) -> Dict[Tuple[str, int], List[float]]:
    agg: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for rec in drift_records:
        key = (rec["variant"], rec["epoch"])
        agg[key].append(float(rec.get("ppl_probe", 0.0)))
    return agg


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_reach_curves(
    stats: Dict[Tuple[str, str, int], Dict[str, float]],
    variants: List[str],
    tasks: List[str],
    gaps: List[int],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    for task in tasks:
        plt.figure(figsize=(8, 5))
        baseline_reach = 0
        for variant in variants:
            y_values = []
            available_gaps = []
            for gap in sorted(gaps):
                key = (variant, task, gap)
                if key not in stats:
                    continue
                available_gaps.append(gap)
                y_values.append(stats[key]["mean"])
            if not available_gaps:
                continue
            plt.plot(available_gaps, y_values, marker="o", label=f"{variant}")
            if variant == "baseline":
                above = [gap for gap in available_gaps if stats[(variant, task, gap)]["mean"] >= 0.8]
                if above:
                    baseline_reach = max(above)
        if baseline_reach > 0:
            target_gap = baseline_reach * 4
            plt.axvline(target_gap, linestyle="--", color="black", alpha=0.4, label=f"4x baseline ({target_gap})")
        plt.xlabel("Gap (tokens)")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.title(f"Reach curve - {task}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / f"reach_{task}.png", dpi=200, bbox_inches="tight")
        plt.close()


def plot_fast_usage_violins(
    probe_records: List[Dict],
    tasks: List[str],
    gaps: List[int],
    output_dir: Path,
    variant: str,
) -> None:
    ensure_dir(output_dir)
    for task in tasks:
        for gap in gaps:
            data = collect_fast_usage(probe_records, task, gap, variant)
            if data.size == 0:
                continue
            data = data.T  # layers x samples
            plt.figure(figsize=(10, 4))
            parts = plt.violinplot(data, showmeans=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor("#268bd2")
                pc.set_alpha(0.6)
            layer_means = data.mean(axis=1)
            plt.axhline(0.3, color="red", linestyle="--", linewidth=1.0, label="0.3 threshold")
            for idx, val in enumerate(layer_means, start=1):
                if val >= 0.3:
                    plt.text(idx, val + 0.01, "★", ha="center", va="bottom", color="darkred", fontsize=10)
            plt.xticks(range(1, data.shape[0] + 1), [str(i) for i in range(data.shape[0])])
            plt.ylabel("Fast share")
            plt.xlabel("Layer")
            plt.title(f"Fast usage {variant} - {task} gap {gap}")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.savefig(output_dir / f"fast_usage_{task}_{gap}.png", dpi=200, bbox_inches="tight")
            plt.close()


def plot_gate_vs_surprise(
    probe_records: List[Dict],
    tasks: List[str],
    gaps: List[int],
    output_dir: Path,
    variant: str,
) -> None:
    ensure_dir(output_dir)
    for task in tasks:
        for gap in gaps:
            surprises, gates = collect_gate_pairs(probe_records, task, gap, variant)
            if not surprises:
                continue
            plt.figure(figsize=(6, 5))
            plt.scatter(surprises, gates, alpha=0.6, color="tab:green")
            if len(surprises) > 1:
                corr = np.corrcoef(surprises, gates)[0, 1]
                plt.title(f"Gate vs surprise ({task}, gap {gap}) r={corr:.2f}")
            else:
                plt.title(f"Gate vs surprise ({task}, gap {gap})")
            plt.xlabel("Surprise S")
            plt.ylabel("m gate mean")
            plt.grid(alpha=0.3)
            plt.savefig(output_dir / f"gate_vs_surprise_{task}_{gap}.png", dpi=200, bbox_inches="tight")
            plt.close()


def plot_drift_lines(drift_records: List[Dict], variants: List[str], output_dir: Path) -> None:
    ensure_dir(output_dir)
    aggregated = aggregate_drift(drift_records)
    epochs = sorted({epoch for (_, epoch) in aggregated.keys()})
    for variant in variants:
        variant_means = []
        variant_stds = []
        for epoch in epochs:
            values = aggregated.get((variant, epoch), [])
            if values:
                variant_means.append(mean(values))
                variant_stds.append(stdev(values) if len(values) > 1 else 0.0)
            else:
                variant_means.append(float("nan"))
                variant_stds.append(0.0)
        if not any(math.isfinite(v) for v in variant_means):
            continue
        plt.figure(figsize=(6, 4))
        plt.errorbar(epochs, variant_means, yerr=variant_stds, marker="o", label=variant)
        plt.xlabel("Epoch")
        plt.ylabel("Probe PPL")
        plt.title(f"Drift - {variant}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / f"drift_{variant}.png", dpi=200, bbox_inches="tight")
        plt.close()


def plot_error_modes(probe_records: List[Dict], tasks: List[str], gaps: List[int], output_dir: Path, variant: str) -> None:
    ensure_dir(output_dir)
    for task in tasks:
        task_records = [rec for rec in probe_records if rec["variant"] == variant and rec["task"] == task]
        if not task_records:
            continue
        if task == "nback":
            label_map = {"YES": 0, "NO": 1}
            confusion = np.zeros((2, 2), dtype=int)
            for rec in task_records:
                pred = rec.get("pred_text", "").strip().upper()
                target = rec.get("answer_text", "").strip().upper()
                if pred in label_map and target in label_map:
                    confusion[label_map[target], label_map[pred]] += 1
            plt.figure(figsize=(4, 4))
            plt.imshow(confusion, cmap="Blues")
            plt.xticks([0, 1], ["YES", "NO"])
            plt.yticks([0, 1], ["YES", "NO"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, confusion[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("Target")
            plt.title(f"Errors {task}")
            plt.colorbar()
            plt.savefig(output_dir / f"errors_{task}.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            failure_rates: Dict[int, List[int]] = defaultdict(list)
            for rec in task_records:
                failure_rates[rec["gap"]].append(1 - int(rec.get("correct", 0)))
            xs = sorted(failure_rates.keys())
            ys = [mean(failure_rates[x]) for x in xs]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, marker="o", color="tab:red")
            plt.xlabel("Gap")
            plt.ylabel("Failure rate")
            plt.title(f"Errors {task}")
            plt.grid(alpha=0.3)
            plt.savefig(output_dir / f"errors_{task}.png", dpi=200, bbox_inches="tight")
            plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    models_root = Path(args.models)
    if not models_root.exists():
        raise RuntimeError(f"Models directory not found: {models_root}")
    log_root = Path(args.log_dir)
    plot_root = Path(args.plot_dir)
    ensure_dir(plot_root)

    probe_records = gather_probe_records(log_root, args.variants, args.tasks, args.gaps, args.seeds)
    if not probe_records:
        raise RuntimeError("No probe records found. Ensure training logs are present.")

    drift_records = gather_drift_records(log_root, args.variants, args.seeds)

    stats = compute_accuracy_stats(probe_records)
    plot_reach_curves(stats, args.variants, args.tasks, args.gaps, plot_root)
    plot_fast_usage_violins(probe_records, args.tasks, args.gaps, plot_root, variant="cortex")
    plot_gate_vs_surprise(probe_records, args.tasks, args.gaps, plot_root, variant="cortex")
    if drift_records:
        plot_drift_lines(drift_records, args.variants, plot_root)
    plot_error_modes(probe_records, args.tasks, args.gaps, plot_root, variant="cortex")

    for variant in args.variants:
        for task in args.tasks:
            for gap in args.gaps:
                key = (variant, task, gap)
                if key in stats:
                    summary = stats[key]
                    print(
                        f"{variant} task={task} gap={gap} accuracy={summary['mean']:.3f}"
                        f" sd={summary['stdev']:.3f} count={summary['count']}"
                    )


if __name__ == "__main__":
    main()
