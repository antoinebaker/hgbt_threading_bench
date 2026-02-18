#!/usr/bin/env python3
"""
Load bench_num_threads CSV from results/, compute fit/predict speedups vs single-thread,
plot one combined figure (2×3 subplots) and save to results/speedup_curves.{run_id}.png.
See: https://github.com/scikit-learn/scikit-learn/issues/30662
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import pandas as pd


def compute_speedups(records):
    """Add fit_speedup and predict_speedup columns (reference = max_num_threads == 1)."""
    speedups = []
    for (n_samples, n_features), group in records.groupby(["n_samples", "n_features"]):
        ref_row = group[group["max_num_threads"] == 1]
        if ref_row.empty:
            continue
        group = group.copy()
        for method in ["fit", "predict"]:
            ref_duration = ref_row[f"{method}_time"].values[0]
            group[f"{method}_speedup"] = ref_duration / group[f"{method}_time"]
        speedups.append(group)
    if not speedups:
        return pd.DataFrame()
    return pd.concat(speedups, ignore_index=True)


def load_cpu_info(run_id: str) -> dict:
    """Load CPU metadata from results/bench_num_threads.{run_id}.cpu.json."""
    results_dir = Path("results")
    cpu_path = results_dir / f"bench_num_threads.{run_id}.cpu.json"
    if not cpu_path.exists():
        return {}
    try:
        with open(cpu_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def format_cpu(cpu_dict: dict) -> str:
    """Format CPU dict as multi-line string for figure text."""
    if not cpu_dict:
        return "CPU info not found"
    # Exclude run_id from display if present; show CPU fields only
    skip = {"run_id"}
    lines = [f"{k}: {v}" for k, v in cpu_dict.items() if k not in skip]
    return "\n".join(lines) if lines else "CPU info not found"


def plot_run(run_id: str) -> Path:
    """Load CSV and CPU JSON, build speedup figure, save to results/speedup_curves.{run_id}.png. Returns path to PNG."""
    results_dir = Path("results")
    csv_path = results_dir / f"bench_num_threads.{run_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    records = pd.read_csv(csv_path)
    required = ["n_samples", "n_features", "max_num_threads", "fit_time", "predict_time", "run_id"]
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    speedups = compute_speedups(records)
    if speedups.empty:
        raise ValueError("No groups (n_samples, n_features) with max_num_threads==1")

    cpu_info = load_cpu_info(run_id)
    cpu_text = format_cpu(cpu_info)

    n_features_list = [10, 100, 1000]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

    for col, n_features in enumerate(n_features_list):
        # Row 0: fit
        ax = axs[0, col]
        subset = speedups.query("n_features == @n_features")
        if not subset.empty:
            for (n_samples, n_feat), group in subset.groupby(["n_samples", "n_features"]):
                group.plot(
                    x="max_num_threads",
                    y="fit_speedup",
                    ax=ax,
                    label=f"X.shape=({n_samples}, {n_feat})",
                )
        ax.axhline(1, 0.95, 8.5, linestyle="--", color="gray")
        ax.set(
            xscale="log",
            xlabel="" if col > 0 else "",
            xticks=[1, 2, 4, 8],
            xticklabels=[1, 2, 4, 8],
            yscale="log",
            ylabel="Speedup (fit)" if col == 0 else "",
            yticks=[0.1, 0.2, 0.5, 1, 2, 5, 10],
            yticklabels=["0.1x", "0.2x", "0.5x", "1x", "2x", "5x", "10x"],
            title=f"n_features={n_features}",
        )
        ax.xaxis.set_minor_locator(NullLocator())

        # Row 1: predict
        ax = axs[1, col]
        if not subset.empty:
            for (n_samples, n_feat), group in subset.groupby(["n_samples", "n_features"]):
                group.plot(
                    x="max_num_threads",
                    y="predict_speedup",
                    ax=ax,
                    label=f"X.shape=({n_samples}, {n_feat})",
                )
        ax.axhline(1, 0.95, 8.5, linestyle="--", color="gray")
        ax.set(
            xscale="log",
            xlabel="Number of threads",
            xticks=[1, 2, 4, 8],
            xticklabels=[1, 2, 4, 8],
            yscale="log",
            ylabel="Speedup (predict)" if col == 0 else "",
            yticks=[0.1, 0.2, 0.5, 1, 2, 5, 10],
            yticklabels=["0.1x", "0.2x", "0.5x", "1x", "2x", "5x", "10x"],
            title="",
        )
        ax.xaxis.set_minor_locator(NullLocator())

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.text(0.02, 0.02, cpu_text, fontsize=7, verticalalignment="bottom", family="monospace")
    out_path = results_dir / f"speedup_curves.{run_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot HGBT threading benchmark results")
    parser.add_argument(
        "run_id",
        help="Run ID (e.g. 20260218_152651). Reads results/bench_num_threads.{run_id}.csv and .cpu.json.",
    )
    args = parser.parse_args()
    try:
        path = plot_run(args.run_id)
        print(f"Saved {path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
