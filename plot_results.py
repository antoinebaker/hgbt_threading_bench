#!/usr/bin/env python3
"""
Load bench_num_threads CSV, compute fit/predict speedups vs single-thread,
plot speedup vs max_num_threads, and save figures.
See: https://github.com/scikit-learn/scikit-learn/issues/30662
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_speedup_curves(speedups, n_features, run_id, output_dir, show=False):
    """One figure with two subplots (fit, predict) for the given n_features."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, method in zip(axs, ["fit", "predict"]):
        subset = speedups.query("n_features == @n_features")
        if subset.empty:
            continue
        for (n_samples, n_feat), group in subset.groupby(["n_samples", "n_features"]):
            group.plot(
                x="max_num_threads",
                y=f"{method}_speedup",
                ax=ax,
                label=f"X.shape=({n_samples}, {n_feat})",
            )
        ax.axhline(1, 0.95, 8.5, linestyle="--", color="gray")
        ax.set(
            xscale="log",
            xlabel="Number of threads",
            xticks=[1, 4, 8],
            xticklabels=["1", "4", "8"],
            yscale="log",
            ylabel=f"Speedup ({method})",
            yticks=[0.1, 0.2, 0.5, 1, 2, 5, 10],
            yticklabels=["0.1x", "0.2x", "0.5x", "1x", "2x", "5x", "10x"],
            title=f"Impact of threading on {method} time (n_features={n_features})",
        )
    fig.tight_layout()
    out_path = output_dir / f"speedup_curves_n_features_{n_features}.{run_id}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot HGBT threading benchmark results")
    parser.add_argument(
        "--input",
        default="bench_num_threads.csv",
        help="Input CSV path (default: bench_num_threads.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for saved PNGs (default: current directory).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    records = pd.read_csv(input_path)
    required = ["n_samples", "n_features", "max_num_threads", "fit_time", "predict_time"]
    missing = [c for c in required if c not in records.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Use run_id from CSV if present (first row), else from filename stem (e.g. bench_num_threads.20250218_123456)
    if "run_id" in records.columns and not records["run_id"].empty:
        run_id = str(records["run_id"].iloc[0])
    else:
        run_id = input_path.stem.replace("bench_num_threads.", "") or "run"

    speedups = compute_speedups(records)
    if speedups.empty:
        print("Error: no groups (n_samples, n_features) with max_num_threads==1", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_features in [10, 100, 1000]:
        if (speedups["n_features"] == n_features).any():
            plot_speedup_curves(
                speedups, n_features, run_id, output_dir, show=args.show
            )


if __name__ == "__main__":
    main()
