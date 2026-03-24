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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullLocator
import pandas as pd

RESULTS_DIR = Path("results")


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


def load_cpu_info(run_id):
    """Load CPU metadata from results/bench_num_threads.{run_id}.cpu.json."""
    cpu_path = RESULTS_DIR / f"bench_num_threads.{run_id}.cpu.json"
    if not cpu_path.exists():
        return {}
    try:
        with open(cpu_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def format_cpu(cpu_dict):
    """Format CPU dict as multi-line string for figure text."""
    if not cpu_dict:
        return "CPU info not found"
    skip = {"threadpool_info", "llvm_openmp_version", "libgomp_version", "sklearn_versions"}
    lines = [f"{k}: {v}" for k, v in cpu_dict.items() if k not in skip]
    if "threadpool_info" in cpu_dict and isinstance(cpu_dict["threadpool_info"], list):
        for i, lib in enumerate(cpu_dict["threadpool_info"]):
            if isinstance(lib, dict):
                parts = [f"{k}={v}" for k, v in lib.items()]
                lines.append(f"threadpool {i}: " + " ".join(parts))
    for key in ("llvm_openmp_version", "libgomp_version"):
        if cpu_dict.get(key):
            lines.append(f"{key}: {cpu_dict[key]}")
    return "\n".join(lines) if lines else "CPU info not found"


def get_all_run_ids():
    """Run IDs under results/, ordered by system, machine, cpu_logical_cores."""
    run_ids = []
    for p in RESULTS_DIR.glob("bench_num_threads.*.csv"):
        # stem is e.g. "bench_num_threads.20260226_165149"
        run_id = p.stem.replace("bench_num_threads.", "", 1)
        if run_id:
            run_ids.append(run_id)

    def sort_key(run_id: str):
        cpu = load_cpu_info(run_id)
        system = cpu.get("system")
        machine = cpu.get("machine")
        cores = cpu.get("cpu_logical_cores")
        return (system, machine, cores)

    return sorted(run_ids, key=sort_key)


def load_speedups_for_run(run_id):
    return compute_speedups(pd.read_csv(RESULTS_DIR / f"bench_num_threads.{run_id}.csv"))


def make_speedup_figure(run_id):
    speedups = load_speedups_for_run(run_id)
    cpu_info = load_cpu_info(run_id)
    cpu_text = format_cpu(cpu_info)

    n_features_list = [10, 100, 1000]
    thread_values = sorted(speedups["max_num_threads"].unique())
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

    for col, n_features in enumerate(n_features_list):
        # Row 0: fit
        ax = axs[0, col]
        subset = speedups.query("n_features == @n_features")
        ax.axhline(1, linestyle="--", color="gray", zorder=0)
        ax.plot(thread_values, thread_values, linestyle="--", color="gray", zorder=0)
        if not subset.empty:
            for (n_samples, n_feat), group in subset.groupby(["n_samples", "n_features"]):
                group.plot(
                    x="max_num_threads",
                    y="fit_speedup",
                    ax=ax,
                    label=f"{n_samples=}",
                )
        ax.set(
            xscale="log",
            xlabel="" if col > 0 else "",
            xticks=thread_values,
            xticklabels=thread_values,
            yscale="log",
            ylabel="Speedup (fit)" if col == 0 else "",
            yticks=[0.1, 0.2, 0.5, 1, 2, 5, 10],
            yticklabels=["0.1x", "0.2x", "0.5x", "1x", "2x", "5x", "10x"],
            title=f"{n_features=}",
        )
        ax.xaxis.set_minor_locator(NullLocator())

        # Row 1: predict
        ax = axs[1, col]
        ax.axhline(1, linestyle="--", color="gray", zorder=0)
        ax.plot(thread_values, thread_values, linestyle="--", color="gray", zorder=0)
        if not subset.empty:
            for (n_samples, n_feat), group in subset.groupby(["n_samples", "n_features"]):
                group.plot(
                    x="max_num_threads",
                    y="predict_speedup",
                    ax=ax,
                    label=f"{n_samples=}",
                )
        ax.set(
            xscale="log",
            xlabel="Number of threads",
            xticks=thread_values,
            xticklabels=thread_values,
            yscale="log",
            ylabel="Speedup (predict)" if col == 0 else "",
            yticks=[0.1, 0.2, 0.5, 1, 2, 5, 10],
            yticklabels=["0.1x", "0.2x", "0.5x", "1x", "2x", "5x", "10x"],
            title="",
        )
        ax.xaxis.set_minor_locator(NullLocator())

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.text(0.02, 0.02, cpu_text, fontsize=7, verticalalignment="bottom", family="monospace")
    return fig


def plot_run(run_id):
    """Load CSV and CPU JSON, build speedup figure, save to results/speedup_curves.{run_id}.png. Returns path to PNG."""
    fig = make_speedup_figure(run_id)
    out_path = RESULTS_DIR / f"speedup_curves.{run_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all_runs():
    """Plot all runs to a single PDF (one run per page). Returns path to PDF."""
    run_ids = get_all_run_ids()
    if not run_ids:
        raise ValueError("No bench_num_threads.*.csv files found in results/")
    out_path = RESULTS_DIR / "speedup_curves_all.pdf"
    with PdfPages(out_path) as pdf:
        for run_id in run_ids:
            try:
                fig = make_speedup_figure(run_id)
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {run_id}: {e}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot HGBT threading benchmark results")
    parser.add_argument(
        "run_id",
        nargs="?",
        help="Run ID (e.g. 20260218_152651). Reads results/bench_num_threads.{run_id}.csv and .cpu.json. Required unless --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all runs to one PDF (one run per page). Writes results/speedup_curves_all.pdf.",
    )
    args = parser.parse_args()
    try:
        if args.all:
            path = plot_all_runs()
            print(f"Saved {path}")
        else:
            if not args.run_id:
                parser.error("run_id required unless --all")
            path = plot_run(args.run_id)
            print(f"Saved {path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
