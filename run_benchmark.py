#!/usr/bin/env python3
"""
Run HistGradientBoosting fit/predict benchmark across data shapes and thread counts.
Writes results/bench_num_threads.{run_id}.csv and results/bench_num_threads.{run_id}.cpu.json.
See: https://github.com/scikit-learn/scikit-learn/issues/30662
"""

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import make_regression
from threadpoolctl import threadpool_info, threadpool_limits


def _get_llvm_openmp_version() -> str | None:
    """Try to get llvm-openmp version from conda; return None on failure."""
    try:
        result = subprocess.run(
            ["conda", "list", "llvm-openmp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "llvm-openmp":
                    return parts[1]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_cpu_info():
    """Collect CPU and platform info (portable). Uses psutil for physical cores if available."""
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical_cores": os.cpu_count(),
    }
    try:
        import psutil
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["cpu_physical_cores"] = None

    # Threadpool info (sklearn.ensemble already loaded); omit filepath and version
    try:
        raw = threadpool_info()
        info["threadpool_info"] = [
            {k: v for k, v in d.items() if k not in ("filepath", "version")}
            for d in raw
        ]
    except Exception:
        info["threadpool_info"] = []

    info["llvm_openmp_version"] = _get_llvm_openmp_version()
    return info


def estimate_time(func, *args, min_time=2):
    """Repeatedly call func(*args) until total elapsed time >= min_time; return mean time per call."""
    n_calls = 0
    total_time = 0.0
    while total_time < min_time:
        start = perf_counter()
        _ = func(*args)
        end = perf_counter()
        total_time += end - start
        n_calls += 1
    return total_time / n_calls


def main():
    parser = argparse.ArgumentParser(description="HGBT threading benchmark")
    parser.add_argument(
        "--min-time",
        type=float,
        default=2.0,
        help="Minimum seconds per (shape, thread) for timing (default: 2).",
    )
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cpu_info = get_cpu_info()
    csv_path = results_dir / f"bench_num_threads.{run_id}.csv"
    cpu_path = results_dir / f"bench_num_threads.{run_id}.cpu.json"

    data_shapes = [
        (100, 10),
        (1_000, 10),
        (10_000, 10),
        (100_000, 10),
        (1_000_000, 10),
        (100, 100),
        (1_000, 100),
        (10_000, 100),
        (100_000, 100),
        (1_000_000, 100),
        (100, 1_000),
        (1_000, 1_000),
    ]
    all_max_num_threads = [1, 2, 4, 8]

    records = []
    for max_num_threads in all_max_num_threads:
        for n_samples, n_features in data_shapes:
            X, y = make_regression(
                n_samples=n_samples, n_features=n_features, random_state=0
            )
            with threadpool_limits(limits=max_num_threads):
                hgbt = HistGradientBoostingRegressor()
                fit_time = estimate_time(hgbt.fit, X, y, min_time=args.min_time)
                predict_time = estimate_time(hgbt.predict, X, min_time=args.min_time)
            record = {
                "n_samples": n_samples,
                "n_features": n_features,
                "max_num_threads": max_num_threads,
                "fit_time": fit_time,
                "predict_time": predict_time,
                "run_id": run_id,
            }
            print(record)
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    with open(cpu_path, "w") as f:
        json.dump({"run_id": run_id, **cpu_info}, f, indent=2)
    print(f"Wrote {cpu_path}")

    from plot_results import plot_run
    plot_path = plot_run(run_id)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
