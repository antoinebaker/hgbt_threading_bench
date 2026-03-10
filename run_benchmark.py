#!/usr/bin/env python3
"""
Run HistGradientBoosting fit/predict benchmark across data shapes and thread counts.
Writes results/bench_num_threads.{run_id}.csv and results/bench_num_threads.{run_id}.cpu.json.
See: https://github.com/scikit-learn/scikit-learn/issues/30662
"""

import argparse
import io
import json
import math
import os
import platform
import subprocess
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn import show_versions
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import make_regression
from threadpoolctl import threadpool_info, threadpool_limits


def _get_package_version(pkg_name: str) -> str | None:
    """Try to get package version from conda list; return None on failure."""
    try:
        result = subprocess.run(
            ["conda", "list", pkg_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == pkg_name:
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

    # Threadpool info: keep only openmp entries (omit filepath)
    try:
        info["threadpool_info"] = [
            {k: v for k, v in d.items() if k != "filepath"}
            for d in threadpool_info() if d.get("user_api") == "openmp"
        ]
    except Exception:
        info["threadpool_info"] = []

    info["llvm_openmp_version"] = _get_package_version("llvm-openmp")
    info["libgomp_version"] = _get_package_version("libgomp")

    # Capture sklearn.show_versions() ouput
    info["sklearn_versions"] = show_versions()

    # Optional: L1/L2/L3 cache sizes and CPU architecture/family via py-cpuinfo
    _cpuinfo_keys = (
        "l1_data_cache_size",
        "l1_instruction_cache_size",
        "l2_cache_size",
        "l3_cache_size",
        "cpu_architecture",
        "cpu_family",
    )
    try:
        import cpuinfo
        raw = cpuinfo.get_cpu_info()
        info["l1_data_cache_size"] = raw.get("l1_data_cache_size")
        info["l1_instruction_cache_size"] = raw.get("l1_instruction_cache_size")
        info["l2_cache_size"] = raw.get("l2_cache_size")
        info["l3_cache_size"] = raw.get("l3_cache_size")
        info["cpu_architecture"] = raw.get("arch")
        info["cpu_family"] = raw.get("family")
    except Exception:
        for key in _cpuinfo_keys:
            info[key] = None

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
    max_threads = os.cpu_count() or 8
    all_max_num_threads = [2**i for i in range(int(math.log2(max_threads)) + 1)]

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

    from plot_results import plot_run, plot_all_runs
    png_path = plot_run(run_id)
    print(f"Wrote {png_path}")
    pdf_path = plot_all_runs()
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
