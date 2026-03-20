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


def _coalesce_set(info: dict, key: str, value) -> None:
    """Set info[key] only if missing or None and value is not None."""
    if value is not None and info.get(key) is None:
        info[key] = value


def _apply_from_lscpu(info: dict) -> None:
    if platform.system() != "Linux":
        return
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        label, rest = line.split(":", 1)
        label_l = label.strip().lower()
        val = rest.strip()
        if not val:
            continue
        if label_l in ("l1d cache", "l1d"):
            _coalesce_set(info, "l1_data_cache_size", val)
        elif label_l in ("l1i cache", "l1i"):
            _coalesce_set(info, "l1_instruction_cache_size", val)
        elif label_l in ("l2 cache", "l2"):
            _coalesce_set(info, "l2_cache_size", val)
        elif label_l in ("l3 cache", "l3"):
            _coalesce_set(info, "l3_cache_size", val)
        elif label_l == "cpu family":
            _coalesce_set(info, "cpu_family", val)


def _apply_from_proc_cpuinfo(info: dict) -> None:
    if platform.system() != "Linux":
        return
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    cache_size_raw: str | None = None
    family_raw: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("cache size"):
            _, _, rhs = line.partition(":")
            rhs = rhs.strip()
            if rhs and cache_size_raw is None:
                cache_size_raw = rhs
        elif line.lower().startswith("cpu family"):
            _, _, rhs = line.partition(":")
            rhs = rhs.strip()
            if rhs and family_raw is None:
                family_raw = rhs

    if cache_size_raw is not None:
        _coalesce_set(info, "l2_cache_size", cache_size_raw)
    if family_raw is not None:
        _coalesce_set(info, "cpu_family", family_raw)


def _apply_from_system_profiler(info: dict) -> None:
    if platform.system() != "Darwin":
        return
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    for line in result.stdout.splitlines():
        raw = line.strip()
        if not raw or ":" not in raw:
            continue
        label, rest = raw.split(":", 1)
        label_l = label.strip().lower()
        rhs = rest.strip()
        if label_l == "chip":
            _coalesce_set(info, "cpu_family", rhs)
            continue
        if "cache" in label_l:
            if "l2" in label_l and rhs:
                _coalesce_set(info, "l2_cache_size", rhs)
            if "l3" in label_l and rhs:
                _coalesce_set(info, "l3_cache_size", rhs)


def _sysctl_read_int_darwin(key: str) -> int | None:
    try:
        r = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        v = r.stdout.strip()
        if not v:
            return None
        n = int(v)
        if n <= 0:
            return None
        return n
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _apply_from_sysctl_darwin(info: dict) -> None:
    if platform.system() != "Darwin":
        return

    l1d = _sysctl_read_int_darwin("hw.l1dcachesize")
    if l1d is None:
        l1d = _sysctl_read_int_darwin("hw.perflevel0.l1dcachesize")
    l1i = _sysctl_read_int_darwin("hw.l1icachesize")
    if l1i is None:
        l1i = _sysctl_read_int_darwin("hw.perflevel0.l1icachesize")
    l2 = _sysctl_read_int_darwin("hw.l2cachesize")
    if l2 is None:
        l2 = _sysctl_read_int_darwin("hw.perflevel0.l2cachesize")
    l3 = _sysctl_read_int_darwin("hw.l3cachesize")
    if l3 is None:
        l3 = _sysctl_read_int_darwin("hw.perflevel0.l3cachesize")

    _coalesce_set(info, "l1_data_cache_size", l1d)
    _coalesce_set(info, "l1_instruction_cache_size", l1i)
    _coalesce_set(info, "l2_cache_size", l2)
    _coalesce_set(info, "l3_cache_size", l3)

    # Apple Silicon: perflevel0 = performance, perflevel1 = efficiency.
    # Only record when arm64 or hybrid sysctl is clearly present (avoid Intel mislabels).
    p_perf = _sysctl_read_int_darwin("hw.perflevel0.physicalcpu")
    p_eff = _sysctl_read_int_darwin("hw.perflevel1.physicalcpu")
    if platform.machine() == "arm64" or p_eff is not None:
        _coalesce_set(info, "cpu_performance_cores", p_perf)
        _coalesce_set(info, "cpu_efficiency_cores", p_eff)


def _apply_from_cpuinfo(info: dict) -> None:
    try:
        import cpuinfo
        raw = cpuinfo.get_cpu_info()
    except Exception:
        return
    _coalesce_set(info, "l1_data_cache_size", raw.get("l1_data_cache_size"))
    _coalesce_set(info, "l1_instruction_cache_size", raw.get("l1_instruction_cache_size"))
    _coalesce_set(info, "l2_cache_size", raw.get("l2_cache_size"))
    _coalesce_set(info, "l3_cache_size", raw.get("l3_cache_size"))
    _coalesce_set(info, "cpu_architecture", raw.get("arch"))
    _coalesce_set(info, "cpu_family", raw.get("family"))


def get_cpu_info():
    """Collect CPU and platform info (portable). Uses psutil for physical cores if available.

    Cache sizes and cpu_family are filled from (Linux) lscpu and /proc/cpuinfo, then (macOS)
    system_profiler and sysctl, then py-cpuinfo for any keys still missing. First non-None value
    wins per key.

    On Apple Silicon, cpu_performance_cores / cpu_efficiency_cores come from sysctl when
    available; total physical cores remain in cpu_physical_cores (e.g. 8 = 4 + 4).
    """
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

    # Capture sklearn.show_versions() output (it prints, does not return)
    buf = io.StringIO()
    with redirect_stdout(buf):
        show_versions()
    info["sklearn_versions"] = buf.getvalue()

    _extended_keys = (
        "l1_data_cache_size",
        "l1_instruction_cache_size",
        "l2_cache_size",
        "l3_cache_size",
        "cpu_architecture",
        "cpu_family",
        "cpu_performance_cores",
        "cpu_efficiency_cores",
    )
    for key in _extended_keys:
        info[key] = None

    # lscpu → /proc/cpuinfo → system_profiler → sysctl → py-cpuinfo; first wins per key.
    for apply_func in (
        _apply_from_lscpu,
        _apply_from_proc_cpuinfo,
        _apply_from_system_profiler,
        _apply_from_sysctl_darwin,
        _apply_from_cpuinfo,
    ):
        try:
            apply_func(info)
        except Exception:
            pass

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
