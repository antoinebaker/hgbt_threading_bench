# HGBT threading benchmark

Benchmark HistGradientBoosting fit/predict under different OpenMP thread counts and data shapes. Used to collect data across platforms and core counts to evaluate heuristics for [scikit-learn#30662](https://github.com/scikit-learn/scikit-learn/issues/30662) (OpenMP slowdown on small data).

## Setup

Create environment using `conda` or `mamba`.

```bash
conda create -n hgbt_bench -c conda-forge scikit-learn matplotlib pandas psutil py-cpuinfo
```

`psutil` is optional; with it the CPU JSON records physical core count (`cpu_physical_cores`), otherwise that field is null.
On Linux the runner fills cache sizes and related fields from `lscpu` and `/proc/cpuinfo`; on macOS from `system_profiler` and `sysctl` (including `cpu_performance_cores` / `cpu_efficiency_cores` on Apple Silicon when present). `py-cpuinfo` runs last and only supplies keys still missing (`l1_data_cache_size`, `l1_instruction_cache_size`, `l2_cache_size`, `l3_cache_size`, `cpu_architecture`, `cpu_family`, etc.). First non-null value wins per field.

## Run benchmark

```bash
python run_benchmark.py
```

- Saves the benchmark data as `results/bench_num_threads.{run_id}.csv`
- Saves the CPU metadata (system, machine, processor, cores) as `bench_num_threads.{run_id}.cpu.json`
- Saves the speedup curves as `results/speedup_curves.{run_id}.png` (one figure with 2×3 subplots and CPU info at the bottom).
- Adds this run to `results/speedup_curves_all.pdf`.
- The CSV has one row per (shape, thread): `n_samples`, `n_features`, `max_num_threads`, `fit_time`, `predict_time`, `run_id`.

## Re-plot results

```bash
python plot_results.py <run_id>
```

Example:

```bash
python plot_results.py 20260218_160458
```

Use this to **re-plot** an existing run (e.g. after copying `results/` from another machine or to regenerate the figure without re-running the benchmark). Reads `results/bench_num_threads.{run_id}.csv` and `results/bench_num_threads.{run_id}.cpu.json`, and saves `results/speedup_curves.{run_id}.png`. Each figure includes run_id and CPU info at the bottom.

To plot **all** runs into one multi-page PDF (one run per page):

```bash
python plot_results.py --all
```

Writes `results/speedup_curves_all.pdf`.
