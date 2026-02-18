# HGBT threading benchmark

Benchmark HistGradientBoosting fit/predict under different OpenMP thread counts and data shapes. Saves CSV with CPU/platform metadata and scripts to plot speedup curves. Used to collect data across platforms and core counts to evaluate heuristics for [scikit-learn#30662](https://github.com/scikit-learn/scikit-learn/issues/30662) (OpenMP slowdown on small data).

## Setup

**Conda:**

```bash
cd hgbt_threading_bench
conda create -n hgbt_bench python=3.11 -y
conda activate hgbt_bench
pip install -r requirements.txt
```

**venv:**

```bash
cd hgbt_threading_bench
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`psutil` is required for physical core count; without it only logical cores are recorded.

## Run benchmark

```bash
python run_benchmark.py
```

Writes to **results/**: `bench_num_threads.{run_id}.csv`, `bench_num_threads.{run_id}.cpu.json`, and `speedup_curves.{run_id}.png`. One command produces the CSV, CPU metadata, and the speedup plot. The CSV has one row per (shape, thread): `n_samples`, `n_features`, `max_num_threads`, `fit_time`, `predict_time`, `run_id`. CPU metadata (system, machine, processor, cores) is in the JSON only.

## Plot results

```bash
python plot_results.py <run_id>
```

Example:

```bash
python plot_results.py 20260218_160458
```

Use this to **re-plot** an existing run (e.g. after copying **results/** from another machine or to regenerate the figure without re-running the benchmark). Reads `results/bench_num_threads.{run_id}.csv` and `results/bench_num_threads.{run_id}.cpu.json`, and saves `results/speedup_curves.{run_id}.png` (one figure with 2×3 subplots and CPU info at the bottom).
