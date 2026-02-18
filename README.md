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
python run_benchmark.py [--min-time 2]
```

- Writes to **results/**: `bench_num_threads.{run_id}.csv` and `bench_num_threads.{run_id}.cpu.json` (e.g. `bench_num_threads.20260218_160458.csv`). The CSV has one row per (X_shape, max_num_threads): `n_samples`, `n_features`, `max_num_threads`, `fit_time`, `predict_time`, `run_id`. CPU metadata (system, machine, processor, cores) is in the JSON only.

## Plot results

```bash
python plot_results.py <run_id>
```

Example:

```bash
python plot_results.py 20260218_160458
```

- Reads `results/bench_num_threads.{run_id}.csv` and `results/bench_num_threads.{run_id}.cpu.json`, computes fit/predict speedup vs single-thread, and saves **results/speedup_curves.{run_id}.png** (one figure with 2×3 subplots and CPU info at the bottom).
