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
python run_benchmark.py [--output bench_num_threads.{run_id}.csv] [--min-time 2]
```

- Writes **one CSV per run** as `bench_num_threads.{run_id}.csv` by default (e.g. `bench_num_threads.20250218_143022.csv`). Override with `--output path.csv`.
- Each row: `n_samples`, `n_features`, `max_num_threads`, `fit_time`, `predict_time`, plus CPU columns: `platform`, `machine`, `processor`, `cpu_physical_cores`, `cpu_logical_cores`, `run_id`, `run_timestamp`.

## Plot results

```bash
python plot_results.py [--input bench_num_threads.20250218_143022.csv] [--output-dir .] [--show]
```

- Reads the CSV, computes fit/predict speedup vs single-thread, and saves one PNG per `n_features` (10, 100, 1000): `speedup_curves_n_features_{10,100,1000}.{run_id}.png` in `--output-dir`.
- Use `--show` to display plots after saving.

## Multi-platform comparison

1. Run `run_benchmark.py` on each machine (or with different core affinity, e.g. `OMP_NUM_THREADS=4 python run_benchmark.py`).
2. Copy or concatenate the resulting CSVs (each has a unique `run_id` and CPU columns).
3. Run `plot_results.py --input <each_csv>` to get figures per run; compare speedup curves across platform/machine/core count to see where threading helps or hurts and tune heuristics (e.g. when to disable or cap threads by data shape and core count).
