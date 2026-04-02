#!/usr/bin/env python3
"""Collect fit/predict times and speedups vs 1 thread for each (n_samples, n_features) in X_shapes."""

import os
import platform
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from run_benchmark import estimate_time

MIN_TIME = 2.0


def run_speedup_benchmark(X_shapes):
    n_cpus = os.cpu_count()
    if n_cpus is None:
        raise RuntimeError("os.cpu_count() returned None")

    system = platform.system()
    machine = platform.machine()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"bench_speedup_dataset.{run_id}.csv"

    records = []
    for n_samples, n_features in tqdm(X_shapes):
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, random_state=0
        )
        fit_time_1 = None
        predict_time_1 = None
        for n_threads in range(1, n_cpus + 1):
            with threadpool_limits(limits=n_threads):
                hgbt = HistGradientBoostingRegressor()
                fit_time = estimate_time(hgbt.fit, X, y, min_time=MIN_TIME)
                predict_time = estimate_time(hgbt.predict, X, min_time=MIN_TIME)
            if n_threads == 1:
                fit_time_1 = fit_time
                predict_time_1 = predict_time
            record = {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_threads": n_threads,
                "n_cpus": n_cpus,
                "system": system,
                "machine": machine,
                "fit_time": fit_time,
                "predict_time": predict_time,
                "fit_speedup": fit_time_1 / fit_time,
                "predict_speedup": predict_time_1 / predict_time,
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def main():
    X_shapes = [
        (int(n_samples), int(n_features)) 
        for n_samples in np.logspace(2, 6, num=9)
        for n_features in np.logspace(1, 4, num=7)
        if n_samples*n_features < 1e8
    ]
    run_speedup_benchmark(X_shapes)


if __name__ == "__main__":
    main()
