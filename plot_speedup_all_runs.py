#!/usr/bin/env python3
"""All runs, one row per run_id: n_samples columns + CPU, n_features by color. Order = get_all_run_ids."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import pandas as pd

from plot_results import (
    RESULTS_DIR,
    format_cpu,
    get_all_run_ids,
    load_cpu_info,
    load_speedups_for_run,
)


def make_all_runs_figure(rows_data, speedup_col):
    combined = pd.concat([df for _, df in rows_data], ignore_index=True)
    n_samples_list = sorted(combined["n_samples"].unique())
    n_features_list = sorted(combined["n_features"].unique())
    thread_vals = sorted(combined["max_num_threads"].unique())
    n_rows = len(rows_data)
    n_data = len(n_samples_list)
    n_cols = n_data + 1

    cmap = plt.colormaps["tab10"]
    nf_color = {nf: cmap(i % 10) for i, nf in enumerate(n_features_list)}
    leg_handles = [
        plt.Line2D([0], [0], color=nf_color[nf], marker="o", linestyle="-", label=str(nf))
        for nf in n_features_list
    ]

    row_h = 2.4
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * max(n_data, 1) + 2.8, row_h * max(n_rows, 1) + 0.9),
        squeeze=False,
    )

    for row in range(n_rows):
        for c in range(1, n_data):
            axs[row, c].sharey(axs[row, 0])

    for row, (run_id, speedups) in enumerate(rows_data):
        cpu = load_cpu_info(run_id)
        hdr = f"{run_id}\n{cpu.get('machine', '?')} · {cpu.get('cpu_logical_cores', '?')} cores\n"
        cpu_body = hdr + format_cpu(cpu)

        for col, ns in enumerate(n_samples_list):
            ax = axs[row, col]
            sub = speedups[speedups["n_samples"] == ns]
            for nf in n_features_list:
                g = sub[sub["n_features"] == nf].sort_values("max_num_threads")
                if g.empty:
                    continue
                ax.plot(g["max_num_threads"], g[speedup_col], "o-", ms=4, color=nf_color[nf])
            ax.plot(thread_vals, thread_vals, "--", color="gray", zorder=0)
            ax.axhline(1.0, color="gray", ls=":", lw=0.8, zorder=0)
            ax.set(xscale="log", yscale="log", xticks=thread_vals, xticklabels=thread_vals)
            ax.xaxis.set_minor_locator(NullLocator())
            if row == n_rows - 1:
                ax.set_xlabel("Threads")
            if col == 0:
                ax.set_ylabel(speedup_col, fontsize=9)
                ax.legend(handles=leg_handles, title="n_features", loc="upper left", fontsize=7)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == 0:
                ax.set_title(f"n_samples = {ns:,}")
            ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
            ax.set_yticklabels(["0.1×", "0.2×", "0.5×", "1×", "2×", "5×", "10×"])

        ax_cpu = axs[row, n_data]
        ax_cpu.axis("off")
        ax_cpu.text(0, 1, cpu_body, transform=ax_cpu.transAxes, va="top", ha="left", fontsize=6, family="monospace")
        if row == 0:
            ax_cpu.set_title("CPU")

    fig.suptitle(f"{speedup_col} vs threads — same order as speedup_curves_all.pdf", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def main():
    rows_data = []
    for rid in get_all_run_ids():
        sp = load_speedups_for_run(rid)
        if not sp.empty:
            rows_data.append((rid, sp))
    if not rows_data:
        print("No speedup rows (need CSVs with max_num_threads==1 per shape).", file=sys.stderr)
        sys.exit(1)

    for name, col in (
        ("predict_speedup_all_runs.pdf", "predict_speedup"),
        ("fit_speedup_all_runs.pdf", "fit_speedup"),
    ):
        path = RESULTS_DIR / name
        fig = make_all_runs_figure(rows_data, col)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
