#!/usr/bin/env python3
"""
Analyze H200 sensitivity sweep: 2TB CPU vs 32TB CXL.
Reads 12 output CSVs + prefix hit rates from log, generates plots and markdown report.

Usage:
    python analyze_h200_sensitivity_sweep.py
"""

import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


CONV_COUNTS = [50, 100, 150, 200, 250, 300]
MEM_CONFIGS = ["2tb", "cxl32tb"]
MEM_LABELS = {"2tb": "2 TB CPU", "cxl32tb": "32 TB CXL"}
MEM_COLORS = {"2tb": "#d62728", "cxl32tb": "#2ca02c"}
MEM_MARKERS = {"2tb": "s", "cxl32tb": "^"}
MEM_LINESTYLES = {"2tb": "--", "cxl32tb": "-."}

OUTPUT_DIR = Path("output/8H200")
LOG_PATH = OUTPUT_DIR / "h200_sensitivity_sweep_log.txt"
REPORT_PATH = OUTPUT_DIR / "h200_sensitivity_sweep_report.md"


def load_and_process(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['latency', 'TTFT', 'TPOT', 'queuing_delay']:
        df[f'{col}_ms'] = df[col] / 1e6
    df['compute_ttft_ms'] = df['TTFT_ms'] - df['queuing_delay_ms']
    df['turn'] = (df['arrival'] / 2_000_000_000).astype(int)
    return df


def compute_metrics(df):
    total_time_s = (df['end_time'].max() - df['arrival'].min()) / 1e9
    req_per_sec = len(df) / total_time_s if total_time_s > 0 else 0
    return {
        'num_requests': len(df),
        'simulated_runtime_s': total_time_s,
        'throughput_rps': req_per_sec,
        'compute_ttft_mean': df['compute_ttft_ms'].mean(),
        'compute_ttft_min': df['compute_ttft_ms'].min(),
        'compute_ttft_max': df['compute_ttft_ms'].max(),
        'tpot_mean': df['TPOT_ms'].mean(),
    }


def parse_prefix_hits_from_log(log_path):
    hits = {}
    if not Path(log_path).exists():
        return hits
    with open(log_path, 'r') as f:
        content = f.read()
    content = re.sub(r'\x1B\[[0-9;]*m', '', content)
    npu_hits = [float(m) for m in re.findall(r'NPU prefix hit ratio \(%\):\s+([\d.]+)', content)]
    cpu_hits = [float(m) for m in re.findall(r'(?:CPU|CXL) prefix hit ratio \(%\):\s+([\d.]+)', content)]
    total_hits = [float(m) for m in re.findall(r'Total prefix hit ratio \(%\):\s+([\d.]+)', content)]
    idx = 0
    for n in CONV_COUNTS:
        for mem in MEM_CONFIGS:
            if idx < len(total_hits):
                hits[(mem, n)] = {
                    'npu_hit': npu_hits[idx],
                    'cpu_hit': cpu_hits[idx],
                    'total_hit': total_hits[idx],
                }
            idx += 1
    return hits


def make_plot(data, y_key, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for mem in MEM_CONFIGS:
        xs, ys = [], []
        for n in CONV_COUNTS:
            key = (mem, n)
            if key in data and y_key in data[key]:
                xs.append(n)
                ys.append(data[key][y_key])
        if xs:
            ax.plot(xs, ys, color=MEM_COLORS[mem], linestyle=MEM_LINESTYLES[mem],
                    marker=MEM_MARKERS[mem], markersize=9, linewidth=2.5,
                    label=MEM_LABELS[mem])
    ax.set_xlabel('Number of Conversations', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(CONV_COUNTS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_paper_combined(all_metrics, output_dir):
    """Paper-format side-by-side: Mean TTFT + Prefix Hit"""
    plt.rcParams.update({
        'font.size': 5.5, 'axes.labelsize': 5.5,
        'xtick.labelsize': 5, 'ytick.labelsize': 5,
        'legend.fontsize': 5, 'lines.linewidth': 1.0,
        'lines.markersize': 3, 'axes.linewidth': 0.4,
        'grid.linewidth': 0.3, 'xtick.major.width': 0.4,
        'ytick.major.width': 0.4, 'xtick.major.pad': 1.5,
        'ytick.major.pad': 1.5, 'axes.labelpad': 2,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 1.2))

    for mem in MEM_CONFIGS:
        xs, ys_ttft, ys_hit = [], [], []
        for n in CONV_COUNTS:
            key = (mem, n)
            if key in all_metrics:
                xs.append(n)
                ys_ttft.append(all_metrics[key].get('compute_ttft_mean', 0))
                ys_hit.append(all_metrics[key].get('total_hit', 0))
        if xs:
            ax1.plot(xs, ys_ttft, color=MEM_COLORS[mem], linestyle=MEM_LINESTYLES[mem],
                     marker=MEM_MARKERS[mem], label=MEM_LABELS[mem])
            ax2.plot(xs, ys_hit, color=MEM_COLORS[mem], linestyle=MEM_LINESTYLES[mem],
                     marker=MEM_MARKERS[mem], label=MEM_LABELS[mem])

    ax1.set_xlabel('Conversations')
    ax1.set_ylabel('Mean Compute TTFT (ms)')
    ax1.set_xticks(CONV_COUNTS)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', borderpad=0.3, handlelength=1.5, handletextpad=0.4)

    ax2.set_xlabel('Conversations')
    ax2.set_ylabel('Total Prefix Hit (%)')
    ax2.set_xticks(CONV_COUNTS)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left', borderpad=0.3, handlelength=1.5, handletextpad=0.4)

    plt.tight_layout(pad=0.3, w_pad=0.8)
    path = output_dir / 'h200_sensitivity_ttft_and_hit_combined.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved (paper): {path}")


def make_per_turn_plot(results, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, n in enumerate(CONV_COUNTS):
        ax = axes[idx]
        for mem in MEM_CONFIGS:
            key = (mem, n)
            if key not in results:
                continue
            df = results[key]
            turns, ttfts = [], []
            for turn in range(10):
                sub = df[df['turn'] == turn]
                if len(sub) > 0:
                    turns.append(turn)
                    ttfts.append(sub['compute_ttft_ms'].mean())
            ax.plot(turns, ttfts, color=MEM_COLORS[mem], linestyle=MEM_LINESTYLES[mem],
                    marker=MEM_MARKERS[mem], markersize=6, linewidth=2,
                    label=MEM_LABELS[mem])
        ax.set_xlabel('Turn', fontsize=11)
        ax.set_ylabel('Mean Compute TTFT (ms)', fontsize=11)
        ax.set_title(f'{n} Conversations', fontsize=12, fontweight='bold')
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    results = {}
    all_metrics = {}

    print("Loading simulation results...")
    for n in CONV_COUNTS:
        for mem in MEM_CONFIGS:
            csv_path = OUTPUT_DIR / f"sensitivity_{mem}_{n}conv"
            if not csv_path.exists():
                print(f"  WARNING: Missing {csv_path}")
                continue
            df = load_and_process(str(csv_path))
            results[(mem, n)] = df
            all_metrics[(mem, n)] = compute_metrics(df)
            print(f"  Loaded {csv_path}: {len(df)} requests")

    if not results:
        print("ERROR: No results found.")
        return

    print(f"\nParsing prefix hits from {LOG_PATH}...")
    prefix_hits = parse_prefix_hits_from_log(LOG_PATH)
    for key, hits in prefix_hits.items():
        if key in all_metrics:
            all_metrics[key].update(hits)

    # Print summary
    print(f"\n{'='*130}")
    print("  Summary Table")
    print(f"{'='*130}")
    header = (f"{'Convs':<7} {'Memory':<12} {'Reqs':<7} "
              f"{'TTFT mean':>11} {'TTFT min':>11} {'TTFT max':>11} {'TPOT':>9} "
              f"{'Tput':>9} {'Runtime':>10} {'Hit%':>10}")
    print(header)
    print("-" * 130)
    for n in CONV_COUNTS:
        for mem in MEM_CONFIGS:
            key = (mem, n)
            if key not in all_metrics:
                continue
            m = all_metrics[key]
            hit_str = f"{m.get('total_hit', -1):>10.2f}" if m.get('total_hit', -1) >= 0 else f"{'N/A':>10}"
            print(f"{n:<7} {MEM_LABELS[mem]:<12} {m['num_requests']:<7} "
                  f"{m['compute_ttft_mean']:>11.1f} {m['compute_ttft_min']:>11.1f} "
                  f"{m['compute_ttft_max']:>11.1f} {m['tpot_mean']:>9.1f} "
                  f"{m['throughput_rps']:>9.3f} {m['simulated_runtime_s']:>10.1f} "
                  f"{hit_str}")

    # Speedup table
    print(f"\n  Speedup over 2TB CPU:")
    print(f"  {'Convs':<7} {'32TB CXL':>12}")
    print("  " + "-" * 22)
    for n in CONV_COUNTS:
        k2 = ("2tb", n)
        if k2 not in all_metrics:
            continue
        km = ("cxl32tb", n)
        if km in all_metrics and all_metrics[km]['compute_ttft_mean'] > 0:
            sp = all_metrics[k2]['compute_ttft_mean'] / all_metrics[km]['compute_ttft_mean']
            print(f"  {n:<7} {sp:>11.2f}x")
        else:
            print(f"  {n:<7} {'N/A':>12}")

    # Generate plots
    print("\nGenerating plots...")
    make_plot(all_metrics, 'compute_ttft_mean', 'Mean Compute TTFT (ms)',
              OUTPUT_DIR / 'h200_sensitivity_mean_ttft.png')
    make_plot(all_metrics, 'total_hit', 'Total Prefix Hit Rate (%)',
              OUTPUT_DIR / 'h200_sensitivity_prefix_hit.png')
    make_paper_combined(all_metrics, OUTPUT_DIR)


if __name__ == "__main__":
    main()
