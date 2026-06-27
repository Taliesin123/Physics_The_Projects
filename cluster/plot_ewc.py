"""
Standalone plotting for the EWC experiment.

Reads  ewc_rotated_mnist_data.csv  and writes  ewc_rotated_mnist_plot.png,
so you can tweak plot styling WITHOUT re-running the (expensive) training.

    python plot_ewc.py
"""

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless-cluster safe (no X11 needed)
import matplotlib.pyplot as plt

# Same hardcoded location used by EWC_plots.py
HERE = '/iopsstor/scratch/cscs/tperez/cluster/'


def load_results(csv_path):
    """CSV (step, method, task, accuracy) -> {method: [task0, task1, task2]}."""
    tmp = defaultdict(lambda: defaultdict(dict))   # method -> task -> {step: acc}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            tmp[row['method']][int(row['task'])][int(row['step'])] = float(row['accuracy'])
    results = {}
    for m, per_task in tmp.items():
        results[m] = [[per_task[t][s] for s in sorted(per_task[t])] for t in range(3)]
    return results


def plot_from_csv(csv_path, out_path):
    results = load_results(csv_path)

    n_steps_total      = len(results['sgd'][0])
    steps_per_phase    = n_steps_total // 3
    phase_boundaries_x = [steps_per_phase, 2 * steps_per_phase]

    x = np.arange(1, n_steps_total + 1)

    method_color = {'sgd': '#3a6fa0', 'l2': '#5a9b5a', 'ewc': '#c14242'}
    method_label = {'sgd': 'SGD',     'l2': r'L$_2$', 'ewc': 'EWC'}

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True)

    for tid, ax in enumerate(axes):
        start = tid * steps_per_phase

        for method in ['sgd', 'l2', 'ewc']:
            accs = np.array(results[method][tid])
            ax.plot(
                x[start:], accs[start:],
                color=method_color[method],
                linewidth=2,
                label=method_label[method] if tid == 0 else None,
            )

        for b in phase_boundaries_x:
            ax.axvline(b + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.2, 0.5, 1.0])
        ax.set_yticklabels(['0.2', '0.5', '1.0'])

        ax.axhline(0.1, color='gray', linewidth=0.7, linestyle=':', alpha=0.5)
        ax.set_ylabel(f"Task {chr(ord('A') + tid)}", fontsize=12, rotation=90, labelpad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_xticks([])
        ax.set_xlim(0.5, n_steps_total + 0.5)

    top_ax = axes[0]
    phase_centres = [steps_per_phase * (i + 0.5) for i in range(3)]
    for cx, label in zip(phase_centres, ['Train A (0°)', 'Train B (45°)', 'Train C (90°)']):
        top_ax.text(cx, 1.05, label, ha='center', va='bottom',
                    fontsize=12, transform=top_ax.get_xaxis_transform())

    last_x = n_steps_total + 0.5
    last_y = {m: results[m][0][-1] for m in ['sgd', 'l2', 'ewc']}
    ys_sorted = sorted(last_y.items(), key=lambda kv: kv[1])
    min_gap = 0.025
    adjusted = []
    prev_y = -1
    for m, y in ys_sorted:
        y = max(y, prev_y + min_gap)
        adjusted.append((m, y))
        prev_y = y
    for m, y in adjusted:
        top_ax.text(last_x + 0.5, y, method_label[m],
                    color=method_color[m], fontsize=12, fontweight='bold',
                    va='center', ha='left')

    axes[-1].set_xlabel("Training time (Epochs × Phases)", fontsize=12)

    plt.subplots_adjust(left=0.13, right=0.94, top=0.92, bottom=0.10, hspace=0.35)
    plt.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    csv_path = os.path.join(HERE, 'ewc_rotated_mnist_data.csv')
    out_path = os.path.join(HERE, 'ewc_rotated_mnist_plot.png')
    plot_from_csv(csv_path, out_path)
    print(f"Saved plot -> {out_path}")
