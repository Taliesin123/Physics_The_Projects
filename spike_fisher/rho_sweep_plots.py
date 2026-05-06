"""
Plot the rho-sweep produced by  rho_sweep.py.

Three figures :

    1. Overlaps with x_1 as a function of rho, all three methods on the
       same axes (with standard-error error bars).

    2. Overlap with x_2 as a function of rho  (the leakage panel).

    3. The "gap" plot :  overlap_x1(Stage B) - overlap_x1(naive)  vs rho.
       This is the single number that says "how much does EWC help, as
       a function of the correlation between the spikes?"

All three are saved to Plots/Fisher/.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def _mean_sem(arr: np.ndarray):
    """arr shape (n_rhos, n_repeats) -> means, SEMs along axis 1."""
    n = arr.shape[1]
    return arr.mean(axis=1), arr.std(axis=1) / np.sqrt(n)


# -----------------------------------------------------------------------------
def plot_overlap_vs_rho(data, target: str = "x1", save_path=None):
    """
    target = "x1"  -> overlaps with the first spike (recovery success).
    target = "x2"  -> overlaps with the second spike (leakage).
    """
    rhos = data["rhos"]
    if target == "x1":
        arrs = {"naive": data["naive_o1"],
                "Stage A": data["A_o1"],
                "Stage B (EWC)": data["B_o1"]}
        ylabel = r"$|\langle \hat x, x_1\rangle|$"
        title = r"Overlap with $x_1$  (success)"
    elif target == "x2":
        arrs = {"naive": data["naive_o2"],
                "Stage A": data["A_o2"],
                "Stage B (EWC)": data["B_o2"]}
        ylabel = r"$|\langle \hat x, x_2\rangle|$"
        title = r"Overlap with $x_2$  (leakage)"
    else:
        raise ValueError(target)

    colors = {"naive": "tab:blue", "Stage A": "tab:orange",
              "Stage B (EWC)": "tab:green"}
    markers = {"naive": "o", "Stage A": "s", "Stage B (EWC)": "^"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    for label, arr in arrs.items():
        m, s = _mean_sem(arr)
        ax.errorbar(rhos, m, yerr=s, label=label, marker=markers[label],
                    color=colors[label], capsize=3, lw=1.6)
    # reference line :  |<x_1, x_2>| = rho  (lower bound for x_2 leakage
    # of a perfect x_1 estimator)
    if target == "x2":
        ax.plot(rhos, rhos, color="gray", ls=":", lw=1.2,
                label=r"$\rho$  (true correlation)")
    ax.set_xlabel(r"$\rho$  (correlation between $x_1$ and $x_2$)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title +
                 rf"   ($N$={int(data['N'])}, "
                 rf"$\lambda_1=\lambda_2={float(data['lambda1'])}$, "
                 rf"$\mu={float(data['mu'])}$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
def plot_gap_vs_rho(data, save_path=None):
    rhos = data["rhos"]
    # per-repeat gap, then mean and SEM
    gap = data["B_o1"] - data["naive_o1"]
    m, s = _mean_sem(gap)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.fill_between(rhos, m - s, m + s, color="tab:purple", alpha=0.25)
    ax.plot(rhos, m, marker="D", color="tab:purple", lw=1.8,
            label=r"Stage B $-$ naive")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"overlap$_{x_1}$(EWC) $-$ overlap$_{x_1}$(naive)")
    ax.set_title(rf"How much EWC helps, as a function of $\rho$"
                 rf"   ($\mu={float(data['mu'])}$)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "Data", "Fisher", "rho_sweep.npz")
    plot_dir  = os.path.join(here, "..", "Plots", "Fisher")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run rho_sweep.py first.")
    data = np.load(data_path)

    print("1/3  overlap vs rho  (x_1) ...")
    plot_overlap_vs_rho(
        data, target="x1",
        save_path=os.path.join(plot_dir, "rho_sweep_overlap_x1.png"))

    print("2/3  overlap vs rho  (x_2 leakage) ...")
    plot_overlap_vs_rho(
        data, target="x2",
        save_path=os.path.join(plot_dir, "rho_sweep_overlap_x2.png"))

    print("3/3  gap vs rho ...")
    plot_gap_vs_rho(
        data, save_path=os.path.join(plot_dir, "rho_sweep_gap.png"))

    print("done.")
