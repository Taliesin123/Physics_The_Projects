"""
Plot the Fisher-vs-L2 ablation produced by  l2_comparison.py.

Three figures :

    1. One panel per rho :  overlap-with-x_1 vs mu, Fisher (anisotropic)
       vs L2 (isotropic), with naive spectral as a horizontal reference.

    2. "Fisher minus L2" gap vs mu, one curve per rho on the same axes.
       If the curves climb with rho, Fisher's anisotropy is earning its
       keep.

    3. Best-achievable overlap vs rho :  for each rho we take the best
       mu for Fisher and for L2 separately, and compare them.  This is
       the "best-case" story.
"""

from __future__ import annotations

import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def _mean_sem(arr: np.ndarray, axis: int = 1):
    n = arr.shape[axis]
    return arr.mean(axis=axis), arr.std(axis=axis) / np.sqrt(n)


def _load_all(data_dir):
    """Return list of (rho, data-dict) sorted by rho."""
    files = sorted(glob.glob(os.path.join(data_dir,
                                          "l2_comparison_rho*.npz")))
    out = []
    for f in files:
        d = np.load(f)
        out.append((float(d["rho"]), d))
    out.sort(key=lambda t: t[0])
    return out


# -----------------------------------------------------------------------------
def plot_overlap_vs_mu_panels(datasets, save_path=None):
    n_rho = len(datasets)
    fig, axs = plt.subplots(1, n_rho, figsize=(4.5 * n_rho, 4.3), dpi=130,
                            sharey=True)
    if n_rho == 1:
        axs = [axs]

    for ax, (rho, d) in zip(axs, datasets):
        mus = d["mus"]
        mF, sF = _mean_sem(d["fisher_o1"])
        mL, sL = _mean_sem(d["l2_o1"])
        mN, sN = _mean_sem(d["naive_o1"])

        ax.errorbar(mus, mF, yerr=sF, marker="^", color="tab:green",
                    label="Fisher (EWC)", capsize=3, lw=1.6)
        ax.errorbar(mus, mL, yerr=sL, marker="s", color="tab:orange",
                    label=r"isotropic $L^2$", capsize=3, lw=1.6)
        # naive is horizontal (doesn't depend on mu)
        ax.axhline(mN.mean(), color="tab:blue", ls="--",
                   label="naive spectral")

        ax.set_xscale("symlog", linthresh=0.2)
        ax.set_xlabel(r"$\mu$")
        ax.set_title(rf"$\rho = {rho:.2f}$")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    axs[0].set_ylabel(r"$|\langle \hat x, x_1\rangle|$")
    fig.suptitle("Fisher (anisotropic) vs isotropic $L^2$ anchor at "
                 r"$\hat x_A^*$")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
def plot_gap_vs_mu(datasets, save_path=None):
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=130)
    cmap = plt.get_cmap("viridis")
    for k, (rho, d) in enumerate(datasets):
        mus = d["mus"]
        gap = d["fisher_o1"] - d["l2_o1"]   # per-repeat gap
        m, s = _mean_sem(gap)
        color = cmap(k / max(1, len(datasets) - 1))
        ax.errorbar(mus, m, yerr=s, label=rf"$\rho={rho:.2f}$",
                    marker="o", color=color, capsize=3, lw=1.6)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xscale("symlog", linthresh=0.2)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"overlap$_{x_1}$(Fisher) $-$ overlap$_{x_1}$($L^2$)")
    ax.set_title("Where does the *anisotropy* of Fisher matter?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
def plot_best_vs_rho(datasets, save_path=None):
    rhos, best_fisher, best_l2, naive = [], [], [], []
    for rho, d in datasets:
        mF, _ = _mean_sem(d["fisher_o1"])
        mL, _ = _mean_sem(d["l2_o1"])
        mN, _ = _mean_sem(d["naive_o1"])
        rhos.append(rho)
        best_fisher.append(float(mF.max()))
        best_l2.append(float(mL.max()))
        naive.append(float(mN.mean()))

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)
    ax.plot(rhos, best_fisher, marker="^", color="tab:green",
            label=r"best Fisher (over $\mu$)", lw=1.8)
    ax.plot(rhos, best_l2, marker="s", color="tab:orange",
            label=r"best $L^2$ (over $\mu$)", lw=1.8)
    ax.plot(rhos, naive, marker="o", color="tab:blue",
            label="naive spectral", lw=1.8)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$|\langle \hat x, x_1\rangle|$")
    ax.set_title(r"Best achievable overlap vs $\rho$  "
                 r"(best $\mu$ for each method)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
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
    data_dir = os.path.join(here, "..", "Data", "Fisher")
    plot_dir = os.path.join(here, "..", "Plots", "Fisher")

    datasets = _load_all(data_dir)
    if not datasets:
        raise FileNotFoundError(
            f"No l2_comparison_rho*.npz in {data_dir}.  "
            f"Run l2_comparison.py first.")
    print(f"Loaded {len(datasets)} rho values: "
          f"{[f'{r:.2f}' for r, _ in datasets]}")

    print("1/3  overlap-vs-mu panels ...")
    plot_overlap_vs_mu_panels(
        datasets,
        save_path=os.path.join(plot_dir, "l2_vs_fisher_panels.png"))

    print("2/3  Fisher - L2 gap vs mu ...")
    plot_gap_vs_mu(
        datasets,
        save_path=os.path.join(plot_dir, "l2_vs_fisher_gap.png"))

    print("3/3  best overlap vs rho ...")
    plot_best_vs_rho(
        datasets,
        save_path=os.path.join(plot_dir, "l2_vs_fisher_best_vs_rho.png"))

    print("done.")
