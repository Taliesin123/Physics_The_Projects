"""
Plot the EWC phase diagram produced by phase_diagram_ewc.py.

Three figures are produced :

    1. RGB phase diagrams   (R = overlap with x_1, G = overlap with x_2)
       for the three methods side-by-side :  naive spectral / Stage A /
       Stage B.

    2. Heatmap of the *gap*  overlap_1_stageB - overlap_1_naive
       on the (lambda_1, lambda_2) plane :  the region where EWC
       strictly outperforms the naive spectral method.

    3. Heatmap of the leakage  overlap_2_stageB - overlap_2_naive :
       negative = EWC leaks less into x_2 than naive does.

All three are saved to  Plots/Fisher/.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


OVERLAP_MAX = 0.5   # max meaningful per-channel overlap for the RGB panel


# -----------------------------------------------------------------------------
def _rgb(o1, o2, vmax=OVERLAP_MAX):
    r = np.clip(o1.T / vmax, 0, 1)
    g = np.clip(o2.T / vmax, 0, 1)
    rgb = np.zeros((*r.shape, 3))
    rgb[..., 0] = r
    rgb[..., 1] = g
    return rgb


def plot_rgb_side_by_side(data, save_path=None):
    lambas1 = data["lambas1"]; lambas2 = data["lambas2"]
    rho     = float(data["rho"])
    alpha   = float(data["alpha"])
    mu      = float(data["mu"])

    triples = [
        ("naive spectral",  data["naive_o1"], data["naive_o2"]),
        ("Stage A (Y1)",    data["A_o1"],     data["A_o2"]),
        (rf"Stage B (EWC, $\mu={mu}$)", data["B_o1"], data["B_o2"]),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), dpi=130)
    extent = [lambas1[0], lambas1[-1], lambas2[0], lambas2[-1]]

    for ax, (title, o1, o2) in zip(axs, triples):
        ax.imshow(_rgb(o1, o2), origin="lower", extent=extent,
                  aspect="auto", interpolation="bicubic")
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_1$")
        ax.axvline(1.0, color="white", ls=":", lw=0.7)
        ax.axhline(1.0, color="white", ls=":", lw=0.7)
    axs[0].set_ylabel(r"$\lambda_2$")

    legend_patches = [
        mpatches.Patch(facecolor=(0, 0, 0), edgecolor="white",
                       label="neither recovered"),
        mpatches.Patch(facecolor=(1, 0, 0), edgecolor="white",
                       label=r"only $x_1$ recovered"),
        mpatches.Patch(facecolor=(0, 1, 0), edgecolor="white",
                       label=r"only $x_2$ recovered"),
        mpatches.Patch(facecolor=(1, 1, 0), edgecolor="white",
                       label="both recovered"),
    ]
    axs[-1].legend(handles=legend_patches, loc="upper left",
                   fontsize=7, framealpha=0.85)

    fig.suptitle(rf"Phase diagram   ($N$={int(data['N'])},"
                 rf"  $\rho$={rho:.2f},  $\alpha$={alpha:.2f})")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
def plot_gap(data, save_path=None):
    """
    Heatmap of overlap-with-x_1 of Stage B minus naive spectral.
    Positive (blue) = EWC wins, negative (red) = EWC loses.
    """
    lambas1 = data["lambas1"]; lambas2 = data["lambas2"]
    gap = (data["B_o1"] - data["naive_o1"]).T
    vmax = max(0.05, float(np.max(np.abs(gap))))

    fig, ax = plt.subplots(figsize=(6.2, 5), dpi=130)
    im = ax.imshow(gap, origin="lower",
                   extent=[lambas1[0], lambas1[-1], lambas2[0], lambas2[-1]],
                   aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=+vmax, interpolation="bicubic")
    ax.set_xlabel(r"$\lambda_1$")
    ax.set_ylabel(r"$\lambda_2$")
    ax.set_title(r"Gap in overlap with $x_1$:   "
                 r"$|\langle \hat x_B, x_1\rangle| - "
                 r"|\langle \hat x_{\mathrm{naive}}, x_1\rangle|$"
                 f"   ($\\mu={float(data['mu'])}$)")
    ax.axvline(1.0, color="black", ls=":", lw=0.7)
    ax.axhline(1.0, color="black", ls=":", lw=0.7)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("overlap gap  (blue = EWC better)")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
def plot_leakage_reduction(data, save_path=None):
    """
    Heatmap of overlap-with-x_2 of Stage B minus naive spectral.
    Negative (blue) = EWC leaks LESS into x_2 than naive does = good.
    """
    lambas1 = data["lambas1"]; lambas2 = data["lambas2"]
    diff = (data["B_o2"] - data["naive_o2"]).T
    vmax = max(0.05, float(np.max(np.abs(diff))))

    fig, ax = plt.subplots(figsize=(6.2, 5), dpi=130)
    im = ax.imshow(diff, origin="lower",
                   extent=[lambas1[0], lambas1[-1], lambas2[0], lambas2[-1]],
                   aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=+vmax, interpolation="bicubic")
    ax.set_xlabel(r"$\lambda_1$")
    ax.set_ylabel(r"$\lambda_2$")
    ax.set_title(r"$x_2$ leakage:   "
                 r"$|\langle \hat x_B, x_2\rangle| - "
                 r"|\langle \hat x_{\mathrm{naive}}, x_2\rangle|$"
                 f"   ($\\mu={float(data['mu'])}$)")
    ax.axvline(1.0, color="black", ls=":", lw=0.7)
    ax.axhline(1.0, color="black", ls=":", lw=0.7)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("leakage change  (blue = EWC leaks less)")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "Data", "Fisher",
                             "phase_diagram_ewc.npz")
    plot_dir  = os.path.join(here, "..", "Plots", "Fisher")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found.  Run  phase_diagram_ewc.py  first.")

    data = np.load(data_path)

    print("1/3  RGB side-by-side ...")
    plot_rgb_side_by_side(
        data, save_path=os.path.join(plot_dir, "phase_diagram_rgb.png"))

    print("2/3  x_1 overlap gap ...")
    plot_gap(
        data, save_path=os.path.join(plot_dir, "phase_diagram_gap_x1.png"))

    print("3/3  x_2 leakage reduction ...")
    plot_leakage_reduction(
        data, save_path=os.path.join(plot_dir, "phase_diagram_leakage_x2.png"))

    print("done.")
