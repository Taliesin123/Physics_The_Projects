from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import model_2 as m
import importlib
import os

importlib.reload(m)

N = 50
lambas1 = np.linspace(0.1, 5, 100)
lambas2 = np.linspace(0.1, 5, 100)
N_gradient_descent = 100
N_average = 20

alphas = [np.sqrt(0.5)]
rhos = [0.0]

# Maximum meaningful per-spike overlap (used to normalize the two
# channels to [0, 1] before packing them into an RGB image).
OVERLAP_MAX = 0.5

save_dir = os.path.join(os.path.dirname(__file__), "..", "Plots", "Spike")
os.makedirs(save_dir, exist_ok=True)

for rho in rhos:
    for alpha in alphas:
        print(f"Running rho={rho:.2f}, alpha={alpha:.2f} ...")
        overlap1, overlap2 = m.phase_diagram_continuous(
            N, rho, alpha, lambas1, lambas2, N_gradient_descent, N_average
        )

        # Normalize each overlap to [0, 1] and transpose so that
        # axis 0 -> lambda2 (y) and axis 1 -> lambda1 (x).
        o1 = np.clip(overlap1 / OVERLAP_MAX, 0, 1).T
        o2 = np.clip(overlap2 / OVERLAP_MAX, 0, 1).T

        # Build an RGB image.
        #   R = overlap1  ->  red  = only spike 1 recovered
        #   G = overlap2  ->  green = only spike 2 recovered
        #   R + G         ->  yellow = both recovered
        #   0             ->  black  = neither recovered
        rgb = np.zeros((*o1.shape, 3))
        rgb[..., 0] = o1
        rgb[..., 1] = o2

        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        ax.imshow(
            rgb,
            origin="lower",
            extent=[lambas1[0], lambas1[-1], lambas2[0], lambas2[-1]],
            aspect="auto",
            interpolation="bicubic",
        )
        lam1_bbp = 1.0 / alpha
        lam2_bbp = 1.0 / np.sqrt(1.0 - alpha**2)

        ax.axvline(lam1_bbp, color="white", ls="--", lw=2, alpha=0.8,
                label=rf"$\lambda_1 = 1/\alpha \approx {lam1_bbp:.2f}$")
        ax.axhline(lam2_bbp, color="blue", ls="--",  lw=2, alpha=0.8,
                label=rf"$\lambda_2 = 1/\sqrt{{1-\alpha^2}} \approx {lam2_bbp:.2f}$")

        ax.set(xlabel=r"$\lambda_1$", ylabel=r"$\lambda_2$")
        ax.set_title(rf"Phase diagram  —  $\rho={rho:.2f}$,  $\alpha={alpha:.2f}$")

        legend_patches = [
            mpatches.Patch(facecolor=(0, 0, 0), edgecolor="white",
                           label="Neither recovered"),
            mpatches.Patch(facecolor=(1, 0, 0), edgecolor="white",
                           label=r"Only $m_1$ recovered"),
            mpatches.Patch(facecolor=(0, 1, 0), edgecolor="white",
                           label=r"Only $m_2$ recovered"),
            mpatches.Patch(facecolor=(1, 1, 0), edgecolor="white",
                           label="Both recovered"),
            Line2D([0], [0], color="white", ls="--", lw=1.2,
                label=rf"BBP $\lambda_1 = 1/\alpha \approx {lam1_bbp:.2f}$"),
            Line2D([0], [0], color="blue", ls="--",  lw=1.2,
           label=rf"BBP $\lambda_2 = 1/\sqrt{{1-\alpha^2}} \approx {lam2_bbp:.2f}$"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            framealpha=0.85,
            fontsize=8,
        )

        fig.tight_layout()
        fig.savefig(
            f"./Plots/phase_transi/Phase_Diagram_alpha={alpha:.2f}_rho={rho:.2f}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Done.")