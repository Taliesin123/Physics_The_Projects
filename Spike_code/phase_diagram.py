import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import model_2 as m
import importlib
import os

importlib.reload(m)

N = 100        
lambas1 = np.linspace(0.1, 5, 100)  
lambas2 = np.linspace(0.1, 5, 100)
N_gradient_descent = 100
N_average = 20           

alphas = [0.5]
rhos = [0.0, 0.1, 0.2, 0.6, 1.0]


cmap = mcolors.LinearSegmentedColormap.from_list(
    "overlap_green",
    ["#1a1a2e", "#145a32", "#2DB87B", "#82ffb5"],
)

save_dir = os.path.join(os.path.dirname(__file__), "..", "Plots", "Spike")
os.makedirs(save_dir, exist_ok=True)

for rho in rhos:
    for alpha in alphas:
        print(f"Running rho={rho:.2f}, alpha={alpha:.2f} ...")
        overlap1, overlap2 = m.phase_diagram_continuous(
            N, rho, alpha, lambas1, lambas2, N_gradient_descent, N_average
        )

        combined = np.clip((overlap1 + overlap2) / 2, 0, 0.5).T

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(
            combined,
            origin="lower",
            extent=[lambas1[0], lambas1[-1], lambas2[0], lambas2[-1]],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=0.5,
            interpolation="bicubic",   # sharper-looking edges
        )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Average overlap")
        ax.set(xlabel=r"$\lambda_1$", ylabel=r"$\lambda_2$")
        ax.set_title(rf"Phase diagram  —  $\rho={rho:.2f}$,  $\alpha={alpha:.2f}$")

        fig.tight_layout()
        fig.savefig(f"./Plots/phase_transi/Phase_Diagram_alpha={alpha:.2f}_rho={rho:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Done.")