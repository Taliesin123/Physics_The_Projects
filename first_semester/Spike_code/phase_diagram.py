"""
Phase-diagram script split into two stages:

    1. compute_results(...) -> populates / reuses an in-memory cache so the
       expensive `m.phase_diagram_continuous` call is only run once per
       (rho, alpha) combination. Optionally persists the cache to disk so
       results survive a Python restart.

    2. plot_results(...) -> reads from the cache and only does the plotting,
       so you can tweak aesthetics and re-run as many times as you want
       without recomputing anything.

Typical workflow in an interactive session (IPython / Jupyter):

    from phase_diagram import compute_results, plot_results, RESULTS

    compute_results()        # heavy, do this once
    plot_results()           # cheap, re-run after editing aesthetics

If you change a plotting parameter (colors, fontsize, OVERLAP_MAX, ...),
just edit `plot_results` and call it again — `RESULTS` stays in memory.
"""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import model_2 as m
import importlib
import os
import pickle

importlib.reload(m)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 300
lambas1 = np.linspace(0.1, 5, 100)
lambas2 = np.linspace(0.1, 5, 100)
N_gradient_descent = 100
N_average = 20

alphas = [np.sqrt(0.5)]
rhos = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]

# Maximum meaningful per-spike overlap (used to normalize the two
# channels to [0, 1] before packing them into an RGB image).
OVERLAP_MAX = 0.5

save_dir = os.path.join(os.path.dirname(__file__), "..", "Plots", "Spike")
os.makedirs(save_dir, exist_ok=True)

# Where to optionally persist the cache between Python sessions.
CACHE_FILE = os.path.join(os.path.dirname(__file__), "phase_diagram_cache.pkl")


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
# Maps (rho, alpha, N, len(lambas1), len(lambas2), N_gradient_descent, N_average)
# -> (overlap1, overlap2). Including the grid params in the key prevents
# silently reusing results computed with a different grid.
RESULTS = {}


def _cache_key(rho, alpha):
    return (
        float(rho),
        float(alpha),
        int(N),
        int(len(lambas1)),
        int(len(lambas2)),
        int(N_gradient_descent),
        int(N_average),
    )


def load_cache(path=CACHE_FILE):
    """Load a previously-saved cache from disk into RESULTS (if present)."""
    global RESULTS
    if os.path.exists(path):
        with open(path, "rb") as f:
            RESULTS = pickle.load(f)
        print(f"Loaded {len(RESULTS)} cached entries from {path}")
    else:
        print(f"No cache file at {path}; starting fresh.")
    return RESULTS


def save_cache(path=CACHE_FILE):
    """Persist the current RESULTS dict to disk."""
    with open(path, "wb") as f:
        pickle.dump(RESULTS, f)
    print(f"Saved {len(RESULTS)} cached entries to {path}")


# ---------------------------------------------------------------------------
# Stage 1: computation (only does work for missing entries)
# ---------------------------------------------------------------------------
def compute_results(force=False, persist=False):
    """
    Fill RESULTS for every (rho, alpha) in the configured grids.

    Parameters
    ----------
    force : bool
        If True, recompute even when an entry is already cached.
    persist : bool
        If True, save the cache to disk after computing.
    """
    for rho in rhos:
        for alpha in alphas:
            key = _cache_key(rho, alpha)
            if not force and key in RESULTS:
                print(f"Skipping rho={rho:.2f}, alpha={alpha:.2f} (cached).")
                continue

            print(f"Running rho={rho:.2f}, alpha={alpha:.2f} ...")
            overlap1, overlap2 = m.phase_diagram_continuous(
                N, rho, alpha, lambas1, lambas2, N_gradient_descent, N_average
            )
            RESULTS[key] = (overlap1, overlap2)
            print("  Done.")

    if persist:
        save_cache()

    return RESULTS


# ---------------------------------------------------------------------------
# Stage 2: plotting (cheap; only reads from RESULTS)
# ---------------------------------------------------------------------------
def plot_results(overlap_max=None, save=True, show=False):
    """
    Render every cached (rho, alpha) result. Re-run this freely after
    tweaking aesthetics — it does not recompute anything.

    Parameters
    ----------
    overlap_max : float, optional
        Override the module-level OVERLAP_MAX without editing the file.
    save : bool
        If True, write a PNG to ./Plots/phase_transi/.
    show : bool
        If True, leave figures open instead of closing them (useful in
        notebooks).
    """
    omax = OVERLAP_MAX if overlap_max is None else overlap_max
    figs = {}

    for rho in rhos:
        for alpha in alphas:
            key = _cache_key(rho, alpha)
            if key not in RESULTS:
                print(
                    f"No cached result for rho={rho:.2f}, alpha={alpha:.2f}; "
                    "call compute_results() first."
                )
                continue

            overlap1, overlap2 = RESULTS[key]

            # Normalize each overlap to [0, 1] and transpose so that
            # axis 0 -> lambda2 (y) and axis 1 -> lambda1 (x).
            o1 = np.clip(overlap1 / omax, 0, 1).T
            o2 = np.clip(overlap2 / omax, 0, 1).T

            # Build an RGB image.
            #   R = overlap1  ->  red    = only spike 1 recovered
            #   G = overlap2  ->  green  = only spike 2 recovered
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

            ax.axvline(lam1_bbp, color="black", ls="--", lw=2, alpha=0.8,
                       label=rf"BBP threshold for $\lambda_1$")
            ax.axhline(lam2_bbp, color="blue", ls="--", lw=2, alpha=0.8,
                       label=rf"BBP threshold for $\lambda_2$")

            ax.set_xlabel(r"$\lambda_1$", fontsize=17)
            ax.set_ylabel(r"$\lambda_2$", fontsize=17)
            ax.tick_params(axis='both', which='major', labelsize=17)

            legend_patches = [
                mpatches.Patch(facecolor=(0, 0, 0), edgecolor="white",
                               label="Neither recovered"),
                mpatches.Patch(facecolor=(1, 0, 0), edgecolor="white",
                               label=r"Only $m_1$ recovered"),
                mpatches.Patch(facecolor=(0, 1, 0), edgecolor="white",
                               label=r"Only $m_2$ recovered"),
                mpatches.Patch(facecolor=(1, 1, 0), edgecolor="white",
                               label="Both recovered"),
                #Line2D([0], [0], color="black", ls="--", lw=1.2, label=rf"BBP threshold for $\lambda_1$"),
                #Line2D([0], [0], color="blue", ls="--", lw=1.2, label=rf"BBP threshold for $\lambda_2$"),
            ]
            ax.legend(
                handles=legend_patches,
                loc="upper left",
                framealpha=0.8,
                fontsize=17,
                borderaxespad=0,
            )

            fig.tight_layout()

            if save:
                out_dir = "./Plots/phase_transi"
                os.makedirs(out_dir, exist_ok=True)
                fig.savefig(
                    f"{out_dir}/Phase_Diagram_alpha={alpha:.2f}_rho={rho:.2f}.png",
                    dpi=200,
                    bbox_inches="tight",
                )

            figs[(rho, alpha)] = fig
            if not show:
                plt.close(fig)

    return figs


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Uncomment to reuse results across Python sessions:
    # load_cache()

    compute_results()
    plot_results()

    # Uncomment to persist for next time:
    save_cache()