"""
Extract the phase-transition boundary for several values of rho and overlay
all the boundaries on a single plot.

For each rho we sweep (lambda1, lambda2), compute the two recovery overlaps
exactly as in the original phase-diagram script, then define the
"recovery indicator"
        R(lambda1, lambda2) = max(overlap1, overlap2) / OVERLAP_MAX
The phase-transition line is the level set R = THRESHOLD, extracted with
matplotlib's contour finder. The result is one curve per rho on the same
(lambda1, lambda2) axes.

Optional: an "m1-only" and "m2-only" boundary can also be drawn by
thresholding overlap1 and overlap2 individually -- toggle DRAW_PER_SPIKE.
"""

import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import model_2 as m
importlib.reload(m)

# -------------------------------------------------------------------------
# Sweep parameters (same as the original phase-diagram script)
# -------------------------------------------------------------------------
N = 100
lambas1 = np.linspace(0.1, 5, 50)
lambas2 = np.linspace(0.1, 5, 50)
N_gradient_descent = 100
N_average = 20

alpha = np.sqrt(0.5)
rhos = [0.0, 0.1, 0.2, 0.6, 1.0]

# Same normalization constant used in the original script.
OVERLAP_MAX = 0.5
# Fraction of OVERLAP_MAX above which a spike is considered "recovered".
# 0.5 is a reasonable mid-point; tweak if your visual transition sits
# elsewhere.
THRESHOLD = 0.5

# If True, also draw the individual overlap1 / overlap2 transition curves
# (dashed). If False, only the combined "any-recovery" boundary is drawn.
DRAW_PER_SPIKE = False

# Cache the (expensive) sweep results so re-plotting is cheap.
CACHE_PATH = "./phase_transition_cache.npz"
USE_CACHE = True

save_dir = "./Plots/phase_transi"
os.makedirs(save_dir, exist_ok=True)


# -------------------------------------------------------------------------
# Run / load the sweeps
# -------------------------------------------------------------------------
def compute_all():
    """Return dict[rho] = (overlap1, overlap2) on the (lambda1, lambda2) grid."""
    results = {}
    for rho in rhos:
        print(f"Running rho={rho:.2f}, alpha={alpha:.2f} ...")
        o1, o2 = m.phase_diagram_continuous(
            N, rho, alpha, lambas1, lambas2, N_gradient_descent, N_average
        )
        results[rho] = (o1, o2)
        print("  Done.")
    return results


def cache_key(rho):
    return f"rho_{rho:.4f}"


def load_cache():
    if not (USE_CACHE and os.path.exists(CACHE_PATH)):
        return None
    data = np.load(CACHE_PATH)
    needed = {f"{cache_key(r)}_o1" for r in rhos} | {f"{cache_key(r)}_o2" for r in rhos}
    if not needed.issubset(set(data.files)):
        return None
    return {
        rho: (data[f"{cache_key(rho)}_o1"], data[f"{cache_key(rho)}_o2"])
        for rho in rhos
    }


def save_cache(results):
    if not USE_CACHE:
        return
    payload = {}
    for rho, (o1, o2) in results.items():
        payload[f"{cache_key(rho)}_o1"] = o1
        payload[f"{cache_key(rho)}_o2"] = o2
    np.savez_compressed(CACHE_PATH, **payload)


results = load_cache()
if results is None:
    results = compute_all()
    save_cache(results)
else:
    print(f"Loaded cached overlaps from {CACHE_PATH}")


# -------------------------------------------------------------------------
# Plot all the phase-transition lines on the same axes
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

# A perceptually ordered colour per rho.
colors = cm.viridis(np.linspace(0.05, 0.9, len(rhos)))

# We need (lambda1, lambda2) as 2-D meshgrids matching the orientation of
# the overlap arrays. In the original script the imshow used a transpose
# of the overlap arrays; here we work directly with the un-transposed
# arrays and pass (lambas1, lambas2) so that contour interprets axis 0 as
# lambda1 and axis 1 as lambda2.
L1, L2 = np.meshgrid(lambas1, lambas2, indexing="ij")

for rho, color in zip(rhos, colors):
    o1, o2 = results[rho]

    # Normalize to [0, 1] like in the original.
    o1n = np.clip(o1 / OVERLAP_MAX, 0, 1)
    o2n = np.clip(o2 / OVERLAP_MAX, 0, 1)

    # Combined "any-recovery" indicator.
    recovery = np.maximum(o1n, o2n)

    # contour returns silently if the level isn't crossed -- that's fine.
    ax.contour(
        L1, L2, recovery,
        levels=[THRESHOLD],
        colors=[color],
        linewidths=2.2,
    )

    if DRAW_PER_SPIKE:
        ax.contour(
            L1, L2, o1n, levels=[THRESHOLD],
            colors=[color], linewidths=1.0, linestyles="--",
        )
        ax.contour(
            L1, L2, o2n, levels=[THRESHOLD],
            colors=[color], linewidths=1.0, linestyles=":",
        )

    # Dummy line for the legend (contour has no native label support).
    ax.plot([], [], color=color, linewidth=2.2, label=rf"$\rho={rho:.2f}$")

if DRAW_PER_SPIKE:
    # Style legend explaining solid / dashed / dotted.
    ax.plot([], [], color="0.3", linewidth=2.2, label="any recovery")
    ax.plot([], [], color="0.3", linewidth=1.0, linestyle="--", label=r"$m_1$ recovery")
    ax.plot([], [], color="0.3", linewidth=1.0, linestyle=":",  label=r"$m_2$ recovery")

ax.set_xlim(lambas1[0], lambas1[-1])
ax.set_ylim(lambas2[0], lambas2[-1])
ax.set_xlabel(r"$\lambda_1$")
ax.set_ylabel(r"$\lambda_2$")
ax.set_title(
    rf"Phase-transition lines  —  $\alpha={alpha:.2f}$,  threshold = {THRESHOLD:.2f}"
)
ax.grid(alpha=0.25)
ax.legend(loc="best", framealpha=0.85, fontsize=9)

fig.tight_layout()
out_path = os.path.join(save_dir, f"Phase_Transition_Lines_alpha={alpha:.2f}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()