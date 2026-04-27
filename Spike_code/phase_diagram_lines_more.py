"""
Two extra phase-transition figures, built from the same overlap sweeps as
phase_transition_lines.py (and sharing its on-disk cache).

Figure 1  -- "single-spike" boundaries:
    For each rho, draw the contour enclosing the region where exactly one
    spike is recovered:
        only m1 :  o1 > THRESHOLD  AND  o2 < THRESHOLD
        only m2 :  o2 > THRESHOLD  AND  o1 < THRESHOLD
    The boundary is found as the level set of a smooth indicator
        f_m1_only = min(o1, 1 - o2)        (high inside the m1-only region)
        f_m2_only = min(o2, 1 - o1)        (high inside the m2-only region)
    contoured at THRESHOLD. m1-only is solid, m2-only is dashed, color = rho.

Figure 2  -- "neither recovered" boundary:
    For each rho, draw the boundary of the region where both overlaps are
    below THRESHOLD. Smooth indicator
        f_neither = 1 - max(o1, o2)
    contoured at THRESHOLD. Color = rho.

The two figures share rho colors, so a curve's color always means the same
thing across plots.
"""

import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import model_2 as m
importlib.reload(m)

# -------------------------------------------------------------------------
# Sweep parameters (must match phase_transition_lines.py for cache reuse)
# -------------------------------------------------------------------------
N = 1000
lambas1 = np.linspace(0.1, 5, 100)
lambas2 = np.linspace(0.1, 5, 100)
N_gradient_descent = 100
N_average = 20

alpha = np.sqrt(0.5)
rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]

OVERLAP_MAX = 0.5
THRESHOLD = 0.5

CACHE_PATH = "./phase_transition_cache.npz"
USE_CACHE = True

save_dir = "./Plots/phase_transi"
os.makedirs(save_dir, exist_ok=True)


# -------------------------------------------------------------------------
# Run / load the sweeps (shares the cache file with phase_transition_lines.py)
# -------------------------------------------------------------------------
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


def compute_all():
    results = {}
    for rho in rhos:
        print(f"Running rho={rho:.2f}, alpha={alpha:.2f} ...")
        o1, o2 = m.phase_diagram_continuous(
            N, rho, alpha, lambas1, lambas2, N_gradient_descent, N_average
        )
        results[rho] = (o1, o2)
        print("  Done.")
    return results


results = load_cache()
if results is None:
    results = compute_all()
    save_cache(results)
else:
    print(f"Loaded cached overlaps from {CACHE_PATH}")


# -------------------------------------------------------------------------
# Common setup
# -------------------------------------------------------------------------
colors = cm.viridis(np.linspace(0.05, 0.9, len(rhos)))
L1, L2 = np.meshgrid(lambas1, lambas2, indexing="ij")


def normalized(rho):
    o1, o2 = results[rho]
    o1n = np.clip(o1 / OVERLAP_MAX, 0, 1)
    o2n = np.clip(o2 / OVERLAP_MAX, 0, 1)
    return o1n, o2n


# =========================================================================
# Figure 1 -- single-spike-only boundaries
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)

for rho, color in zip(rhos, colors):
    o1n, o2n = normalized(rho)

    # Smooth indicator for "only m1": high where o1 is high AND o2 is low.
    f_m1_only = np.minimum(o1n, 1.0 - o2n)
    # Smooth indicator for "only m2": high where o2 is high AND o1 is low.
    f_m2_only = np.minimum(o2n, 1.0 - o1n)

    ax1.contour(
        L1, L2, f_m1_only,
        levels=[THRESHOLD], colors=[color],
        linewidths=2.0, linestyles="-",
    )
    ax1.contour(
        L1, L2, f_m2_only,
        levels=[THRESHOLD], colors=[color],
        linewidths=2.0, linestyles="--",
    )

    # Legend entry per rho.
    ax1.plot([], [], color=color, linewidth=2.0, label=rf"$\rho={rho:.2f}$")

# Style legend entries explaining solid vs dashed.
ax1.plot([], [], color="0.3", linewidth=2.0, linestyle="-",
         label=r"only $m_1$ recovered")
ax1.plot([], [], color="0.3", linewidth=2.0, linestyle="--",
         label=r"only $m_2$ recovered")

ax1.set_xlim(lambas1[0], lambas1[-1])
ax1.set_ylim(lambas2[0], lambas2[-1])
ax1.set_xlabel(r"$\lambda_1$")
ax1.set_ylabel(r"$\lambda_2$")
ax1.set_title(
    rf"Single-spike recovery boundaries  —  $\alpha={alpha:.2f}$,  "
    rf"threshold = {THRESHOLD:.2f}"
)
ax1.grid(alpha=0.25)
ax1.legend(loc="best", framealpha=0.85, fontsize=9, ncol=2)

fig1.tight_layout()
out1 = os.path.join(save_dir, f"Phase_Transition_OneSpikeOnly_alpha={alpha:.2f}.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight")
print(f"Saved {out1}")


# =========================================================================
# Figure 2 -- "neither recovered" boundary
# =========================================================================
fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)

for rho, color in zip(rhos, colors):
    o1n, o2n = normalized(rho)

    # Smooth indicator for "neither recovered": high where BOTH overlaps
    # are small.
    f_neither = 1.0 - np.maximum(o1n, o2n)

    ax2.contour(
        L1, L2, f_neither,
        levels=[THRESHOLD], colors=[color],
        linewidths=2.2, linestyles="-",
    )
    ax2.plot([], [], color=color, linewidth=2.2, label=rf"$\rho={rho:.2f}$")

ax2.set_xlim(lambas1[0], lambas1[-1])
ax2.set_ylim(lambas2[0], lambas2[-1])
ax2.set_xlabel(r"$\lambda_1$")
ax2.set_ylabel(r"$\lambda_2$")
ax2.set_title(
    rf"Neither-recovered boundary  —  $\alpha={alpha:.2f}$,  "
    rf"threshold = {THRESHOLD:.2f}"
)
ax2.grid(alpha=0.25)
ax2.legend(loc="best", framealpha=0.85, fontsize=9)

fig2.tight_layout()
out2 = os.path.join(save_dir, f"Phase_Transition_Neither_alpha={alpha:.2f}.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved {out2}")

plt.show()