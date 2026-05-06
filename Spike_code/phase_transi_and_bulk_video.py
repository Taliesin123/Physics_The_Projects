"""
Animation of the (lambda1, lambda2) phase diagram as rho varies linearly in
time, paired with a side panel showing the top two eigenvalues of the
two-spiked Wigner matrix as a function of rho. A vertical marker on the
right panel tracks the current rho.

Layout
------
+------------------------+------------------------+
|  RGB phase diagram     |  Top-two eigenvalues   |
|  (regenerated per rho) |  vs rho (static curve  |
|                        |  + moving rho marker)  |
+------------------------+------------------------+

Both panels reuse the parameter conventions of your existing scripts.

Output: an .mp4 file (falls back to .gif if ffmpeg is unavailable).
"""

import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import model_2 as m
importlib.reload(m)


# =========================================================================
# Parameters
# =========================================================================

# ---- Left panel: phase diagram sweep (matches phase_diagram script) -----
N_pd                = 100                           # spike dimension
lambas1             = np.linspace(0.1, 5, 50)
lambas2             = np.linspace(0.1, 5, 50)
N_gradient_descent  = 100
N_average           = 20
alpha               = np.sqrt(0.5)
OVERLAP_MAX         = 0.5

# ---- Right panel: eigenvalue curve (matches eigenvalue script) ----------
N_ev            = 500                               # matrix size
LAMBDA1_FIX     = 5.0                               # fixed lambda_1 for right panel
LAMBDA2_FIX     = 7.0                               # fixed lambda_2 for right panel
N_REALISATIONS  = 50
N_RHO_EV        = 100                               # rho samples for the curve

# ---- Animation -----------------------------------------------------------
N_FRAMES   = 40        # number of rho values used as animation frames
RHO_MIN    = 0.0
RHO_MAX    = 1.0
FPS        = 12        # video frame rate

# ---- Caching -------------------------------------------------------------
CACHE_PATH_PD = "./animation_phase_cache.npz"
CACHE_PATH_EV = "./animation_eigen_cache.npz"
USE_CACHE     = True

# ---- Output --------------------------------------------------------------
OUT_DIR  = "./Plots/phase_transi"
OUT_PATH = os.path.join(OUT_DIR, f"phase_evolution_alpha={alpha:.2f}.mp4")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================================
# Step 1 -- compute / load phase diagrams for every animation rho
# =========================================================================
rhos_anim = np.linspace(RHO_MIN, RHO_MAX, N_FRAMES)


def load_pd_cache():
    if not (USE_CACHE and os.path.exists(CACHE_PATH_PD)):
        return None
    data = np.load(CACHE_PATH_PD)
    needed = {"rhos", "lambas1", "lambas2", "o1", "o2"}
    if not needed.issubset(data.files):
        return None
    if not (np.array_equal(data["rhos"], rhos_anim)
            and np.array_equal(data["lambas1"], lambas1)
            and np.array_equal(data["lambas2"], lambas2)):
        return None
    expected = (N_FRAMES, len(lambas1), len(lambas2))
    if data["o1"].shape != expected or data["o2"].shape != expected:
        return None
    return data["o1"], data["o2"]


def save_pd_cache(o1, o2):
    if not USE_CACHE:
        return
    np.savez_compressed(
        CACHE_PATH_PD,
        rhos=rhos_anim, lambas1=lambas1, lambas2=lambas2, o1=o1, o2=o2,
    )


cached = load_pd_cache()
if cached is None:
    print(f"Computing phase diagrams for {N_FRAMES} rho values "
          f"(this is the slow step)...")
    o1_arr = np.zeros((N_FRAMES, len(lambas1), len(lambas2)))
    o2_arr = np.zeros_like(o1_arr)
    for i, rho in enumerate(rhos_anim):
        print(f"  [{i+1:3d}/{N_FRAMES}]  rho = {rho:.3f}")
        o1, o2 = m.phase_diagram_continuous(
            N_pd, rho, alpha, lambas1, lambas2,
            N_gradient_descent, N_average,
        )
        o1_arr[i] = o1
        o2_arr[i] = o2
    save_pd_cache(o1_arr, o2_arr)
    print("  cached.")
else:
    o1_arr, o2_arr = cached
    print("Loaded phase diagrams from cache.")


# =========================================================================
# Step 2 -- compute / load top-two-eigenvalue curve in rho
# =========================================================================
rhos_ev = np.linspace(0.0, 1.0, N_RHO_EV)


def load_ev_cache():
    if not (USE_CACHE and os.path.exists(CACHE_PATH_EV)):
        return None
    data = np.load(CACHE_PATH_EV)
    needed = {"rhos", "y1", "y2", "e1", "e2",
              "lambda1", "lambda2", "N", "n_real"}
    if not needed.issubset(data.files):
        return None
    same_params = (
        np.array_equal(data["rhos"], rhos_ev)
        and float(data["lambda1"]) == LAMBDA1_FIX
        and float(data["lambda2"]) == LAMBDA2_FIX
        and int(data["N"]) == N_ev
        and int(data["n_real"]) == N_REALISATIONS
    )
    if not same_params:
        return None
    return data["y1"], data["y2"], data["e1"], data["e2"]


def save_ev_cache(y1, y2, e1, e2):
    if not USE_CACHE:
        return
    np.savez_compressed(
        CACHE_PATH_EV,
        rhos=rhos_ev, y1=y1, y2=y2, e1=e1, e2=e2,
        lambda1=LAMBDA1_FIX, lambda2=LAMBDA2_FIX,
        N=N_ev, n_real=N_REALISATIONS,
    )


cached_ev = load_ev_cache()
if cached_ev is None:
    print(f"Computing top-two-eigenvalue curve "
          f"({N_RHO_EV} rho values, {N_REALISATIONS} realisations each)...")
    y1, y2, e1, e2 = [], [], [], []
    for j, rho in enumerate(rhos_ev):
        if j % 10 == 0:
            print(f"  [{j+1:3d}/{N_RHO_EV}]  rho = {rho:.3f}")
        a, b = [], []
        for _ in range(N_REALISATIONS):
            M = m.TwoSpikedWignerMatrix(
                N_ev, LAMBDA1_FIX, LAMBDA2_FIX, rho, alpha,
            )
            x1, x2 = M.double_max_eigenvalue()
            a.append(x1)
            b.append(x2)
        y1.append(np.mean(a)); y2.append(np.mean(b))
        e1.append(np.std(a));  e2.append(np.std(b))
    y1, y2, e1, e2 = map(np.asarray, (y1, y2, e1, e2))
    save_ev_cache(y1, y2, e1, e2)
    print("  cached.")
else:
    y1, y2, e1, e2 = cached_ev
    print("Loaded eigenvalue curve from cache.")


# =========================================================================
# Step 3 -- build the figure and the animation
# =========================================================================
fig, (ax_pd, ax_ev) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=120)


def make_rgb(o1, o2):
    """Build the same RGB image as the static phase-diagram script."""
    o1n = np.clip(o1 / OVERLAP_MAX, 0, 1).T
    o2n = np.clip(o2 / OVERLAP_MAX, 0, 1).T
    rgb = np.zeros((*o1n.shape, 3))
    rgb[..., 0] = o1n          # red   = m1 recovered
    rgb[..., 1] = o2n          # green = m2 recovered
    return rgb


# ---- Left panel ---------------------------------------------------------
im = ax_pd.imshow(
    make_rgb(o1_arr[0], o2_arr[0]),
    origin="lower",
    extent=[lambas1[0], lambas1[-1], lambas2[0], lambas2[-1]],
    aspect="auto",
    interpolation="bicubic",
)
ax_pd.set_xlabel(r"$\lambda_1$")
ax_pd.set_ylabel(r"$\lambda_2$")
title_pd = ax_pd.set_title(
    rf"Phase diagram  —  $\rho={rhos_anim[0]:.2f}$,  $\alpha={alpha:.2f}$"
)
legend_patches = [
    mpatches.Patch(facecolor=(0, 0, 0), edgecolor="white",
                   label="Neither recovered"),
    mpatches.Patch(facecolor=(1, 0, 0), edgecolor="white",
                   label=r"Only $m_1$ recovered"),
    mpatches.Patch(facecolor=(0, 1, 0), edgecolor="white",
                   label=r"Only $m_2$ recovered"),
    mpatches.Patch(facecolor=(1, 1, 0), edgecolor="white",
                   label="Both recovered"),
]
ax_pd.legend(handles=legend_patches, loc="upper left",
             framealpha=0.85, fontsize=8)

# ---- Right panel --------------------------------------------------------
ax_ev.errorbar(rhos_ev, y1, yerr=e1, capsize=2, color="C0", alpha=0.85,
               label=r"$\lambda_1$ outlier")
ax_ev.errorbar(rhos_ev, y2, yerr=e2, capsize=2, color="C1", alpha=0.85,
               label=r"$\lambda_2$ outlier")
ax_ev.axhline(2, color="red", linestyle="--", label="BBP threshold")
rho_marker = ax_ev.axvline(rhos_anim[0], color="black", linewidth=1.5,
                           label=rf"$\rho_\mathrm{{current}}$")
rho_dot1, = ax_ev.plot([rhos_anim[0]], [y1[0]], "o", color="C0",
                       markersize=8, markeredgecolor="black", zorder=5)
rho_dot2, = ax_ev.plot([rhos_anim[0]], [y2[0]], "o", color="C1",
                       markersize=8, markeredgecolor="black", zorder=5)
ax_ev.set_xlabel(r"$\rho$")
ax_ev.set_ylabel("Top two eigenvalues")
ax_ev.set_xlim(0, 1)
ax_ev.set_title(
    rf"Top two eigenvalues  —  $\lambda_1={LAMBDA1_FIX:g}$, "
    rf"$\lambda_2={LAMBDA2_FIX:g}$, $\alpha={alpha:.2f}$"
)
ax_ev.grid(alpha=0.3)
ax_ev.legend(loc="best", fontsize=8, framealpha=0.85)

fig.tight_layout()


def rho_to_curve(rho):
    """Interpolate the precomputed eigenvalue curve at an arbitrary rho."""
    return (np.interp(rho, rhos_ev, y1),
            np.interp(rho, rhos_ev, y2))


def update(frame):
    rho = rhos_anim[frame]
    im.set_data(make_rgb(o1_arr[frame], o2_arr[frame]))
    title_pd.set_text(
        rf"Phase diagram  —  $\rho={rho:.2f}$,  $\alpha={alpha:.2f}$"
    )
    rho_marker.set_xdata([rho, rho])
    yy1, yy2 = rho_to_curve(rho)
    rho_dot1.set_data([rho], [yy1])
    rho_dot2.set_data([rho], [yy2])
    return im, title_pd, rho_marker, rho_dot1, rho_dot2


anim = FuncAnimation(
    fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False,
)


# =========================================================================
# Step 4 -- save
# =========================================================================
print(f"\nWriting video to {OUT_PATH} ...")
try:
    writer = FFMpegWriter(fps=FPS, bitrate=2400)
    anim.save(OUT_PATH, writer=writer, dpi=120)
    print(f"Saved {OUT_PATH}")
except Exception as exc:
    print(f"FFMpeg writer failed ({exc!r}); falling back to GIF.")
    gif_path = OUT_PATH.replace(".mp4", ".gif")
    anim.save(gif_path, writer=PillowWriter(fps=FPS), dpi=120)
    print(f"Saved {gif_path}")

# Comment this out if you don't want a window to pop up at the end.
plt.show()