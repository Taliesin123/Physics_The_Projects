"""
Animation of the (lambda1, lambda2) phase diagram as rho varies linearly in
time, paired with a side panel showing the eigenvalue distribution of the
two-spiked Wigner matrix: Wigner-semicircle bulk plus two outlier spikes
that slide as rho advances.

Layout
------
+------------------------+------------------------+
|  RGB phase diagram     |  Eigenvalue density    |
|  (regenerated per rho) |  - bulk histogram      |
|                        |  - Wigner semicircle   |
|                        |  - BBP line at lambda=2|
|                        |  - two moving spikes   |
+------------------------+------------------------+

Output: an .mp4 (falls back to .gif if ffmpeg is unavailable).
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

# ---- Left panel: phase diagram sweep ------------------------------------
N_pd                = 100
lambas1             = np.linspace(0.1, 5, 50)
lambas2             = np.linspace(0.1, 5, 50)
N_gradient_descent  = 100
N_average           = 20
alpha               = np.sqrt(0.5)
OVERLAP_MAX         = 0.5

# ---- Right panel: eigenvalue density / spikes ---------------------------
LAMBDA1_FIX     = 5.0          # signal strength of spike 1
LAMBDA2_FIX     = 7.0          # signal strength of spike 2
N_REALISATIONS  = 50           # for the (rho -> top-2 eigenvalues) curve
N_RHO_EV        = 100          # number of rho samples in that curve
N_BULK          = 1500         # matrix size for the static bulk histogram
BBP_THRESHOLD   = 2.0          # right edge of the Wigner semicircle
SEMICIRCLE_R    = 2.0          # semicircle radius (= BBP threshold)

# ---- Animation -----------------------------------------------------------
N_FRAMES = 40
RHO_MIN  = 0.0
RHO_MAX  = 1.0
FPS      = 12

# ---- Caching -------------------------------------------------------------
CACHE_PATH_PD   = "./animation_phase_cache.npz"
CACHE_PATH_EV   = "./animation_eigen_cache.npz"
CACHE_PATH_BULK = "./animation_bulk_cache.npz"
USE_CACHE       = True

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
              "lambda1", "lambda2", "n_real"}
    if not needed.issubset(data.files):
        return None
    same = (
        np.array_equal(data["rhos"], rhos_ev)
        and float(data["lambda1"]) == LAMBDA1_FIX
        and float(data["lambda2"]) == LAMBDA2_FIX
        and int(data["n_real"]) == N_REALISATIONS
    )
    if not same:
        return None
    return data["y1"], data["y2"], data["e1"], data["e2"]


def save_ev_cache(y1, y2, e1, e2):
    if not USE_CACHE:
        return
    np.savez_compressed(
        CACHE_PATH_EV,
        rhos=rhos_ev, y1=y1, y2=y2, e1=e1, e2=e2,
        lambda1=LAMBDA1_FIX, lambda2=LAMBDA2_FIX,
        n_real=N_REALISATIONS,
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
                N_BULK if False else 500,  # match the original eigenvalue script's N
                LAMBDA1_FIX, LAMBDA2_FIX, rho, alpha,
            )
            x1, x2 = M.double_max_eigenvalue()
            a.append(x1); b.append(x2)
        y1.append(np.mean(a)); y2.append(np.mean(b))
        e1.append(np.std(a));  e2.append(np.std(b))
    y1, y2, e1, e2 = map(np.asarray, (y1, y2, e1, e2))
    save_ev_cache(y1, y2, e1, e2)
    print("  cached.")
else:
    y1, y2, e1, e2 = cached_ev
    print("Loaded eigenvalue curve from cache.")


# =========================================================================
# Step 3 -- compute / load bulk eigenvalues (one large realisation)
# =========================================================================
def extract_matrix(M_obj):
    """Best-effort extraction of the underlying numpy matrix."""
    for attr in ("matrix", "M", "A", "Y", "H"):
        if hasattr(M_obj, attr):
            cand = getattr(M_obj, attr)
            arr = np.asarray(cand)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                return arr
    return None


def load_bulk_cache():
    if not (USE_CACHE and os.path.exists(CACHE_PATH_BULK)):
        return None
    data = np.load(CACHE_PATH_BULK)
    if "bulk_eigs" not in data.files:
        return None
    if int(data.get("N_BULK", -1)) != N_BULK:
        return None
    if float(data.get("alpha", -1)) != float(alpha):
        return None
    return data["bulk_eigs"]


def save_bulk_cache(bulk_eigs):
    if not USE_CACHE:
        return
    np.savez_compressed(
        CACHE_PATH_BULK,
        bulk_eigs=bulk_eigs, N_BULK=N_BULK, alpha=alpha,
    )


bulk_eigs = load_bulk_cache()
if bulk_eigs is None:
    print(f"Computing bulk eigenvalues (one realisation, N={N_BULK}) ...")
    M_bulk = m.TwoSpikedWignerMatrix(
        N_BULK, LAMBDA1_FIX, LAMBDA2_FIX, 0.5, alpha,
    )
    mat = extract_matrix(M_bulk)
    if mat is not None:
        eigs_full = np.linalg.eigvalsh(mat)
    else:
        # Fallback: build a pure GOE-style Wigner matrix with bulk on
        # [-SEMICIRCLE_R, +SEMICIRCLE_R].
        print("  (couldn't read .matrix from TwoSpikedWignerMatrix; "
              "using a manually constructed Wigner noise matrix instead)")
        W = np.random.randn(N_BULK, N_BULK) / np.sqrt(N_BULK)
        W = (W + W.T) / np.sqrt(2)
        eigs_full = np.linalg.eigvalsh(W)
    # Strip the two outliers; the remaining N-2 eigenvalues are the bulk.
    bulk_eigs = np.sort(eigs_full)[:-2]
    save_bulk_cache(bulk_eigs)
    print("  cached.")
else:
    print("Loaded bulk eigenvalues from cache.")


# =========================================================================
# Step 4 -- build the figure and the animation
# =========================================================================
def wigner_semicircle_pdf(x, R=SEMICIRCLE_R):
    """Wigner semicircle density supported on [-R, R]."""
    pdf = np.zeros_like(x, dtype=float)
    inside = np.abs(x) < R
    pdf[inside] = 2.0 * np.sqrt(R * R - x[inside] ** 2) / (np.pi * R * R)
    return pdf


fig, (ax_pd, ax_ev) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=120)


def make_rgb(o1, o2):
    o1n = np.clip(o1 / OVERLAP_MAX, 0, 1).T
    o2n = np.clip(o2 / OVERLAP_MAX, 0, 1).T
    rgb = np.zeros((*o1n.shape, 3))
    rgb[..., 0] = o1n
    rgb[..., 1] = o2n
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


# ---- Right panel: eigenvalue density ------------------------------------
spike_max = max(np.max(y1), np.max(y2))
x_min = -SEMICIRCLE_R - 0.6
x_max = max(SEMICIRCLE_R, spike_max) + 1.0

# Static bulk histogram (density).
ax_ev.hist(
    bulk_eigs, bins=80, range=(-SEMICIRCLE_R - 0.2, SEMICIRCLE_R + 0.2),
    density=True, color="lightsteelblue", edgecolor="steelblue",
    alpha=0.75, label="Empirical bulk",
)

# Static Wigner semicircle.
x_curve = np.linspace(-SEMICIRCLE_R, SEMICIRCLE_R, 400)
ax_ev.plot(x_curve, wigner_semicircle_pdf(x_curve),
           color="black", linewidth=2.0, label="Wigner semicircle")

# BBP threshold line.
ax_ev.axvline(BBP_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
              label=rf"BBP threshold ($\lambda={BBP_THRESHOLD:g}$)")

# Top of the semicircle, used to draw spike lines that span the panel.
ymax = wigner_semicircle_pdf(np.array([0.0]))[0] * 1.25
ax_ev.set_ylim(0, ymax)

# Moving spike lines.
spike1 = ax_ev.axvline(y1[0], color="C0", linewidth=2.5,
                       label=rf"spike 1  (initial $\lambda_1={LAMBDA1_FIX:g}$)")
spike2 = ax_ev.axvline(y2[0], color="C1", linewidth=2.5,
                       label=rf"spike 2  (initial $\lambda_2={LAMBDA2_FIX:g}$)")

# Text annotations that follow the spikes.
spike1_txt = ax_ev.text(
    y1[0], ymax * 0.92, f"{y1[0]:.2f}",
    color="C0", ha="center", va="bottom", fontsize=9,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
              edgecolor="C0", alpha=0.85),
)
spike2_txt = ax_ev.text(
    y2[0], ymax * 0.78, f"{y2[0]:.2f}",
    color="C1", ha="center", va="bottom", fontsize=9,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
              edgecolor="C1", alpha=0.85),
)

ax_ev.set_xlim(x_min, x_max)
ax_ev.set_xlabel(r"eigenvalue $\lambda$")
ax_ev.set_ylabel("density")
title_ev = ax_ev.set_title(
    rf"Eigenvalue distribution  —  $\lambda_1={LAMBDA1_FIX:g}$, "
    rf"$\lambda_2={LAMBDA2_FIX:g}$, $\alpha={alpha:.2f}$,  "
    rf"$\rho={rhos_anim[0]:.2f}$"
)
ax_ev.legend(loc="upper right", fontsize=8, framealpha=0.85)
ax_ev.grid(alpha=0.25)

fig.tight_layout()


def rho_to_spikes(rho):
    return (np.interp(rho, rhos_ev, y1),
            np.interp(rho, rhos_ev, y2))


def update(frame):
    rho = rhos_anim[frame]

    # left
    im.set_data(make_rgb(o1_arr[frame], o2_arr[frame]))
    title_pd.set_text(
        rf"Phase diagram  —  $\rho={rho:.2f}$,  $\alpha={alpha:.2f}$"
    )

    # right
    s1, s2 = rho_to_spikes(rho)
    spike1.set_xdata([s1, s1])
    spike2.set_xdata([s2, s2])
    spike1_txt.set_position((s1, ymax * 0.92))
    spike1_txt.set_text(f"{s1:.2f}")
    spike2_txt.set_position((s2, ymax * 0.78))
    spike2_txt.set_text(f"{s2:.2f}")
    title_ev.set_text(
        rf"Eigenvalue distribution  —  $\lambda_1={LAMBDA1_FIX:g}$, "
        rf"$\lambda_2={LAMBDA2_FIX:g}$, $\alpha={alpha:.2f}$,  "
        rf"$\rho={rho:.2f}$"
    )

    return (im, title_pd, spike1, spike2,
            spike1_txt, spike2_txt, title_ev)


anim = FuncAnimation(
    fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False,
)


# =========================================================================
# Step 5 -- save
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

plt.show()