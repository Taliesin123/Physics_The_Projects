"""WITH the noise bulk: Y = P + W, W Wigner.

Theory (report, eq. bbp): outliers of Y sit at theta + 1/theta when theta > 1 (else at the
bulk edge 2); the top eigenvector x_hat of Y overlaps the eigenvector v_+ of P with
sqrt(1 - theta_+^-2), hence  |<x_hat, x_i>| -> sqrt(1 - theta_+^-2) |<v_+, x_i>|.

Figures (saved next to this file):
  eig   : eigenvalues_Y.png       top-2 eigenvalues of Y vs a, b, rho (+ bulk edge)
  lam   : overlaps_vs_lambda.png  |<u_pm, v_pm>| vs lambda (lambda1 = lambda2 = lambda)
  heat  : heat_lambda1_rho.png    |<x_hat,x1>|, |<x_hat,x2>| vs (lambda1, rho), lambda2 fixed
          heat_a_rho.png          same vs (a, rho), b fixed
  phase : phase_rho=*.png         RGB phase diagrams in the (a, b) plane, one per rho
Run:  python check_bulk.py [eig lam heat phase]      (no argument = all)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from spike_lib import TwoSpikes

ALPHA = np.sqrt(0.5)


# ---------------------------------------------------------------- theory
def theta_pm(a, b, rho):
    d = np.sqrt((a - b) ** 2 / 4 + a * b * rho**2)
    return (a + b) / 2 + d, (a + b) / 2 - d


def bbp(theta):
    return np.where(theta > 1, theta + 1 / np.maximum(theta, 1e-12), 2.0)


def theory_overlaps(a, b, rho):
    """Predicted |<x_hat, x1>|, |<x_hat, x2>| for the top eigenvector x_hat of Y.

    v_+ = X u with u the top eigenvector of K G (2x2), see report; then
    <v_+, x_i> = (G u)_i / sqrt(u^T G u), damped by the BBP factor sqrt(1 - theta_+^-2).
    """
    G = np.array([[1, rho], [rho, 1]])
    K = np.diag([a, b])
    w, U = np.linalg.eig(K @ G)
    u = U[:, np.argmax(w)]
    th = w.max()
    ov = np.abs(G @ u) / np.sqrt(u @ G @ u)
    return (np.sqrt(1 - th**-2) if th > 1 else 0.0) * ov


# ---------------------------------------------------------------- numerics
def make(n, a, b, rho):
    return TwoSpikes(n, a / ALPHA, b / np.sqrt(1 - ALPHA**2), rho, alpha=ALPHA)


def top2(Y):
    w, V = np.linalg.eigh(Y)
    return w[-1:-3:-1], V[:, -1:-3:-1]        # 2 largest, descending


def sample_overlaps(n, a, b, rho, reps):
    """Mean over reps of (|<x_hat,x1>|, |<x_hat,x2>|), x_hat = top eigenvector of Y."""
    o = np.zeros(2)
    for _ in range(reps):
        S = make(n, a, b, rho)
        _, V = top2(S.naive_matrix)          # naive_matrix = alpha*Y1 + sqrt(1-alpha^2)*Y2 = Y
        o += abs(V[:, 0] @ S.x1), abs(V[:, 0] @ S.x2)
    return o / reps


# ---------------------------------------------------------------- figures
def fig_eig(n=500, sims=30):
    """Top-2 eigenvalues of Y vs a, b, rho: mean +- std over `sims` simulations."""
    base = dict(a=2.0, b=3.0, rho=0.5)
    sweeps = dict(a=np.linspace(0.1, 4, 40), b=np.linspace(0.1, 4, 40), rho=np.linspace(0, 1, 40))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, (name, grid) in zip(axes, sweeps.items()):
        p = dict(base)
        th, mean, std = [], [], []
        for val in grid:
            p[name] = val
            th.append(theta_pm(**p))
            runs = [top2(make(n, **p).naive_matrix)[0] for _ in range(sims)]
            mean.append(np.mean(runs, 0)); std.append(np.std(runs, 0))
        th, mean, std = np.array(th), np.array(mean), np.array(std)
        ax.plot(grid, bbp(th[:, 0]), "k--", label=r"theory $\theta_\pm + 1/\theta_\pm$")
        ax.plot(grid, bbp(th[:, 1]), "k--")
        ax.errorbar(grid, mean[:, 0], std[:, 0], color="C0", capsize=2, lw=1, label=r"$\lambda_1(Y)$ simulation")
        ax.errorbar(grid, mean[:, 1], std[:, 1], color="C1", capsize=2, lw=1, label=r"$\lambda_2(Y)$ simulation")
        ax.axhline(2, color="gray", ls=":", label="bulk edge")
        others = ", ".join(f"{k}={v}" for k, v in base.items() if k != name)
        ax.set_xlabel(name); ax.set_ylabel("eigenvalue of Y"); ax.set_title(f"n={n}, {sims} sims, {others}")
        ax.grid(); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("eigenvalues_Y.png", dpi=120)


def fig_lambda(n=2000, rho=0.3, sims=30):
    """BBP eigenvector check vs lambda (lambda1 = lambda2 = lambda, so a = b = lambda/sqrt2
    and theta_pm = a(1 +- rho)).

    Plots |<u_pm, v_pm>|, the overlap between the eigenvectors u_pm of the NOISY matrix
    Y = P + W and the eigenvectors v_pm of the NOISELESS signal matrix P (report, eq. bbp-vec):
    theory says |<u_pm, v_pm>| -> sqrt(1 - theta_pm^-2) above threshold, 0 below.

    NB this is NOT the overlap with the signals themselves: that one is
    m_i = |<x_hat, x_i>| (report, eq. m12), computed by `theory_overlaps` / `sample_overlaps`.

    Dots/error bars: mean +- std over `sims` simulations. Dashed: theory.
    """
    lams = np.linspace(0.1, 5, 100)
    mean, std, th = [], [], []
    for lam in lams:
        a = b = ALPHA * lam
        runs = []
        for _ in range(sims):
            S = make(n, a, b, rho)
            _, V = top2(S.naive_matrix)
            v1, v2 = S.theory_vp()
            runs.append([abs(V[:, 0] @ v1), abs(V[:, 1] @ v2)])
        mean.append(np.mean(runs, 0)); std.append(np.std(runs, 0))
        th.append([np.sqrt(max(0, 1 - t**-2)) for t in theta_pm(a, b, rho)])
    mean, std, th = np.array(mean), np.array(std), np.array(th)
    sim_labels = [r"simulation: $|\langle \hat{u}_+, \hat{v}_+\rangle|$ ($n=%d$, %d runs)" % (n, sims),
                  r"simulation: $|\langle \hat{u}_-, \hat{v}_-\rangle|$ ($n=%d$, %d runs)" % (n, sims)]
    th_labels = [r"theory: $\sqrt{1-\theta_+^{-2}}$", r"theory: $\sqrt{1-\theta_-^{-2}}$"]
    colors = ["C0", "C1"]                       # + branch blue, - branch orange
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k in range(2):
        ax.errorbar(lams, mean[:, k], std[:, k], color=colors[k], capsize=2, lw=1, label=sim_labels[k])
        ax.plot(lams, th[:, k], "--", color=colors[k], lw=2.2, label=th_labels[k])
    ax.axvline(1 / (ALPHA * (1 + rho)), color="red", lw=1.2, label=r"BBP threshold $\theta_+=1$")
    ax.axvline(1 / (ALPHA * (1 - rho)), color="purple", lw=1.2, label=r"BBP threshold $\theta_-=1$")
    ax.set_xlabel(r"$\lambda$ ($\lambda_1=\lambda_2=\lambda$)")
    ax.set_ylabel(r"$|\langle \hat{u}_\pm, \hat{v}_\pm\rangle|$")
    ax.set_title(rf"Noisy vs noiseless eigenvectors   ($\rho$={rho}, $\lambda_1=\lambda_2=\lambda$)")
    ax.grid(); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig("overlaps_vs_lambda.png", dpi=120)


def heat(xname, xs, b_fixed, fname, n=300, reps=4):
    """Heatmaps of |<x_hat,x1>|, |<x_hat,x2>| vs (x, rho); x is lambda1 or a; b fixed."""
    rhos = np.linspace(0, 1, 30)
    A = xs * ALPHA if xname == r"\lambda_1" else xs
    Z = np.array([[sample_overlaps(n, a, b_fixed, r, reps) for a in A] for r in rhos])  # (rho, x, 2)       We are using Y noisy here
    X, R = np.meshgrid(xs, rhos)
    tp, _ = theta_pm(X * ALPHA if xname == r"\lambda_1" else X, b_fixed, R)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, ax in enumerate(axes):
        im = ax.pcolormesh(xs, rhos, Z[:, :, k], cmap="Greys", vmin=0, vmax=1, shading="nearest")
        ax.contour(X, R, tp, levels=[1], colors="red", linewidths=2)
        ax.plot([], [], color="red", lw=2, label=r"$\theta_+=1$")     # legend entry for the contour
        ax.set_xlabel(xname); ax.set_ylabel(r"$\rho$"); ax.legend(loc="upper right")
        ax.set_title(rf"$|\langle \hat x, x_{k+1}\rangle|$,  b={b_fixed:.2f}, n={n}")
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(fname, dpi=120)


def fig_heat():
    lam2 = 1.0
    heat("lambda1", np.linspace(0.1, 3, 25), np.sqrt(1 - ALPHA**2) * lam2, "heat_lambda1_rho.png")
    heat("a", np.linspace(0.1, 3, 25), 0.7, "heat_a_rho.png")


def fig_phase(rhos=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0), n=500, reps=3, omax=0.7):
    grid = np.linspace(0.1, 3, 75)
    A, B = np.meshgrid(grid, grid)                      # A[j, i] = grid[i] (x), B[j, i] = grid[j] (y)
    for rho in rhos:
        Z = np.array([[sample_overlaps(n, a, b, rho, reps) for a in grid] for b in grid])  # (b, a, 2)
        rgb = np.zeros((*Z.shape[:2], 3)); rgb[..., :2] = np.clip(Z / omax, 0, 1)
        tp, tm = theta_pm(A, B, rho)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(rgb, origin="lower", extent=[grid[0], grid[-1], grid[0], grid[-1]], aspect="auto")
        ax.axvline(1, color="white", ls="--", lw=1); ax.axhline(1, color="white", ls="--", lw=1)
        ax.contour(A, B, tp, levels=[1], colors="red", linewidths=2)
        ax.contour(A, B, tm, levels=[1], colors="red", linewidths=2, linestyles="dashed")
        ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_title(f"rho={rho}, n={n}")
        ax.legend(handles=[mpatches.Patch(color="k", label="none"), mpatches.Patch(color="r", label=r"$x_1$ only"),
                           mpatches.Patch(color=(0, 1, 0), label=r"$x_2$ only"), mpatches.Patch(color="y", label="both"),
                           plt.Line2D([], [], color="white", ls="--", label="BBP a=1, b=1"),
                           plt.Line2D([], [], color="red", label=r"$\theta_+=1$"),
                           plt.Line2D([], [], color="red", ls="--", label=r"$\theta_-=1$")],
                  loc="upper left", fontsize=7, framealpha=0.6)
        fig.tight_layout(); fig.savefig(f"phase_rho={rho}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["eig", "lam", "heat", "phase"]
    if "eig" in todo: fig_eig()
    if "lam" in todo: fig_lambda()
    if "heat" in todo: fig_heat()
    if "phase" in todo: fig_phase()
    plt.show()
