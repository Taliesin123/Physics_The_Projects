"""
Diagnostic plots for the EWC / Fisher-regularised recovery method
implemented in  fisher_regularization.py.

Four figures :

    1. Trajectory of  x_t  in the (x_1, x_2) plane during Stage B,
       for several values of the EWC weight mu.

    2. Overlap of  x_t  with x_1 and with x_2 as a function of
       iteration, across mu values.

    3. Final-overlap scatter :  each run ends at a point
       (|<x, x_1>|, |<x, x_2>|).  Plotting those points across mu
       and seeds shows the trade-off the method offers.

    4. Stage-B loss decomposition :  data loss vs Fisher penalty vs
       total, as a function of iteration.

Nothing in  fisher_regularization.py  or  model_2.py  is modified.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

import model_2 as m
import fisher_regularization as fr


# -----------------------------------------------------------------------------
#  Local helpers that RECORD the trajectory / loss decomposition
#  (we don't want to modify fisher_regularization.py, so we re-run the
#  same dynamics here and remember everything we need)
# -----------------------------------------------------------------------------
def gd_ewc_trajectory(Y1, Y2, alpha, x_star, F, mu,
                      iterations=300, lr=0.1, x0=None):
    """
    Same projected GD as fr.gd_ewc, but records (x_t), data-loss,
    fisher-penalty and total loss at every step.
    """
    x = (x_star.copy() if x0 is None else x0.copy())
    x /= np.linalg.norm(x)

    xs        = np.empty((iterations + 1, x.size))
    L_data    = np.empty(iterations + 1)
    L_fisher  = np.empty(iterations + 1)
    xs[0]       = x
    L_data[0]   = fr.complete_loss(x, Y1, Y2, alpha)
    L_fisher[0] = mu * fr.ewc_penalty(x, x_star, F)

    for t in range(iterations):
        g = (fr.complete_grad(x, Y1, Y2, alpha)
             + mu * fr.ewc_penalty_grad(x, x_star, F))
        x = x - lr * g
        x /= np.linalg.norm(x)
        xs[t + 1]       = x
        L_data[t + 1]   = fr.complete_loss(x, Y1, Y2, alpha)
        L_fisher[t + 1] = mu * fr.ewc_penalty(x, x_star, F)

    return xs, L_data, L_fisher


def project_to_plane(xs: np.ndarray,
                     x1: np.ndarray, x2: np.ndarray):
    """
    Project a trajectory xs (shape (T, N)) onto a 2-D orthonormal basis
    of span(x1, x2) via Gram-Schmidt.
    Returns (coord1, coord2) arrays of length T.
    """
    e1 = x1 / np.linalg.norm(x1)
    r  = x2 - (x2 @ e1) * e1
    e2 = r / np.linalg.norm(r)
    c1 = xs @ e1
    c2 = xs @ e2
    return c1, c2


# =============================================================================
#  FIGURE 1 :  Trajectory in the (x_1, x_2) plane
# =============================================================================
def plot_trajectory_in_spike_plane(
        N=400, lambda1=2.5, lambda2=2.5, rho=0.6,
        alpha=np.sqrt(0.5),
        mus=(0.0, 0.5, 2.0, 5.0, 20.0),
        iterations=300, lr=0.1, seed=0, save_path=None):
    np.random.seed(seed)
    prob = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
    Y1, Y2 = prob.Y1, prob.Y2
    x1, x2 = prob.spike1, prob.spike2

    # Stage A
    x_A, _ = fr.gd_misspecified(Y1, lr=lr, iterations=iterations)
    # fix sign so the anchor sits near +x_1 (eigenvectors are sign-ambiguous)
    if x_A @ x1 < 0:
        x_A = -x_A
    F      = fr.fisher_matrix(x_A, lamba=lambda1)

    # naive spectral (also flip sign for readability)
    v_naive = fr.naive_spectral(Y1, Y2, alpha)
    if v_naive @ x1 < 0:
        v_naive = -v_naive

    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=130)

    # plot unit circle in (x_1, x_2) plane
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgray", lw=1, zorder=0)

    # project the two true spikes (by construction: x1 is the e1 axis,
    # x2 lives somewhere in the plane)
    c1_x1, c2_x1 = project_to_plane(x1[None, :], x1, x2)
    c1_x2, c2_x2 = project_to_plane(x2[None, :], x1, x2)
    ax.scatter(c1_x1, c2_x1, color="tab:green", s=110, zorder=5,
               edgecolor="black", label=r"$x_1$")
    ax.scatter(c1_x2, c2_x2, color="tab:red",   s=110, zorder=5,
               edgecolor="black", label=r"$x_2$")

    # Stage-A anchor
    c1_A, c2_A = project_to_plane(x_A[None, :], x1, x2)
    ax.scatter(c1_A, c2_A, color="tab:blue", marker="*", s=200, zorder=6,
               edgecolor="black", label=r"$\hat x_A^*$ (Stage-A anchor)")

    # naive spectral
    c1_n, c2_n = project_to_plane(v_naive[None, :], x1, x2)
    ax.scatter(c1_n, c2_n, color="black", marker="x", s=100, zorder=6,
               label="naive spectral")

    # Stage-B trajectories for each mu
    cmap = plt.get_cmap("viridis")
    for k, mu in enumerate(mus):
        xs, _, _ = gd_ewc_trajectory(Y1, Y2, alpha, x_A, F, mu,
                                     iterations=iterations, lr=lr)
        c1, c2 = project_to_plane(xs, x1, x2)
        color = cmap(k / max(1, len(mus) - 1))
        ax.plot(c1, c2, color=color, lw=1.8, alpha=0.85,
                label=rf"$\mu={mu}$")
        ax.scatter(c1[-1], c2[-1], color=color, s=55, zorder=4,
                   edgecolor="black", linewidth=0.8)

    ax.set_xlabel(r"projection on $x_1$")
    ax.set_ylabel(r"projection on $x_1^\perp\cap\mathrm{span}(x_1,x_2)$")
    ax.set_title("Stage-B trajectories in the $(x_1, x_2)$ plane")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# =============================================================================
#  FIGURE 2 :  Overlap vs iteration for several mu
# =============================================================================
def plot_overlap_vs_iteration(
        N=400, lambda1=2.5, lambda2=2.5, rho=0.6,
        alpha=np.sqrt(0.5),
        mus=(0.0, 1.0, 5.0, 20.0),
        iterations=300, lr=0.1, seed=0, save_path=None):
    np.random.seed(seed)
    prob = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
    Y1, Y2 = prob.Y1, prob.Y2
    x1, x2 = prob.spike1, prob.spike2

    x_A, _ = fr.gd_misspecified(Y1, lr=lr, iterations=iterations)
    F      = fr.fisher_matrix(x_A, lamba=lambda1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130, sharey=True)

    cmap = plt.get_cmap("viridis")
    for k, mu in enumerate(mus):
        xs, _, _ = gd_ewc_trajectory(Y1, Y2, alpha, x_A, F, mu,
                                     iterations=iterations, lr=lr)
        o1 = np.abs(xs @ x1)
        o2 = np.abs(xs @ x2)
        color = cmap(k / max(1, len(mus) - 1))
        ax[0].plot(o1, color=color, label=rf"$\mu={mu}$")
        ax[1].plot(o2, color=color, label=rf"$\mu={mu}$")

    for a, title in zip(ax, [r"$|\langle x_t, x_1\rangle|$",
                             r"$|\langle x_t, x_2\rangle|$"]):
        a.set_xlabel("iteration")
        a.set_title(title)
        a.grid(alpha=0.4)
        a.set_ylim(0, 1.05)
    ax[0].set_ylabel("overlap")
    ax[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Stage-B convergence across $\\mu$")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# =============================================================================
#  FIGURE 3 :  Final-overlap scatter across mu and seeds
# =============================================================================
def plot_final_overlap_scatter(
        N=300, lambda1=2.5, lambda2=2.5, rho=0.6,
        alpha=np.sqrt(0.5),
        mus=(0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        n_seeds=20, iterations=250, lr=0.1, seed_base=100,
        save_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=130)
    cmap = plt.get_cmap("viridis")

    naive_pts = []
    for k, mu in enumerate(mus):
        pts = []
        for s in range(n_seeds):
            np.random.seed(seed_base + s)
            prob = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
            Y1, Y2 = prob.Y1, prob.Y2
            x1, x2 = prob.spike1, prob.spike2

            x_A, _ = fr.gd_misspecified(Y1, lr=lr, iterations=iterations)
            F      = fr.fisher_matrix(x_A, lamba=lambda1)
            x_B, _ = fr.gd_ewc(Y1, Y2, alpha, x_A, F, mu,
                               lr=lr, iterations=iterations)
            pts.append((fr.overlap(x_B, x1), fr.overlap(x_B, x2)))

            if k == 0:
                v_naive = fr.naive_spectral(Y1, Y2, alpha)
                naive_pts.append((fr.overlap(v_naive, x1),
                                  fr.overlap(v_naive, x2)))

        pts = np.asarray(pts)
        color = cmap(k / max(1, len(mus) - 1))
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=28, alpha=0.8,
                   edgecolor="black", linewidth=0.3, label=rf"$\mu={mu}$")

    naive_pts = np.asarray(naive_pts)
    ax.scatter(naive_pts[:, 0], naive_pts[:, 1],
               color="black", marker="x", s=35,
               label="naive spectral")

    # y = x diagonal
    ax.plot([0, 1], [0, 1], color="gray", lw=0.6, ls=":")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$|\langle \hat x, x_1\rangle|$")
    ax.set_ylabel(r"$|\langle \hat x, x_2\rangle|$")
    ax.set_title(rf"Final overlaps  (N={N}, $\rho$={rho})"
                 "\nlower-right corner = pure $x_1$ recovery")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# =============================================================================
#  FIGURE 4 :  Loss decomposition over iterations
# =============================================================================
def plot_loss_decomposition(
        N=400, lambda1=2.5, lambda2=2.5, rho=0.6,
        alpha=np.sqrt(0.5),
        mus=(0.5, 2.0, 10.0),
        iterations=300, lr=0.1, seed=0, save_path=None):
    np.random.seed(seed)
    prob = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
    Y1, Y2 = prob.Y1, prob.Y2

    x_A, _ = fr.gd_misspecified(Y1, lr=lr, iterations=iterations)
    F      = fr.fisher_matrix(x_A, lamba=lambda1)

    fig, ax = plt.subplots(1, len(mus), figsize=(4.2 * len(mus), 4),
                           dpi=130, sharey=False)
    if len(mus) == 1:
        ax = [ax]

    for k, mu in enumerate(mus):
        _, L_data, L_fish = gd_ewc_trajectory(Y1, Y2, alpha, x_A, F, mu,
                                              iterations=iterations, lr=lr)
        total = L_data + L_fish
        ax[k].plot(L_data, label=r"$L_{\mathrm{complete}}$",
                   color="tab:blue")
        ax[k].plot(L_fish, label=r"$\mu\cdot R_F$",
                   color="tab:orange")
        ax[k].plot(total,  label=r"total", color="black", lw=1.5)
        ax[k].set_xlabel("iteration")
        ax[k].set_title(rf"$\mu = {mu}$")
        ax[k].grid(alpha=0.4)
        ax[k].legend(fontsize=9)
    ax[0].set_ylabel("loss")
    fig.suptitle("Stage-B loss decomposition")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(here, "..", "Plots", "Fisher")

    print("1/4  trajectory in (x_1, x_2) plane ...")
    plot_trajectory_in_spike_plane(
        save_path=os.path.join(plot_dir, "ewc_trajectory_plane.png"))

    print("2/4  overlap vs iteration ...")
    plot_overlap_vs_iteration(
        save_path=os.path.join(plot_dir, "ewc_overlap_vs_iteration.png"))

    print("3/4  final-overlap scatter ...")
    plot_final_overlap_scatter(
        save_path=os.path.join(plot_dir, "ewc_final_overlap_scatter.png"))

    print("4/4  loss decomposition ...")
    plot_loss_decomposition(
        save_path=os.path.join(plot_dir, "ewc_loss_decomposition.png"))

    print("done.")
