import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import source_twospike as sts       # theory
import source_plots as sp           # functions used to make the plots (simulations)
# The TwoSpikes class is in class_twospike.py


ALPHA = np.sqrt(0.5)
DIR_PLOTS = "results"       # la ou on envoie nos plots (quel folder)
os.makedirs(DIR_PLOTS, exist_ok=True)


# ---------------------------------------------------------------- eigenvalues / eigenvectors of Y (naive)
def fig_eig(n=500, sims=30):
    """Top-2 eigenvalues of Y vs a, b, rho: mean +- std over `sims` simulations.
    Dashed: BBP theory theta_pm + 1/theta_pm (bulk edge 2 below threshold), theta_pm = eigenvalues of P."""
    base = dict(a=2.0, b=3.0, rho=0.5)
    sweeps = dict(a=np.linspace(0.1, 4, 40), b=np.linspace(0.1, 4, 40), rho=np.linspace(0, 1, 40))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for ax, (name, grid) in zip(axes, sweeps.items()):
        p = dict(base)
        th, mean, std = [], [], []
        for val in grid:
            p[name] = val
            th.append(sts.eigval_Y(p["a"], p["b"], p["rho"]))       # theory
            runs = []
            for _ in range(sims):
                S = sp.make(n, p["a"], p["b"], p["rho"], ALPHA)
                eigvals, _ = sp.top2(S.naive_matrix)                  # 2 top eigenvalues of Y
                runs.append(eigvals)
            mean.append(np.mean(runs, 0)); std.append(np.std(runs, 0))
        th, mean, std = np.array(th), np.array(mean), np.array(std)

        ax.plot(grid, th[:, 0], "k--", label=r"theory $\theta_\pm + 1/\theta_\pm$")
        ax.plot(grid, th[:, 1], "k--")
        ax.errorbar(grid, mean[:, 0], std[:, 0], color="C0", capsize=2, lw=1, label=r"$\lambda_1(Y)$ simulation")
        ax.errorbar(grid, mean[:, 1], std[:, 1], color="C1", capsize=2, lw=1, label=r"$\lambda_2(Y)$ simulation")
        ax.axhline(2, color="gray", ls=":", label="bulk edge")
        others = ", ".join(f"{k}={v}" for k, v in base.items() if k != name)
        ax.set_xlabel(name); ax.set_ylabel("eigenvalue of Y"); ax.set_title(f"n={n}, {sims} sims, {others}")
        ax.grid(); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(DIR_PLOTS, "eigenvalues_Y.png"), dpi=120)
    plt.close(fig)


def fig_lambda(n=2000, rho=0.3, sims=30):
    """BBP eigenvector check vs lambda (lambda1 = lambda2 = lambda, so a = b = lambda/sqrt2
    and theta_pm = a(1 +- rho)).

    Plots |<u_pm, v_pm>|, the overlap between the eigenvectors u_pm of the NOISY matrix
    Y = P + W and the eigenvectors v_pm of the NOISELESS signal matrix P (report, eq. bbp-vec):
    theory says |<u_pm, v_pm>| -> sqrt(1 - theta_pm^-2) above threshold, 0 below.

    NB this is NOT the overlap with the signals themselves: that one is
    m_i = |<x_hat, x_i>| (report, eq. m12), computed by `sp.sample_overlaps`.

    Dots/error bars: mean +- std over `sims` simulations. Dashed: theory.
    """
    lams = np.linspace(0.1, 5, 100)
    mean, std, th = [], [], []
    for lam in lams:
        a = ALPHA * lam                          # lambda1 = lambda2 = lam
        b = np.sqrt(1 - ALPHA**2) * lam
        runs = []
        for _ in range(sims):
            S = sp.make(n, a, b, rho, ALPHA)
            _, U = sp.top2(S.naive_matrix)                            # u_+, u_- : top-2 eigenvectors of Y
            vp, vm = sts.eigvect_P(a, b, S.get_x1(), S.get_x2(), rho)  # v_+, v_- : eigenvectors of P
            runs.append([abs(U[:, 0] @ vp), abs(U[:, 1] @ vm)])
        mean.append(np.mean(runs, 0)); std.append(np.std(runs, 0))
        th.append([np.sqrt(max(0, 1 - t**-2)) for t in sts.eigval_P(a, b, rho)])
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
    fig.tight_layout(); fig.savefig(os.path.join(DIR_PLOTS, "overlaps_vs_lambda.png"), dpi=120)
    plt.close(fig)


def heat(xname, xs, b_fixed, fname, n=300, reps=4):
    """Heatmaps of |<x_hat,x1>|, |<x_hat,x2>| vs (x, rho); x is "lambda1" or "a"; b fixed."""
    rhos = np.linspace(0, 1, 30)
    if xname == "lambda1":
        a_values = xs * ALPHA                    # a = alpha * lambda1
    else:
        a_values = xs
    Z = np.array([[sp.sample_overlaps(n, a, b_fixed, r, reps, ALPHA) for a in a_values] for r in rhos])  # (rho, x, 2), Y noisy
    X, R = np.meshgrid(xs, rhos)
    if xname == "lambda1":
        tp, _ = sts.eigval_P(X * ALPHA, b_fixed, R)
    else:
        tp, _ = sts.eigval_P(X, b_fixed, R)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, ax in enumerate(axes):
        im = ax.pcolormesh(xs, rhos, Z[:, :, k], cmap="Greys", vmin=0, vmax=1, shading="nearest")
        ax.contour(X, R, tp, levels=[1], colors="red", linewidths=2)
        ax.plot([], [], color="red", lw=2, label=r"$\theta_+=1$")     # legend entry for the contour
        ax.set_xlabel(xname); ax.set_ylabel(r"$\rho$"); ax.legend(loc="upper right")
        ax.set_title(rf"$|\langle \hat x, x_{k+1}\rangle|$,  b={b_fixed:.2f}, n={n}")
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(DIR_PLOTS, fname), dpi=120)
    plt.close(fig)


def fig_heat():
    lam2 = 1.0
    heat("lambda1", np.linspace(0.1, 3, 25), np.sqrt(1 - ALPHA**2) * lam2, "heat_lambda1_rho.png")
    heat("a", np.linspace(0.1, 3, 25), 0.70, "heat_a_rho.png")


def fig_phase(rhos=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0), n=500, reps=3, omax=0.7):
    grid = np.linspace(0.1, 3, 75)
    A, B = np.meshgrid(grid, grid)                      # A[j, i] = grid[i] (x), B[j, i] = grid[j] (y)
    for rho in rhos:
        Z = np.array([[sp.sample_overlaps(n, a, b, rho, reps, ALPHA) for a in grid] for b in grid])  # (b, a, 2)
        rgb = np.zeros((*Z.shape[:2], 3)); rgb[..., :2] = np.clip(Z / omax, 0, 1)
        tp, _ = sts.eigval_P(A, B, rho)                 # only theta_+ is plotted
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(rgb, origin="lower", extent=[grid[0], grid[-1], grid[0], grid[-1]], aspect="auto")
        ax.axvline(1, color="white", ls="--", lw=1); ax.axhline(1, color="white", ls="--", lw=1)
        ax.contour(A, B, tp, levels=[1], colors="red", linewidths=2)
        ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_title(f"rho={rho}, n={n}")
        ax.legend(handles=[mpatches.Patch(color="k", label="none"), mpatches.Patch(color="r", label=r"$x_1$ only"),
                           mpatches.Patch(color=(0, 1, 0), label=r"$x_2$ only"), mpatches.Patch(color="y", label="both"),
                           plt.Line2D([], [], color="white", ls="--", label="BBP a=1, b=1"),
                           plt.Line2D([], [], color="red", label=r"$\theta_+=1$")],
                  loc="upper left", fontsize=7, framealpha=0.6)
        fig.tight_layout(); fig.savefig(os.path.join(DIR_PLOTS, f"phase_rho={rho}.png"), dpi=120); plt.close(fig)


# ---------------------------------------------------------------- the matrix A = Y2 - I + 2 mu F1(x), for the choices of x
# PLOTS DE L'OVERLAP ENTRE x1 ET LE TOP EIGENVECTOR DE A, POUR LES 4 CHOIX DE x (x_hat, x_pg1, x_m1, x_pg12)

def overlap_x1_vs_n(lam1=2.0, lam2=3.0, mu=0.5, rho=0.5, sims=20):
    n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]

    mean_xhat, err_xhat = [], []
    mean_bbp, err_bbp = [], []
    mean_m1x1, err_m1x1 = [], []
    mean_pg12, err_pg12 = [], []

    for n in n_values:
        list_xhat, list_bbp, list_m1x1, list_pg12 = [], [], [], []
        for s in range(sims):
            o_xhat, o_bbp, o_m1x1, o_pg12 = sp.overlaps_A_one_draw(n, lam1, lam2, rho, mu, alpha=ALPHA)
            list_xhat.append(o_xhat)
            list_bbp.append(o_bbp)
            list_m1x1.append(o_m1x1)
            list_pg12.append(o_pg12)
        mean_xhat.append(np.mean(list_xhat))
        err_xhat.append(np.std(list_xhat) / np.sqrt(sims))
        mean_bbp.append(np.mean(list_bbp))
        err_bbp.append(np.std(list_bbp) / np.sqrt(sims))
        mean_m1x1.append(np.mean(list_m1x1))
        err_m1x1.append(np.std(list_m1x1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    theory = sts.overlap_A_theory(lam1, lam2, rho, mu)     # does not depend on n

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(n_values, mean_xhat, err_xhat, color="C0", marker="o", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(\hat x)\rangle|$")
    plt.errorbar(n_values, mean_bbp, err_bbp, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{pg1})\rangle|$")
    plt.errorbar(n_values, mean_m1x1, err_m1x1, color="C3", marker="^", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{m1})\rangle|$")
    plt.errorbar(n_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{pg12})\rangle|$")
    plt.axhline(theory, color="k", ls="--", label=r"theory ($\hat x$, $x_{pg1}$, $x_{pg12}$)")
    plt.xlabel(r"$n$")
    plt.ylabel("overlap")
    plt.title(rf"$\lambda_1$={lam1}, $\rho$={rho}, $\lambda_2$={lam2}, $\mu$={mu}, {sims} sims" + "\n" + r"$v(x)$ = top eigenvector of $A(x) = Y_2 - I + 2\mu F_1(x)$", fontsize=10)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "overlap_x1_vs_n.png"), dpi=120)
    plt.close()


def overlap_x1_vs_rho(n=500, lam1=2.0, lam2=3.0, mu=0.5, sims=20):
    rho_values = np.linspace(0, 1, 21)

    mean_xhat, err_xhat = [], []
    mean_bbp, err_bbp = [], []
    mean_m1x1, err_m1x1 = [], []
    mean_pg12, err_pg12 = [], []

    for rho in rho_values:
        list_xhat, list_bbp, list_m1x1, list_pg12 = [], [], [], []
        for s in range(sims):
            o_xhat, o_bbp, o_m1x1, o_pg12 = sp.overlaps_A_one_draw(n, lam1, lam2, rho, mu, alpha=ALPHA)
            list_xhat.append(o_xhat)
            list_bbp.append(o_bbp)
            list_m1x1.append(o_m1x1)
            list_pg12.append(o_pg12)
        mean_xhat.append(np.mean(list_xhat))
        err_xhat.append(np.std(list_xhat) / np.sqrt(sims))
        mean_bbp.append(np.mean(list_bbp))
        err_bbp.append(np.std(list_bbp) / np.sqrt(sims))
        mean_m1x1.append(np.mean(list_m1x1))
        err_m1x1.append(np.std(list_m1x1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    rho_fine = np.linspace(0, 1, 300)
    theory = []
    for rho in rho_fine:
        theory.append(sts.overlap_A_theory(lam1, lam2, rho, mu))

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(rho_values, mean_xhat, err_xhat, color="C0", marker="o", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(\hat x)\rangle|$")
    plt.errorbar(rho_values, mean_bbp, err_bbp, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{pg1})\rangle|$")
    plt.errorbar(rho_values, mean_m1x1, err_m1x1, color="C3", marker="^", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{m1})\rangle|$")
    plt.errorbar(rho_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{pg12})\rangle|$")
    plt.plot(rho_fine, theory, "k--", label=r"theory ($\hat x$, $x_{pg1}$, $x_{pg12}$)")
    plt.xlabel(r"$\rho$")
    plt.ylabel("overlap")
    plt.title(rf"$\lambda_1$={lam1}, $n$={n}, $\lambda_2$={lam2}, $\mu$={mu}, {sims} sims" + "\n" + r"$v(x)$ = top eigenvector of $A(x) = Y_2 - I + 2\mu F_1(x)$", fontsize=10)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "overlap_x1_vs_rho.png"), dpi=120)
    plt.close()


def overlap_x1_vs_lam1(n=500, lam2=3.0, mu=0.5, rho=0.5, sims=20):
    lam1_values = np.linspace(0.5, 4, 21)

    mean_xhat, err_xhat = [], []
    mean_bbp, err_bbp = [], []
    mean_m1x1, err_m1x1 = [], []
    mean_pg12, err_pg12 = [], []

    for lam1 in lam1_values:
        list_xhat, list_bbp, list_m1x1, list_pg12 = [], [], [], []
        for s in range(sims):
            o_xhat, o_bbp, o_m1x1, o_pg12 = sp.overlaps_A_one_draw(n, lam1, lam2, rho, mu, alpha=ALPHA)
            list_xhat.append(o_xhat)
            list_bbp.append(o_bbp)
            list_m1x1.append(o_m1x1)
            list_pg12.append(o_pg12)
        mean_xhat.append(np.mean(list_xhat))
        err_xhat.append(np.std(list_xhat) / np.sqrt(sims))
        mean_bbp.append(np.mean(list_bbp))
        err_bbp.append(np.std(list_bbp) / np.sqrt(sims))
        mean_m1x1.append(np.mean(list_m1x1))
        err_m1x1.append(np.std(list_m1x1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    lam1_fine = np.linspace(0.5, 4, 300)
    theory = []
    for lam1 in lam1_fine:
        theory.append(sts.overlap_A_theory(lam1, lam2, rho, mu))

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(lam1_values, mean_xhat, err_xhat, color="C0", marker="o", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(\hat x)\rangle|$")
    plt.errorbar(lam1_values, mean_bbp, err_bbp, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{pg1})\rangle|$")
    plt.errorbar(lam1_values, mean_m1x1, err_m1x1, color="C3", marker="^", ms=5, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{m1})\rangle|$")
    plt.errorbar(lam1_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$|\langle x_1, v(x_{pg12})\rangle|$")
    plt.axvline(1, color="red", ls=":", label=r"BBP threshold $\lambda_1=1$")
    plt.plot(lam1_fine, theory, "k--", label=r"theory ($\hat x$, $x_{pg1}$, $x_{pg12}$)")
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel("overlap")
    plt.title(rf"$\rho$={rho}, $n$={n}, $\lambda_2$={lam2}, $\mu$={mu}, {sims} sims" + "\n" + r"$v(x)$ = top eigenvector of $A(x) = Y_2 - I + 2\mu F_1(x)$", fontsize=10)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "overlap_x1_vs_lam1.png"), dpi=120)
    plt.close()

# PLOTS DE L'OVERLAP ENTRE x_hat ET x_pg1, ET ENTRE x_hat ET x_pg12
# Theory: <x_hat, x_pg> = m1 <x_hat, x1> + sqrt(1-m1^2) <x_hat, g>, and <x_hat, g> -> 0 (g random), so both -> m1^2

def overlap_xhat_vs_n(lam1=2.0, lam2=3.0, rho=0.5, sims=20):
    n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]

    mean_pg1, err_pg1 = [], []
    mean_pg12, err_pg12 = [], []

    for n in n_values:
        list_pg1, list_pg12 = [], []
        for s in range(sims):
            o_pg1, o_pg12 = sp.overlaps_xhat_one_draw(n, lam1, lam2, rho, alpha=ALPHA)
            list_pg1.append(o_pg1)
            list_pg12.append(o_pg12)
        mean_pg1.append(np.mean(list_pg1))
        err_pg1.append(np.std(list_pg1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    theory = sts.m1_theory(lam1) ** 2      # does not depend on n

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(n_values, mean_pg1, err_pg1, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$|\langle \hat x, x_{pg1}\rangle|$")
    plt.errorbar(n_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$|\langle \hat x, x_{pg12}\rangle|$")
    plt.axhline(theory, color="k", ls="--", label=r"theory $m_1^2$")
    plt.xlabel(r"$n$")
    plt.ylabel("overlap")
    plt.title(rf"$\lambda_1$={lam1}, $\rho$={rho}, $\lambda_2$={lam2}, {sims} sims")
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "overlap_xhat_vs_n.png"), dpi=120)
    plt.close()


def overlap_xhat_vs_rho(n=500, lam1=2.0, lam2=3.0, sims=20):
    rho_values = np.linspace(0, 1, 21)

    mean_pg1, err_pg1 = [], []
    mean_pg12, err_pg12 = [], []

    for rho in rho_values:
        list_pg1, list_pg12 = [], []
        for s in range(sims):
            o_pg1, o_pg12 = sp.overlaps_xhat_one_draw(n, lam1, lam2, rho, alpha=ALPHA)
            list_pg1.append(o_pg1)
            list_pg12.append(o_pg12)
        mean_pg1.append(np.mean(list_pg1))
        err_pg1.append(np.std(list_pg1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    theory = sts.m1_theory(lam1) ** 2      # does not depend on rho

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(rho_values, mean_pg1, err_pg1, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$|\langle \hat x, x_{pg1}\rangle|$")
    plt.errorbar(rho_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$|\langle \hat x, x_{pg12}\rangle|$")
    plt.axhline(theory, color="k", ls="--", label=r"theory $m_1^2$")
    plt.xlabel(r"$\rho$")
    plt.ylabel("overlap")
    plt.title(rf"$\lambda_1$={lam1}, $n$={n}, $\lambda_2$={lam2}, {sims} sims")
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "overlap_xhat_vs_rho.png"), dpi=120)
    plt.close()


def overlap_xhat_vs_lam1(n=500, lam2=3.0, rho=0.5, sims=20):
    lam1_values = np.linspace(0.5, 4, 21)

    mean_pg1, err_pg1 = [], []
    mean_pg12, err_pg12 = [], []

    for lam1 in lam1_values:
        list_pg1, list_pg12 = [], []
        for s in range(sims):
            o_pg1, o_pg12 = sp.overlaps_xhat_one_draw(n, lam1, lam2, rho, alpha=ALPHA)
            list_pg1.append(o_pg1)
            list_pg12.append(o_pg12)
        mean_pg1.append(np.mean(list_pg1))
        err_pg1.append(np.std(list_pg1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    lam1_fine = np.linspace(0.5, 4, 300)
    theory = []
    for lam1 in lam1_fine:
        theory.append(sts.m1_theory(lam1) ** 2)

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(lam1_values, mean_pg1, err_pg1, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$|\langle \hat x, x_{pg1}\rangle|$")
    plt.errorbar(lam1_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$|\langle \hat x, x_{pg12}\rangle|$")
    plt.axvline(1, color="red", ls=":", label=r"BBP threshold $\lambda_1=1$")
    plt.plot(lam1_fine, theory, "k--", label=r"theory $m_1^2$")
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel("overlap")
    plt.title(rf"$\rho$={rho}, $n$={n}, $\lambda_2$={lam2}, {sims} sims")
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "overlap_xhat_vs_lam1.png"), dpi=120)
    plt.close()


# PLOTS DE LA PLUS GRANDE VALEUR PROPRE DE A = Y2 - I + 2 mu F1(x), POUR LES 4 CHOIX DE x (x_hat, x_pg1, x_m1, x_pg12)
# Theory (for x_hat, x_pg1, x_pg12): theta + 1/theta - 1, theta = top eigenvalue of beta (sts.top_eig_A_theory)

def top_eigen_a_vs_rho(n=500, lam1=2.0, lam2=3.0, mu=0.5, sims=20):
    rho_values = np.linspace(0, 1, 26)

    mean_xhat, err_xhat = [], []
    mean_pg1, err_pg1 = [], []
    mean_m1, err_m1 = [], []
    mean_pg12, err_pg12 = [], []

    for rho in rho_values:
        list_xhat, list_pg1, list_m1, list_pg12 = [], [], [], []
        for s in range(sims):
            t_xhat, t_pg1, t_m1, t_pg12 = sp.top_eig_A_one_draw(n, lam1, lam2, rho, mu, alpha=ALPHA)
            list_xhat.append(t_xhat)
            list_pg1.append(t_pg1)
            list_m1.append(t_m1)
            list_pg12.append(t_pg12)
        mean_xhat.append(np.mean(list_xhat))
        err_xhat.append(np.std(list_xhat) / np.sqrt(sims))
        mean_pg1.append(np.mean(list_pg1))
        err_pg1.append(np.std(list_pg1) / np.sqrt(sims))
        mean_m1.append(np.mean(list_m1))
        err_m1.append(np.std(list_m1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    rho_fine = np.linspace(0, 1, 300)
    theory = []
    for rho in rho_fine:
        theory.append(sts.top_eig_A_theory(lam1, lam2, rho, mu))

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(rho_values, mean_xhat, err_xhat, color="C0", marker="o", ms=5, ls="none", capsize=2, label=r"$\lambda_{\max}(A(\hat x))$")
    plt.errorbar(rho_values, mean_pg1, err_pg1, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$\lambda_{\max}(A(x_{pg1}))$")
    plt.errorbar(rho_values, mean_m1, err_m1, color="C3", marker="^", ms=5, ls="none", capsize=2, label=r"$\lambda_{\max}(A(x_{m1}))$")
    plt.errorbar(rho_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$\lambda_{\max}(A(x_{pg12}))$")
    plt.plot(rho_fine, theory, "k--", label=r"theory $\theta + 1/\theta - 1$, $\theta = \theta_{\max}(\beta)$")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"top eigenvalue of $A$")
    plt.title(rf"$\lambda_1$={lam1}, $n$={n}, $\lambda_2$={lam2}, $\mu$={mu}, {sims} sims" + "\n"
              + r"$A(x) = Y_2 - I + 2\mu F_1(x)$", fontsize=10)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "top_eigen_a_vs_rho.png"), dpi=120)
    plt.close()


def top_eigen_a_vs_lam1(n=500, lam2=3.0, mu=0.5, rho=0.5, sims=20):
    lam1_values = np.linspace(0.5, 4, 26)

    mean_xhat, err_xhat = [], []
    mean_pg1, err_pg1 = [], []
    mean_m1, err_m1 = [], []
    mean_pg12, err_pg12 = [], []

    for lam1 in lam1_values:
        list_xhat, list_pg1, list_m1, list_pg12 = [], [], [], []
        for s in range(sims):
            t_xhat, t_pg1, t_m1, t_pg12 = sp.top_eig_A_one_draw(n, lam1, lam2, rho, mu, alpha=ALPHA)
            list_xhat.append(t_xhat)
            list_pg1.append(t_pg1)
            list_m1.append(t_m1)
            list_pg12.append(t_pg12)
        mean_xhat.append(np.mean(list_xhat))
        err_xhat.append(np.std(list_xhat) / np.sqrt(sims))
        mean_pg1.append(np.mean(list_pg1))
        err_pg1.append(np.std(list_pg1) / np.sqrt(sims))
        mean_m1.append(np.mean(list_m1))
        err_m1.append(np.std(list_m1) / np.sqrt(sims))
        mean_pg12.append(np.mean(list_pg12))
        err_pg12.append(np.std(list_pg12) / np.sqrt(sims))

    lam1_fine = np.linspace(0.5, 4, 300)
    theory = []
    for lam1 in lam1_fine:
        theory.append(sts.top_eig_A_theory(lam1, lam2, rho, mu))

    plt.figure(figsize=(7, 4.8))
    plt.errorbar(lam1_values, mean_xhat, err_xhat, color="C0", marker="o", ms=5, ls="none", capsize=2, label=r"$\lambda_{\max}(A(\hat x))$")
    plt.errorbar(lam1_values, mean_pg1, err_pg1, color="C2", marker="s", ms=5, ls="none", capsize=2, label=r"$\lambda_{\max}(A(x_{pg1}))$")
    plt.errorbar(lam1_values, mean_m1, err_m1, color="C3", marker="^", ms=5, ls="none", capsize=2, label=r"$\lambda_{\max}(A(x_{m1}))$")
    plt.errorbar(lam1_values, mean_pg12, err_pg12, color="C1", marker="+", ms=8, ls="none", capsize=2, label=r"$\lambda_{\max}(A(x_{pg12}))$")
    plt.axvline(1, color="red", ls=":", label=r"BBP threshold $\lambda_1=1$")
    plt.plot(lam1_fine, theory, "k--", label=r"theory $\theta + 1/\theta - 1$, $\theta = \theta_{\max}(\beta)$")
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel(r"top eigenvalue of $A$")
    plt.title(rf"$\rho$={rho}, $n$={n}, $\lambda_2$={lam2}, $\mu$={mu}, {sims} sims" + "\n"
              + r"$A(x) = Y_2 - I + 2\mu F_1(x)$", fontsize=10)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "top_eigen_a_vs_lam1.png"), dpi=120)
    plt.close()

# PLOT DU BULK DES VALEURS PROPRES DE A(x) = Y2 - I + 2 mu F1(x), POUR x = x_hat, x_pg1, x_pg12
# Theory: bulk = semicircle of Y2 shifted by -1 (the "- I"), so on [-3, 1].
#         top eigenvalue (outlier) = theta + 1/theta - 1, theta = top eigenvalue of beta (sts.top_eig_A_theory)

def bulk_A(n=2000, lam1=2.0, lam2=3.0, mu=0.5, rho=0.5):
    eig_xhat, eig_pg1, eig_pg12 = sp.eigvals_A_one_draw(n, lam1, lam2, rho, mu, alpha=ALPHA)

    z = np.linspace(-3.5, 1.5, 500)
    bulk_theory = sts.semi_circle(z + 1)                     # semicircle shifted by -1
    top_theory = sts.top_eig_A_theory(lam1, lam2, rho, mu)

    bins = np.linspace(-3.5, 3.0, 100)                       # same bins for the three histograms

    plt.figure(figsize=(8, 5))

    # histograms (outlines only, so that the three stay visible)
    plt.hist(eig_xhat, bins=bins, density=True, lw=1.5, color="C0", label=r"eigenvalues of $A(\hat x)$")
    plt.hist(eig_pg1, bins=bins, density=True, lw=1.5, color="C2", label=r"eigenvalues of $A(x_{pg1})$")
    plt.hist(eig_pg12, bins=bins, density=True, lw=1.5, color="C1", label=r"eigenvalues of $A(x_{pg12})$")

    # top eigenvalue of each, on the x axis
    plt.plot(eig_xhat[-1], 0, "o", color="C0", ms=7, clip_on=False, label=rf"top eigenvalue of $A(\hat x)$ = {eig_xhat[-1]:.3f}")
    plt.plot(eig_pg1[-1], 0, "s", color="C2", ms=7, clip_on=False, label=rf"top eigenvalue of $A(x_{{pg1}})$ = {eig_pg1[-1]:.3f}")
    plt.plot(eig_pg12[-1], 0, "P", color="C1", ms=8, clip_on=False, label=rf"top eigenvalue of $A(x_{{pg12}})$ = {eig_pg12[-1]:.3f}")

    # theory (drawn last)
    plt.plot(z, bulk_theory, "k-", lw=1.5, label="theory: semicircle shifted by -1")
    plt.axvline(top_theory, color="k", ls="--", label=f"theory top eigenvalue = {top_theory:.3f}")

    plt.xlabel("eigenvalue")
    plt.ylabel("density")
    plt.title(rf"Spectrum of $A(x) = Y_2 - I + 2\mu F_1(x)$" + "\n"
              + rf"$n$={n}, $\lambda_1$={lam1}, $\lambda_2$={lam2}, $\rho$={rho}, $\mu$={mu}, one draw", fontsize=10)
    plt.grid()
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "bulk_A.png"), dpi=120)
    plt.close()



if __name__ == "__main__":
    todo = sys.argv[1:] or ["eig", "lam", "heat", "phase", "top_eigen_a_rho", "top_eigen_a_lam1", "bulk_A", "overlap_n", "overlap_rho", "overlap_lam1",
                             "overlap_xhat_n", "overlap_xhat_rho", "overlap_xhat_lam1"]
    if "eig" in todo: fig_eig()
    if "lam" in todo: fig_lambda()
    if "heat" in todo: fig_heat()
    if "phase" in todo: fig_phase()
    if "top_eigen_a_rho" in todo: top_eigen_a_vs_rho()
    if "top_eigen_a_lam1" in todo: top_eigen_a_vs_lam1()
    if "bulk_A" in todo: bulk_A()
    if "overlap_n" in todo: overlap_x1_vs_n()
    if "overlap_rho" in todo: overlap_x1_vs_rho()
    if "overlap_lam1" in todo: overlap_x1_vs_lam1()
    if "overlap_xhat_n" in todo: overlap_xhat_vs_n()
    if "overlap_xhat_rho" in todo: overlap_xhat_vs_rho()
    if "overlap_xhat_lam1" in todo: overlap_xhat_vs_lam1()
    plt.show()
