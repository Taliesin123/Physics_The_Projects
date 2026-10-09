import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import two_spikes_th_lib as theory
from spike_lib import TwoSpikes
from two_spikes_th_lib import TwoS_theory
import os

    
ALPHA = np.sqrt(0.5)
DIR_PLOTS = "results"       # la ou on envoie nos plots (quel folder)
os.makedirs(DIR_PLOTS, exist_ok=True)




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
        S = TwoSpikes(n, a / ALPHA, b / np.sqrt(1 - ALPHA**2), rho, alpha=ALPHA)
                # naive_matrix = alpha*Y1 + sqrt(1-alpha^2)*Y2 = Y
        o += S.overlaps("naive")
    return o / reps


# ---------------------------------------------------------------- figures
def fig_eig(n=500, sims=30):
    """Top-2 eigenvalues of Y vs a, b, rho: mean +- std over `sims` simulations."""
    base = dict(a=2.0, b=3.0, rho=0.5)
    sweeps = dict(a=np.linspace(0.1, 4, 40), b=np.linspace(0.1, 4, 40), rho=np.linspace(0, 1, 40))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for ax, (name, grid) in zip(axes, sweeps.items()):

        p = dict(base)
        th, bbp, mean, std = [], [], []
        for val in grid:
            p[name] = val

            lam1 = p["a"]/ALPHA
            lam2 = p["b"]/np.sqrt(1 - ALPHA**2)
            S = TwoS_theory(n, lam1, lam2, p["rho"])
            _, eig = S.get_eigY()
            th.append(eig)
            bbp.append()
            runs = [top2(make(n, **p).naive_matrix)[0] for _ in range(sims)]  #top eigenval pf Y
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


#### a refaire 
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
        a = ALPHA * lam
        #b =
        runs = []
        for _ in range(sims):
            S = make(n, lam, lam, rho)
            Sth = TwoS_theory(n, lam, lam, rho)
    ### compute methods (called function in here are at the bottom of the class)
            S.compute_method(["naive"])
            V = S.get_theta("naive")
            _ , V1= Sth.get_eigY()
            v1 = V1[0]
            th
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
    heat("a", np.linspace(0.1, 3, 25), 2, "heat_a_rho.png")


def fig_phase(rhos=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0), n=500, reps=3, omax=0.7):
    grid = np.linspace(0.1, 3, 75)
    A, B = np.meshgrid(grid, grid)                      # A[j, i] = grid[i] (x), B[j, i] = grid[j] (y)
    for rho in rhos:
        Z = np.array([[sample_overlaps(n, a, b, rho, reps) for a in grid] for b in grid])  # (b, a, 2)
        rgb = np.zeros((*Z.shape[:2], 3)); rgb[..., :2] = np.clip(Z / omax, 0, 1)
        tp, tm = theta_pm(A, B, rho)  # juste garder thetaPlus
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

# ---------------------------------------------------------------- figures

def _comp_x_hat_draw(n, lam1, lam2, rho, mu):
    """One draw: top eigenvalue of A = Y2 - I + 2 mu F1(x) for the three choices of x
    (true x_hat, BBP surrogate, m1 x1). All three use the same Y2."""
    S = TwoSpikes(n, lam1, lam2, rho, mu=mu, alpha=ALPHA)
    with np.errstate(invalid="ignore"):          # np.where evaluates sqrt(<0) for lam1<1 (result is 0 anyway)
        m1 = float(theory.overlap_Y_P(lam1))    # BBP overlap of top eigvec of Y1 with x1
    x1 = S.get_x1()
    g = np.random.normal(0, 1, n)
    g_perp = g - (g @ x1) * x1                  # project out x1
    g_perp /= np.linalg.norm(g_perp)
    _, xhat = S.top_eigenvector(S.Y1)           # exact top eigvec of Y1 (eigh). NB S.compute_theta1()
                                                # (50 power iterations) is not converged for lam1 < ~1.3
    xs = [xhat,                                          # le vrai x_hat
          m1 * x1 + np.sqrt(1 - m1**2) * g_perp,         # x_hat BBP (proj + g)
          m1 * x1]                                       # m1 x1 (norm m1, not 1)
    return [np.linalg.eigvalsh(S.A(x, mu))[-1] for x in xs]


def fig_comp_x_hat(n=500, lam1=2.0, lam2=3.0, mu=0.5, rho=0.5, sims=50, n_pts=26):
    """Top eigenvalue of A = Y2 - I + 2 mu F1(x) for three choices of x:
      - the true x_hat (top eigenvector of Y1),
      - the BBP surrogate m1 x1 + sqrt(1 - m1^2) g_perp  (g_perp random, orthogonal to x1),
      - m1 x1 alone.
    Left: vs rho (lam1 fixed). Right: vs lam1 (rho fixed).
    Points: mean over `sims` draws; error bars: standard error of the mean (std / sqrt(sims)).
    Dashed: theory, theta + 1/theta - 1 with theta = theory.beta_value(mu, lam1, m1, lam2, rho).

    g_perp only needs to be orthogonal to x1: the noise part of the true x_hat comes from Z1,
    which is independent of x2. (Projecting out x2 as well made X^T X singular at rho = 1.)
    """
    sweeps = {"rho": np.linspace(0, 1, n_pts), "lam1": np.linspace(0.5, 4, n_pts)}
    base = dict(lam1=lam1, rho=rho)
    xlabels = {"rho": r"$\rho$", "lam1": r"$\lambda_1$"}
    fixed = {"rho": rf"$\lambda_1$={lam1}", "lam1": rf"$\rho$={rho}"}
    labels = [r"true $\hat x$ (top eigvec of $Y_1$)",
              r"$m_1 x_1 + \sqrt{1-m_1^2}\, g_\perp$",
              r"$m_1 x_1$"]
    colors = ["C0", "C2", "C3"]

    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (name, grid) in zip(axes, sweeps.items()):
        res = np.zeros((len(grid), sims, 3))    # (grid point, simulation, which x)
        for i, val in enumerate(grid):
            p = dict(base); p[name] = val
            for s in range(sims):
                res[i, s] = _comp_x_hat_draw(n, p["lam1"], lam2, p["rho"], mu)
        mean, sem = res.mean(1), res.std(1) / np.sqrt(sims)
        out[name] = (grid, mean, sem)
        for k in range(3):
            ax.errorbar(grid, mean[:, k], sem[:, k], color=colors[k], capsize=2, lw=1.2, label=labels[k])
        # theory: theta = top eigenvalue of beta, mapped through BBP: theta + 1/theta - 1
        fine = np.linspace(grid[0], grid[-1], 300)
        pred = []
        for val in fine:
            p = dict(base); p[name] = val
            with np.errstate(invalid="ignore"):
                m1 = float(theory.overlap_Y_P(p["lam1"]))
            th = theory.beta_value(mu, p["lam1"], m1, lam2, p["rho"])
            pred.append(th + 1 / th - 1 if th > 1 else 1.0)
        ax.plot(fine, pred, "k--", lw=1.5,
                label=r"theory: $\theta + 1/\theta - 1$, $\theta = \theta_{\max}(\beta)$")
        if name == "lam1":
            ax.axvline(1, color="red", ls=":", label=r"BBP threshold $\lambda_1=1$")
        ax.set_xlabel(xlabels[name])
        ax.set_ylabel(r"top eigenvalue of $A$")
        ax.set_title(rf"{fixed[name]}, $\lambda_2$={lam2}, $\mu$={mu}, $n$={n}")
        ax.grid(); ax.legend(fontsize=8)
    fig.suptitle(f"all the x and theory over {sims} simulations")
    fig.tight_layout(); fig.savefig(os.path.join(DIR_PLOTS, "comp_x_hat.png"), dpi=120)
    return out


if __name__ == "__main__":
    todo = sys.argv[1:] or ["eig", "lam", "heat", "phase", "comp_x_hat"]
    if "eig" in todo: fig_eig()
    if "lam" in todo: fig_lambda()
    if "heat" in todo: fig_heat()
    if "phase" in todo: fig_phase()
    if "comp_x_hat" in todo: fig_comp_x_hat()
    plt.show()
