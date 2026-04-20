"""
Fisher-regularised recovery of the first spike from two correlated spikes.
==========================================================================

Setup
-----
We have two correlated unit spikes  x1, x2  (<x1, x2> ~ rho) and two
noisy observations :
        Y1 = G1 + lambda1 * x1 x1^T
        Y2 = G2 + lambda2 * x2 x2^T
where G1, G2 are independent Wigner matrices (symmetric Gaussian,
normalised as in model_2.WignerMatrix).

The "naive" spectral method looks at the top eigenvector of
        M(alpha) = alpha * Y1 + sqrt(1 - alpha^2) * Y2
and hopes it lies close to x1.  For non-zero rho the top eigenvector
mixes x1 and x2.

Idea (EWC-style)
----------------
Two-stage optimisation.  We keep   x   on the unit sphere  (||x|| = 1),
so projected gradient descent = normalise after every step, exactly
as in TwoSpikedWignerMatrix.gradient_descent_loss_1.

 * Stage A :  minimise the misspecified loss on Y1 alone,
                 L_A(x) = (1/2) || Y1 - x x^T ||_F^2,      ||x|| = 1,
          using (projected) gradient descent.  The optimum
          x_A*  points along the top eigenvector of Y1, which is close
          to x1 once lambda1 > 1 (BBP).

 * Stage B :  minimise the complete loss (on Y1 and Y2 together) plus
          an anisotropic L2 regulariser centred at  x_A*  and
          weighted by the Fisher matrix at  x_A* :

                 F(x_A*) = ((lambda1 - 1)^2 + 1) * x_A* x_A*^T + I

          The regulariser reads
                 (mu / 2) (x - x_A*)^T F(x_A*) (x - x_A*).

          This pins x strongly along the x_A*-direction (large Fisher
          eigenvalue = (lambda1 - 1)^2 + 2) while letting it move in
          the orthogonal subspace (Fisher eigenvalue = 1) to fit Y2.

This file does NOT modify any other file in the repo.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

import model_2 as m   # local module


# -----------------------------------------------------------------------------
#  Losses, gradients, Fisher matrix
# -----------------------------------------------------------------------------
def misspecified_loss(x: np.ndarray, Y: np.ndarray) -> float:
    """  L(x) = (1/2) || Y - x x^T ||_F^2                                  """
    R = Y - np.outer(x, x)
    return 0.5 * float(np.sum(R * R))


def misspecified_grad(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Gradient of  L(x) = (1/2) || Y - x x^T ||_F^2  w.r.t. x,  Y symmetric:

        grad = 2 * ( ||x||^2 * x  -  Y x ).
    """
    return 2.0 * ((x @ x) * x - Y @ x)


def complete_loss(x: np.ndarray, Y1: np.ndarray, Y2: np.ndarray,
                  alpha: float) -> float:
    """
    Complete loss combining both observations :
        L(x) = alpha * L_1(x) + sqrt(1 - alpha^2) * L_2(x).
    """
    a1, a2 = alpha, np.sqrt(1.0 - alpha ** 2)
    return a1 * misspecified_loss(x, Y1) + a2 * misspecified_loss(x, Y2)


def complete_grad(x: np.ndarray, Y1: np.ndarray, Y2: np.ndarray,
                  alpha: float) -> np.ndarray:
    a1, a2 = alpha, np.sqrt(1.0 - alpha ** 2)
    return a1 * misspecified_grad(x, Y1) + a2 * misspecified_grad(x, Y2)


def fisher_matrix(x_hat: np.ndarray, lamba: float) -> np.ndarray:
    """
    Fisher information at  x_hat  for the misspecified model (user's formula)

        F(x_hat) = ((lambda - 1)^2 + 1) * x_hat x_hat^T + I.

    Here we interpret  x_hat  as the stage-A optimum (unit norm).
    lamba = signal strength around which we linearise (typically lambda1).
    """
    n = x_hat.size
    coef = (lamba - 1.0) ** 2 + 1.0
    return coef * np.outer(x_hat, x_hat) + np.eye(n)


def ewc_penalty(x: np.ndarray, x_star: np.ndarray, F: np.ndarray) -> float:
    d = x - x_star
    return 0.5 * float(d @ (F @ d))


def ewc_penalty_grad(x: np.ndarray, x_star: np.ndarray,
                     F: np.ndarray) -> np.ndarray:
    return F @ (x - x_star)


# -----------------------------------------------------------------------------
#  Projected gradient descent routines  (x is kept unit-norm)
# -----------------------------------------------------------------------------
def gd_misspecified(Y: np.ndarray, x0: np.ndarray | None = None,
                    lr: float = 0.1, iterations: int = 500,
                    record_history: bool = False):
    """
    Stage A : projected gradient descent on  (1/2) ||Y - xx^T||_F^2
              on the unit sphere.

    Returns (x_final, history_or_None).
    """
    n = Y.shape[0]
    if x0 is None:
        x = np.random.normal(0.0, 1.0, n)
    else:
        x = x0.copy()
    x /= np.linalg.norm(x)

    history = [] if record_history else None
    for _ in range(iterations):
        if record_history:
            history.append(misspecified_loss(x, Y))
        x = x - lr * misspecified_grad(x, Y)
        x /= np.linalg.norm(x)
    return x, history


def gd_ewc(Y1: np.ndarray, Y2: np.ndarray, alpha: float,
           x_star: np.ndarray, F: np.ndarray, mu: float,
           x0: np.ndarray | None = None,
           lr: float = 0.1, iterations: int = 500,
           record_history: bool = False):
    """
    Stage B :  minimise
        L_complete(x) + (mu/2) (x - x_star)^T F (x - x_star)
    on the unit sphere (projected GD : normalise after every step).

    Warm-start :  x_0 = x_star  by default.

    x_star : Stage-A optimum (anchor)
    F      : Fisher matrix at x_star
    mu     : EWC regularisation weight
    """
    if x0 is None:
        x = x_star.copy()
    else:
        x = x0.copy()
    x /= np.linalg.norm(x)

    history = [] if record_history else None
    for _ in range(iterations):
        if record_history:
            history.append(
                complete_loss(x, Y1, Y2, alpha)
                + mu * ewc_penalty(x, x_star, F)
            )
        g = complete_grad(x, Y1, Y2, alpha) + mu * ewc_penalty_grad(x, x_star, F)
        x = x - lr * g
        x /= np.linalg.norm(x)
    return x, history


# -----------------------------------------------------------------------------
#  Baselines
# -----------------------------------------------------------------------------
def naive_spectral(Y1: np.ndarray, Y2: np.ndarray, alpha: float) -> np.ndarray:
    """ Top eigenvector of  M(alpha) = alpha Y1 + sqrt(1 - alpha^2) Y2. """
    M = alpha * Y1 + np.sqrt(1.0 - alpha ** 2) * Y2
    vals, vecs = np.linalg.eigh(M)
    return vecs[:, -1]


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
def overlap(u: np.ndarray, v: np.ndarray) -> float:
    """ Absolute cosine-similarity (robust to un-normalised inputs). """
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(abs(np.dot(u, v) / (nu * nv)))


# -----------------------------------------------------------------------------
#  Demo
# -----------------------------------------------------------------------------
def demo(N: int = 400,
         lambda1: float = 2.5,
         lambda2: float = 2.5,
         rho: float = 0.6,
         alpha: float = np.sqrt(0.5),
         lr_A: float = 0.1,
         iter_A: int = 300,
         lr_B: float = 0.1,
         iter_B: int = 300,
         mu: float = 2.0,
         seed: int | None = 0,
         save_path: str | None = None):
    """ Run a single experiment and report overlaps with x1 and x2. """
    if seed is not None:
        np.random.seed(seed)

    problem = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
    Y1, Y2 = problem.Y1, problem.Y2
    x1, x2 = problem.spike1, problem.spike2     # already unit-normalised

    # 1) naive spectral baseline
    v_naive = naive_spectral(Y1, Y2, alpha)

    # 2) Stage A : GD on L_1 only
    x_A, hist_A = gd_misspecified(
        Y1, lr=lr_A, iterations=iter_A, record_history=True
    )

    # 3) Fisher matrix at x_A (unit norm)
    F = fisher_matrix(x_A, lamba=lambda1)

    # 4) Stage B : GD on complete loss + Fisher penalty
    x_B, hist_B = gd_ewc(
        Y1, Y2, alpha, x_A, F, mu,
        lr=lr_B, iterations=iter_B, record_history=True,
    )

    results = {
        "overlap_naive_x1":  overlap(v_naive, x1),
        "overlap_naive_x2":  overlap(v_naive, x2),
        "overlap_stageA_x1": overlap(x_A, x1),
        "overlap_stageA_x2": overlap(x_A, x2),
        "overlap_stageB_x1": overlap(x_B, x1),
        "overlap_stageB_x2": overlap(x_B, x2),
    }

    print("=" * 68)
    print(f"  N={N}  lambda1={lambda1}  lambda2={lambda2}  "
          f"rho={rho}  alpha={alpha:.3f}  mu={mu}")
    print("=" * 68)
    for k, v in results.items():
        print(f"  {k:<22s} = {v:.4f}")

    # --------- figure ---------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)

    ax[0].plot(hist_A, label="Stage A  (L_1 on Y1)", color="tab:blue")
    ax[0].plot(hist_B, label="Stage B  (complete + Fisher)", color="tab:orange")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Training curves")
    ax[0].legend()
    ax[0].grid(alpha=0.4)

    labels = ["naive spectral", "Stage A (Y1 only)", "Stage B (Fisher)"]
    ov1 = [results["overlap_naive_x1"], results["overlap_stageA_x1"],
           results["overlap_stageB_x1"]]
    ov2 = [results["overlap_naive_x2"], results["overlap_stageA_x2"],
           results["overlap_stageB_x2"]]
    xs = np.arange(len(labels))
    ax[1].bar(xs - 0.18, ov1, width=0.35,
              label=r"$|\langle \hat x, x_1\rangle|$", color="tab:green")
    ax[1].bar(xs + 0.18, ov2, width=0.35,
              label=r"$|\langle \hat x, x_2\rangle|$", color="tab:red")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("overlap (unit-norm)")
    ax[1].set_title(rf"Overlaps with true spikes  "
                    rf"($\rho$={rho},  $\mu$={mu})")
    ax[1].legend()
    ax[1].grid(alpha=0.4, axis="y")

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")
    plt.close(fig)
    return results


# -----------------------------------------------------------------------------
#  Sweep over mu  (effect of the EWC weight)
# -----------------------------------------------------------------------------
def sweep_mu(mus=(0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
             N: int = 400, lambda1: float = 2.5, lambda2: float = 2.5,
             rho: float = 0.6, alpha: float = np.sqrt(0.5),
             n_repeats: int = 15, seed: int = 1,
             save_path: str | None = None):
    """
    For each mu, run `n_repeats` independent experiments and average
    stage-B overlaps with x1 and x2. Also show the naive-spectral overlap
    with x1 as a horizontal reference.
    """
    rng = np.random.default_rng(seed)
    mean_o1, std_o1 = [], []
    mean_o2, std_o2 = [], []
    naive_o1 = []

    for mu in mus:
        o1s, o2s, no1 = [], [], []
        for _ in range(n_repeats):
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            problem = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
            Y1, Y2 = problem.Y1, problem.Y2
            x1, x2 = problem.spike1, problem.spike2

            v_naive = naive_spectral(Y1, Y2, alpha)
            x_A, _  = gd_misspecified(Y1, lr=0.1, iterations=300)
            F       = fisher_matrix(x_A, lamba=lambda1)
            x_B, _  = gd_ewc(Y1, Y2, alpha, x_A, F, mu,
                             lr=0.1, iterations=300)

            o1s.append(overlap(x_B, x1))
            o2s.append(overlap(x_B, x2))
            no1.append(overlap(v_naive, x1))

        mean_o1.append(np.mean(o1s)); std_o1.append(np.std(o1s) / np.sqrt(n_repeats))
        mean_o2.append(np.mean(o2s)); std_o2.append(np.std(o2s) / np.sqrt(n_repeats))
        naive_o1.append(np.mean(no1))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
    ax.errorbar(mus, mean_o1, yerr=std_o1, marker="o",
                label=r"Stage B : $|\langle \hat x, x_1\rangle|$",
                color="tab:green", capsize=3)
    ax.errorbar(mus, mean_o2, yerr=std_o2, marker="s",
                label=r"Stage B : $|\langle \hat x, x_2\rangle|$",
                color="tab:red", capsize=3)
    ax.plot(mus, naive_o1, ls="--", color="tab:blue",
            label=r"naive spectral : $|\langle \hat x, x_1\rangle|$")
    ax.set_xlabel(r"EWC weight  $\mu$")
    ax.set_ylabel("overlap")
    ax.set_title(rf"Effect of Fisher regularisation  "
                 rf"($\lambda_1$={lambda1}, $\lambda_2$={lambda2}, "
                 rf"$\rho$={rho}, $\alpha$={alpha:.2f})")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.4)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(here, "..", "Plots", "Fisher")

    # 1) single-run demo
    demo(
        N=400, lambda1=2.5, lambda2=2.5,
        rho=0.1, alpha=np.sqrt(0.5),
        mu=2.0, seed=0,
        save_path=os.path.join(plot_dir, "fisher_demo.png"),
    )

    # 2) sweep over mu
    sweep_mu(
        mus=(0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        N=400, lambda1=2.5, lambda2=2.5,
        rho=0.6, alpha=np.sqrt(0.5),
        n_repeats=15, seed=1,
        save_path=os.path.join(plot_dir, "fisher_sweep_mu.png"),
    )
