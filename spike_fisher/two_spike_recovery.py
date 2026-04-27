"""
Two-spike recovery :  naive spectral  vs  Fisher-deflation.

Goal
----
Produce an estimator that beats the naive spectral method for BOTH x_1
and x_2, not just x_1.

Procedure
---------
Let  M(alpha) = alpha * Y_1 + sqrt(1 - alpha^2) * Y_2.

Step 1  (identical for both methods) :
        x_hat_1  =  top eigenvector of  M(alpha).
        (This is the usual "naive spectral" estimator of the first
        spike --- it is good for x_1 and poor for x_2.)

Step 2a (naive baseline) :
        x_hat_2^naive  =  second eigenvector of  M(alpha).
        This is the natural "take the next best direction" choice when
        one only uses eigenstructure.  It lies in the x_1-x_2 plane
        but is severely rotated away from x_2 when rho grows.

Step 2b (Fisher method) :
        Minimise on the unit sphere
            (1/2) || Y_2 - x x^T ||_F^2  +  (mu / 2)  x^T F(x_hat_1) x
        with  F(x_hat_1) = ((lambda_1 - 1)^2 + 1) x_hat_1 x_hat_1^T + I.

        The Fisher penalty  x^T F x  has eigenvalue ((lambda_1-1)^2+2)
        along x_hat_1 and eigenvalue 1 orthogonal to it.  On the unit
        sphere only the "along x_hat_1" part is discriminative, so the
        effect is a Fisher-anisotropic *repulsion* away from x_hat_1.
        The Y_2 data term then pulls the estimator toward the dominant
        direction of Y_2 inside the complement of x_hat_1, which is
        exactly where (the orthogonal component of) x_2 lives.

Output
------
ONE bar chart.  4 bars in two groups (step 1, step 2); each group has
two bars (naive, Fisher).  The step-1 bars are identical by design.

Only this file is created.  No existing file is modified.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

import model_2 as m
import fisher_regularization as fr


# -----------------------------------------------------------------------------
#  Step-1 estimator  (identical for both methods)
# -----------------------------------------------------------------------------
def step1_naive_spectral(Y1: np.ndarray, Y2: np.ndarray,
                         alpha: float) -> np.ndarray:
    """ Top eigenvector of  M(alpha) = alpha Y_1 + sqrt(1 - alpha^2) Y_2. """
    M = alpha * Y1 + np.sqrt(1.0 - alpha ** 2) * Y2
    _, vecs = np.linalg.eigh(M)
    return vecs[:, -1]


# -----------------------------------------------------------------------------
#  Step-2 estimators
# -----------------------------------------------------------------------------
def step2_naive_second_eig(Y1: np.ndarray, Y2: np.ndarray,
                           alpha: float) -> np.ndarray:
    """
    The "naive" second-spike estimate : the *second* top eigenvector of
    the same combined matrix  M(alpha).  This is the standard naive
    deflation strategy when one only has eigendecomposition at hand.
    """
    M = alpha * Y1 + np.sqrt(1.0 - alpha ** 2) * Y2
    _, vecs = np.linalg.eigh(M)
    return vecs[:, -2]


def step2_fisher(Y2: np.ndarray, x_hat1: np.ndarray, lambda1: float,
                 mu: float = 1.0, iterations: int = 500,
                 lr: float = 0.05) -> np.ndarray:
    """
    Fisher-deflation step 2.  Projected GD on the unit sphere :

        min  (1/2) || Y_2 - x x^T ||_F^2  +  (mu / 2)  x^T F(x_hat_1) x
         x, ||x|| = 1

    where  F(x_hat_1) = ((lambda_1 - 1)^2 + 1) x_hat_1 x_hat_1^T + I.
    """
    n = Y2.shape[0]
    F = fr.fisher_matrix(x_hat1, lamba=lambda1)

    # Warm-start orthogonal to x_hat_1 (random direction in the complement)
    x = np.random.normal(0.0, 1.0, n)
    x = x - (x @ x_hat1) * x_hat1
    if np.linalg.norm(x) < 1e-12:
        x = np.random.normal(0.0, 1.0, n)
    x /= np.linalg.norm(x)

    for _ in range(iterations):
        g = fr.misspecified_grad(x, Y2) + mu * (F @ x)
        x = x - lr * g
        x /= np.linalg.norm(x)
    return x


# -----------------------------------------------------------------------------
#  Experiment
# -----------------------------------------------------------------------------
def run(N: int = 400,
        lambda1: float = 2.5,
        lambda2: float = 2.5,
        rho: float = 0.6,
        alpha: float = np.sqrt(0.5),
        mu: float = 1.0,
        iterations: int = 500,
        lr: float = 0.05,
        n_repeats: int = 30,
        seed: int = 0,
        save_path: str | None = None):
    """
    Run n_repeats independent realisations and draw one bar chart with
    4 bars (2 groups x 2 methods).
    """
    rng = np.random.default_rng(seed)

    naive_o1, naive_o2   = [], []
    fisher_o1, fisher_o2 = [], []

    for _ in range(n_repeats):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        prob = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
        Y1, Y2 = prob.Y1, prob.Y2
        x1, x2 = prob.spike1, prob.spike2

        # Step 1  (shared)
        x_hat1 = step1_naive_spectral(Y1, Y2, alpha)

        # Step 2 --- naive  : second eigenvector of M(alpha)
        x_hat2_naive = step2_naive_second_eig(Y1, Y2, alpha)

        # Step 2 --- Fisher : projected GD with Fisher repulsion
        x_hat2_fisher = step2_fisher(Y2, x_hat1, lambda1,
                                     mu=mu, iterations=iterations, lr=lr)

        naive_o1.append(fr.overlap(x_hat1,        x1))
        naive_o2.append(fr.overlap(x_hat2_naive,  x2))
        fisher_o1.append(fr.overlap(x_hat1,       x1))   # shared
        fisher_o2.append(fr.overlap(x_hat2_fisher, x2))

    naive_o1, naive_o2   = np.array(naive_o1),  np.array(naive_o2)
    fisher_o1, fisher_o2 = np.array(fisher_o1), np.array(fisher_o2)

    print("=" * 70)
    print(f"  N={N}  lambda1={lambda1}  lambda2={lambda2}  "
          f"rho={rho}  alpha={alpha:.3f}  mu={mu}  "
          f"(n_repeats={n_repeats})")
    print("=" * 70)
    print(f"  step 1   naive   |<x_hat_1, x_1>|  =  "
          f"{naive_o1.mean():.3f}  +/-  "
          f"{naive_o1.std() / np.sqrt(n_repeats):.3f}")
    print(f"  step 1   Fisher  |<x_hat_1, x_1>|  =  "
          f"{fisher_o1.mean():.3f}  +/-  "
          f"{fisher_o1.std() / np.sqrt(n_repeats):.3f}   (same estimator)")
    print(f"  step 2   naive   |<x_hat_2, x_2>|  =  "
          f"{naive_o2.mean():.3f}  +/-  "
          f"{naive_o2.std() / np.sqrt(n_repeats):.3f}")
    print(f"  step 2   Fisher  |<x_hat_2, x_2>|  =  "
          f"{fisher_o2.mean():.3f}  +/-  "
          f"{fisher_o2.std() / np.sqrt(n_repeats):.3f}")

    # ------------------------ single bar chart ------------------------
    groups = ["Step 1  :  recover $x_1$", "Step 2  :  recover $x_2$"]
    naive_means  = [naive_o1.mean(),  naive_o2.mean()]
    naive_sems   = [naive_o1.std() / np.sqrt(n_repeats),
                    naive_o2.std() / np.sqrt(n_repeats)]
    fisher_means = [fisher_o1.mean(), fisher_o2.mean()]
    fisher_sems  = [fisher_o1.std() / np.sqrt(n_repeats),
                    fisher_o2.std() / np.sqrt(n_repeats)]

    xs = np.arange(len(groups))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=130)
    ax.bar(xs - w / 2, naive_means,  width=w, yerr=naive_sems,  capsize=4,
           color="tab:blue",  label="naive spectral")
    ax.bar(xs + w / 2, fisher_means, width=w, yerr=fisher_sems, capsize=4,
           color="tab:green", label="Fisher method")

    for i, (v, e) in enumerate(zip(naive_means, naive_sems)):
        ax.text(xs[i] - w / 2, v + e + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    for i, (v, e) in enumerate(zip(fisher_means, fisher_sems)):
        ax.text(xs[i] + w / 2, v + e + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylabel(r"overlap with true spike")
    ax.set_ylim(0, 1.1)
    ax.set_title(r"Two-spike recovery :  naive spectral  vs  Fisher"
                 rf"   ($\rho={rho}$,  $\mu={mu}$, "
                 rf" $\lambda_1={lambda1}$,  $\lambda_2={lambda2}$)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"\nFigure saved to : {save_path}")
    plt.close(fig)

    return {
        "naive_o1":  naive_o1,  "naive_o2":  naive_o2,
        "fisher_o1": fisher_o1, "fisher_o2": fisher_o2,
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(here, "..", "Plots", "Fisher")

    run(
        N=400, lambda1=2, lambda2=3,
        rho=0.2, alpha=np.sqrt(0.5),
        mu=1.0, iterations=500, lr=0.05,
        n_repeats=30, seed=0,
        save_path=os.path.join(plot_dir, "two_spike_recovery.png"),
    )
