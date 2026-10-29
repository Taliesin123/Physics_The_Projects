"""
Ablation : Fisher regularisation vs plain isotropic L2 anchor at x_A*.

Stage B of  fisher_regularization.py  uses a penalty
        (mu / 2) (x - x_A*)^T F(x_A*) (x - x_A*)
with  F(x_A*) = ((lambda_1 - 1)^2 + 1) x_A* x_A*^T + I.

A natural ablation is to replace F by the identity :
        (mu / 2) || x - x_A* ||^2.
That is "anchor at x_A*, isotropically."  If EWC strongly outperforms
L2 we have learned that the anisotropy of Fisher — not merely the act
of anchoring at x_A* — is what's doing the work.

For each mu in a grid, this script runs n_repeats random realisations
of the two-spike problem and records the overlap-with-x_1 of both
methods.  A .npz is written for the plotting script.

Only this file is created.
"""

from __future__ import annotations

import os
import time
import numpy as np
from multiprocessing import Pool

import model_2 as m
import fisher_regularization as fr


# -----------------------------------------------------------------------------
#  Projected GD with an *arbitrary* anchor matrix M (replaces Fisher)
# -----------------------------------------------------------------------------
def gd_anchored(Y1, Y2, alpha, x_star, M_anchor, mu,
                iterations=300, lr=0.1):
    """
    Stage B with a generic positive-semidefinite anchor matrix M :
        minimise  L_complete(x) + (mu/2) (x - x_star)^T M (x - x_star)
    on the unit sphere.  Set M = Fisher for EWC, M = I for plain L2.
    """
    x = x_star.copy()
    x /= np.linalg.norm(x)
    for _ in range(iterations):
        g = (fr.complete_grad(x, Y1, Y2, alpha)
             + mu * (M_anchor @ (x - x_star)))
        x = x - lr * g
        x /= np.linalg.norm(x)
    return x


# -----------------------------------------------------------------------------
def _one_point(args):
    (mu, N, lambda1, lambda2, rho, alpha,
     iter_A, iter_B, lr, n_repeats, seed_base) = args

    rng = np.random.default_rng(seed_base)
    fisher_o1, fisher_o2 = [], []
    l2_o1,     l2_o2     = [], []
    naive_o1,  naive_o2  = [], []

    n_eye = np.eye(N)

    for _ in range(n_repeats):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        prob = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
        Y1, Y2 = prob.Y1, prob.Y2
        x1, x2 = prob.spike1, prob.spike2

        v_naive = fr.naive_spectral(Y1, Y2, alpha)
        x_A, _  = fr.gd_misspecified(Y1, lr=lr, iterations=iter_A)

        F = fr.fisher_matrix(x_A, lamba=lambda1)
        x_fisher = gd_anchored(Y1, Y2, alpha, x_A, F,  mu,
                               iterations=iter_B, lr=lr)
        x_l2     = gd_anchored(Y1, Y2, alpha, x_A, n_eye, mu,
                               iterations=iter_B, lr=lr)

        naive_o1.append(fr.overlap(v_naive, x1))
        naive_o2.append(fr.overlap(v_naive, x2))
        fisher_o1.append(fr.overlap(x_fisher, x1))
        fisher_o2.append(fr.overlap(x_fisher, x2))
        l2_o1.append(fr.overlap(x_l2, x1))
        l2_o2.append(fr.overlap(x_l2, x2))

    return (mu,
            np.array(naive_o1), np.array(naive_o2),
            np.array(fisher_o1), np.array(fisher_o2),
            np.array(l2_o1),     np.array(l2_o2))


# -----------------------------------------------------------------------------
def run_l2_comparison(
        N: int = 400,
        lambda1: float = 2.5,
        lambda2: float = 2.5,
        rho: float = 0.6,
        alpha: float = np.sqrt(0.5),
        mus=np.array([0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]),
        iter_A: int = 300,
        iter_B: int = 300,
        lr: float = 0.1,
        n_repeats: int = 25,
        seed: int = 0,
        parallel: bool = True,
        save_path: str | None = None,
):
    mus = np.asarray(mus, dtype=float)
    print(f"L2 vs Fisher  ({mus.size} mu values)  "
          f"rho={rho}  lambda1={lambda1}  lambda2={lambda2}")

    rng = np.random.default_rng(seed)
    args = [(float(mu), N, lambda1, lambda2, rho, alpha,
             iter_A, iter_B, lr, n_repeats,
             int(rng.integers(0, 2**31 - 1)))
            for mu in mus]

    t0 = time.time()
    if parallel:
        with Pool() as pool:
            results = pool.map(_one_point, args)
    else:
        results = [_one_point(a) for a in args]
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")

    stack = lambda k: np.stack([r[k] for r in results], axis=0)
    data = dict(
        mus=mus,
        naive_o1=stack(1),  naive_o2=stack(2),
        fisher_o1=stack(3), fisher_o2=stack(4),
        l2_o1=stack(5),     l2_o2=stack(6),
        N=N, lambda1=lambda1, lambda2=lambda2, rho=rho,
        alpha=alpha, n_repeats=n_repeats,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **data)
        print(f"  saved: {save_path}")
    return data


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Data", "Fisher")

    # We run the ablation at two correlations to see if the gap depends
    # on rho (we expect Fisher to increasingly beat L2 as rho grows).
    for rho in [0.3, 0.6, 0.85]:
        run_l2_comparison(
            N=300, lambda1=2.5, lambda2=2.5,
            rho=rho, alpha=np.sqrt(0.5),
            mus=np.array([0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]),
            iter_A=250, iter_B=250,
            n_repeats=20, seed=int(100 * rho),
            parallel=True,
            save_path=os.path.join(data_dir,
                                   f"l2_comparison_rho{rho:.2f}.npz"),
        )
