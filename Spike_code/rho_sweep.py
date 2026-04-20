"""
Sweep over the correlation rho between the two true spikes.

For each rho we run n_repeats independent realisations and record the
overlap with  x_1  (and, separately, with x_2) of
    * naive spectral        (top eigvec of  alpha Y1 + sqrt(1-alpha^2) Y2)
    * Stage A only          (GD on Y1 alone)
    * Stage B (EWC)         (complete loss + mu Fisher penalty)

The rationale is that the naive spectral method should degrade *fast*
with rho (once the two spikes are correlated, the top eigvec mixes them)
while Stage A and Stage B should be more robust.

A .npz is saved so the plot script can reload the arrays without
recomputing the sweep.

Only this file is created.  No existing file is modified.
"""

from __future__ import annotations

import os
import time
import numpy as np
from multiprocessing import Pool

import model_2 as m
import fisher_regularization as fr


# -----------------------------------------------------------------------------
def _one_rho(args):
    (rho, N, lambda1, lambda2, alpha, mu,
     iter_A, iter_B, lr, n_repeats, seed_base) = args

    rng = np.random.default_rng(seed_base)
    naive_o1, naive_o2 = [], []
    A_o1,     A_o2     = [], []
    B_o1,     B_o2     = [], []

    for _ in range(n_repeats):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        prob = m.TwoSpikedWignerMatrix(N, lambda1, lambda2, rho, alpha)
        Y1, Y2 = prob.Y1, prob.Y2
        x1, x2 = prob.spike1, prob.spike2

        v_naive = fr.naive_spectral(Y1, Y2, alpha)
        x_A, _  = fr.gd_misspecified(Y1, lr=lr, iterations=iter_A)
        F       = fr.fisher_matrix(x_A, lamba=lambda1)
        x_B, _  = fr.gd_ewc(Y1, Y2, alpha, x_A, F, mu,
                            lr=lr, iterations=iter_B)

        naive_o1.append(fr.overlap(v_naive, x1))
        naive_o2.append(fr.overlap(v_naive, x2))
        A_o1.append(fr.overlap(x_A, x1))
        A_o2.append(fr.overlap(x_A, x2))
        B_o1.append(fr.overlap(x_B, x1))
        B_o2.append(fr.overlap(x_B, x2))

    return (rho,
            np.array(naive_o1), np.array(naive_o2),
            np.array(A_o1),     np.array(A_o2),
            np.array(B_o1),     np.array(B_o2))


# -----------------------------------------------------------------------------
def run_rho_sweep(
        N: int = 400,
        lambda1: float = 2.5,
        lambda2: float = 2.5,
        alpha: float = np.sqrt(0.5),
        mu: float = 5.0,
        rhos: np.ndarray = np.linspace(0.0, 0.95, 15),
        iter_A: int = 300,
        iter_B: int = 300,
        lr: float = 0.1,
        n_repeats: int = 25,
        seed: int = 0,
        parallel: bool = True,
        save_path: str | None = None,
):
    rhos = np.asarray(rhos)
    print(f"rho sweep   {rhos.size} pts   mu={mu}  "
          f"lambda1={lambda1}  lambda2={lambda2}")
    rng = np.random.default_rng(seed)
    args = [(float(rho), N, lambda1, lambda2, alpha, mu,
             iter_A, iter_B, lr, n_repeats,
             int(rng.integers(0, 2**31 - 1)))
            for rho in rhos]

    t0 = time.time()
    if parallel:
        with Pool() as pool:
            results = pool.map(_one_rho, args)
    else:
        results = [_one_rho(a) for a in args]
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")

    # Stack : shape (n_rhos, n_repeats)
    stack = lambda k: np.stack([r[k] for r in results], axis=0)
    data = dict(
        rhos=rhos,
        naive_o1=stack(1), naive_o2=stack(2),
        A_o1=stack(3),     A_o2=stack(4),
        B_o1=stack(5),     B_o2=stack(6),
        N=N, lambda1=lambda1, lambda2=lambda2,
        alpha=alpha, mu=mu, n_repeats=n_repeats,
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
    run_rho_sweep(
        N=400, lambda1=2.5, lambda2=2.5,
        alpha=np.sqrt(0.5), mu=5.0,
        rhos=np.linspace(0.0, 0.95, 15),
        iter_A=300, iter_B=300,
        n_repeats=25, seed=0, parallel=True,
        save_path=os.path.join(data_dir, "rho_sweep.npz"),
    )
