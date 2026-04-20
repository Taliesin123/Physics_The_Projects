"""
Phase diagram of the EWC / Fisher-regularised spike recovery method
over the (lambda_1, lambda_2) plane.

For each (lambda_1, lambda_2) we average n_repeats independent runs of
    * naive spectral
    * Stage A (GD on Y_1 only)
    * Stage B (complete loss + mu * Fisher penalty centred at Stage-A)

and store the mean absolute overlap with x_1 and with x_2.

Results are saved as a .npz so that plots can be regenerated later
without re-running the sweep.

Only this file is created.  It does NOT modify any other script.
"""

from __future__ import annotations

import os
import time
import numpy as np
from multiprocessing import Pool

import model_2 as m
import fisher_regularization as fr


# -----------------------------------------------------------------------------
#  Single (lambda_1, lambda_2) cell
# -----------------------------------------------------------------------------
def _cell(args):
    (lamba1, lamba2, N, rho, alpha,
     mu, iter_A, iter_B, lr, n_repeats, seed_base) = args

    rng = np.random.default_rng(seed_base)
    naive_o1, naive_o2 = [], []
    A_o1,     A_o2     = [], []
    B_o1,     B_o2     = [], []

    for rep in range(n_repeats):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        prob = m.TwoSpikedWignerMatrix(N, lamba1, lamba2, rho, alpha)
        Y1, Y2 = prob.Y1, prob.Y2
        x1, x2 = prob.spike1, prob.spike2

        v_naive = fr.naive_spectral(Y1, Y2, alpha)
        x_A, _  = fr.gd_misspecified(Y1, lr=lr, iterations=iter_A)
        F       = fr.fisher_matrix(x_A, lamba=lamba1)
        x_B, _  = fr.gd_ewc(Y1, Y2, alpha, x_A, F, mu,
                            lr=lr, iterations=iter_B)

        naive_o1.append(fr.overlap(v_naive, x1))
        naive_o2.append(fr.overlap(v_naive, x2))
        A_o1.append(fr.overlap(x_A, x1))
        A_o2.append(fr.overlap(x_A, x2))
        B_o1.append(fr.overlap(x_B, x1))
        B_o2.append(fr.overlap(x_B, x2))

    return (lamba1, lamba2,
            np.mean(naive_o1), np.mean(naive_o2),
            np.mean(A_o1),     np.mean(A_o2),
            np.mean(B_o1),     np.mean(B_o2))


# -----------------------------------------------------------------------------
#  Sweep
# -----------------------------------------------------------------------------
def run_phase_diagram(
        N: int = 200,
        rho: float = 0.6,
        alpha: float = np.sqrt(0.5),
        mu: float = 5.0,
        lambas1=np.linspace(0.1, 4.0, 12),
        lambas2=np.linspace(0.1, 4.0, 12),
        iter_A: int = 200,
        iter_B: int = 200,
        lr: float = 0.1,
        n_repeats: int = 6,
        seed: int = 0,
        parallel: bool = True,
        save_path: str | None = None,
):
    lambas1 = np.asarray(lambas1)
    lambas2 = np.asarray(lambas2)
    n1, n2 = lambas1.size, lambas2.size
    print(f"Sweep  ({n1} x {n2})   mu={mu}  rho={rho}  alpha={alpha:.3f}")

    rng = np.random.default_rng(seed)
    args = []
    for l1 in lambas1:
        for l2 in lambas2:
            args.append((l1, l2, N, rho, alpha, mu, iter_A, iter_B, lr,
                         n_repeats,
                         int(rng.integers(0, 2**31 - 1))))

    t0 = time.time()
    if parallel:
        with Pool() as pool:
            results = pool.map(_cell, args)
    else:
        results = [_cell(a) for a in args]
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")

    # re-shape into grids (axis 0 = lambda1, axis 1 = lambda2)
    naive_o1 = np.zeros((n1, n2)); naive_o2 = np.zeros((n1, n2))
    A_o1     = np.zeros((n1, n2)); A_o2     = np.zeros((n1, n2))
    B_o1     = np.zeros((n1, n2)); B_o2     = np.zeros((n1, n2))
    idx = {(float(l1), float(l2)): (i, j)
           for i, l1 in enumerate(lambas1)
           for j, l2 in enumerate(lambas2)}
    for r in results:
        i, j = idx[(float(r[0]), float(r[1]))]
        (naive_o1[i, j], naive_o2[i, j],
         A_o1[i, j], A_o2[i, j],
         B_o1[i, j], B_o2[i, j]) = r[2:]

    data = dict(
        lambas1=lambas1, lambas2=lambas2,
        naive_o1=naive_o1, naive_o2=naive_o2,
        A_o1=A_o1, A_o2=A_o2,
        B_o1=B_o1, B_o2=B_o2,
        N=N, rho=rho, alpha=alpha, mu=mu, n_repeats=n_repeats,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **data)
        print(f"  saved: {save_path}")
    return data


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Data", "Fisher")

    run_phase_diagram(
        N=200,
        rho=0.6,
        alpha=np.sqrt(0.5),
        mu=5.0,
        lambas1=np.linspace(0.1, 4.0, 12),
        lambas2=np.linspace(0.1, 4.0, 12),
        iter_A=200,
        iter_B=200,
        n_repeats=6,
        seed=0,
        parallel=True,
        save_path=os.path.join(data_dir, "phase_diagram_ewc.npz"),
    )
