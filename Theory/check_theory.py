"""Numerical check of the closed-form spectrum of P = a x1 x1^T + b x2 x2^T.

Produces three figures (saved next to this file):
  1. overlap_vs_n.png       : overlap |<v_theory, v_empirical>| of the two top eigenvectors vs n
  2. overlap_vs_n_params.png: same, for several (a, b, rho)
  3. eigenvalues_P.png      : two top eigenvalues of P, theory vs numerics, vs a, b and rho
Run:  python check_theory.py
"""
import numpy as np
import matplotlib.pyplot as plt
from spike_lib import TwoSpikes

ALPHA = np.sqrt(0.5)          # so that a = lam1/sqrt2, b = lam2/sqrt2
NS = [50, 100, 200, 500, 1000, 2000, 3000]
REPEATS = 5


def make(n, a, b, rho):
    """TwoSpikes with prescribed a, b (signature is TwoSpikes(N, lam1, lam2, rho))."""
    return TwoSpikes(n, a / ALPHA, b / np.sqrt(1 - ALPHA**2), rho, alpha=ALPHA)


def overlaps(n, a, b, rho):
    """Overlap of the 2 theoretical eigenvectors with the 2 empirical ones, averaged over REPEATS."""
    o = np.zeros((REPEATS, 2))
    for r in range(REPEATS):
        S = make(n, a, b, rho)
        _, vec = S.eigen_P()
        v1, v2 = S.theory_vp()
        o[r] = abs(v1 @ vec[:, 0]), abs(v2 @ vec[:, 1])
    return o.mean(0), o.std(0) / np.sqrt(REPEATS)


def plot_overlap_vs_n(ax, a, b, rho):
    res = np.array([overlaps(n, a, b, rho) for n in NS])   # shape (len(NS), 2, 2)
    for k, lab in enumerate([r"$v_+$", r"$v_-$"]):
        ax.errorbar(NS, res[:, 0, k], res[:, 1, k], marker="o", label=lab)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("overlap")
    ax.set_title(f"a={a}, b={b}, rho={rho}"); ax.legend(); ax.grid()



