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


# ---- 1. overlap vs n, one parameter set -------------------------------
fig, ax = plt.subplots()
plot_overlap_vs_n(ax, a=2, b=3, rho=0.5)
fig.savefig("overlap_vs_n.png", dpi=120)

# ---- 2. overlap vs n, varying a, b, rho -------------------------------
PARAMS = [(2, 3, 0.1), (2, 3, 0.9), (1, 4, 0.5), (3, 3, 0.5)]
fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 3.5), sharey=True)
for ax, (a, b, rho) in zip(axes, PARAMS):
    plot_overlap_vs_n(ax, a, b, rho)
fig.tight_layout(); fig.savefig("overlap_vs_n_params.png", dpi=120)

# ---- 3. eigenvalues of P vs a, b, rho: theory vs numerics -------------
n, SIMS = 500, 30                      # SIMS simulations per point, error bars = std
base = dict(a=2.0, b=3.0, rho=0.5)
sweeps = dict(a=np.linspace(0.1, 5, 40), b=np.linspace(0.1, 5, 40), rho=np.linspace(0, 1, 40))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, (name, grid) in zip(axes, sweeps.items()):
    p = dict(base)
    th, num = [], []
    for val in grid:
        p[name] = val
        runs = [make(n, **p).eigen_P()[0][:2] for _ in range(SIMS)]
        th.append(make(n, **p).theory_eig())
        num.append((np.mean(runs, 0), np.std(runs, 0)))
    th, num = np.array(th), np.array(num)          # num: (points, mean/std, 2)
    ax.plot(grid, th[:, 0], "k--", label="theory")
    ax.plot(grid, th[:, 1], "k--")
    ax.errorbar(grid, num[:, 0, 0], num[:, 1, 0], color="C0", capsize=2, lw=1, label=r"$\theta_+$ simulation")
    ax.errorbar(grid, num[:, 0, 1], num[:, 1, 1], color="C1", capsize=2, lw=1, label=r"$\theta_-$ simulation")
    others = ", ".join(f"{k}={v}" for k, v in base.items() if k != name)
    ax.set_xlabel(name); ax.set_ylabel("eigenvalue of P"); ax.set_title(f"n={n}, {others}")
    ax.grid(); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("eigenvalues_P.png", dpi=120)
plt.show()
