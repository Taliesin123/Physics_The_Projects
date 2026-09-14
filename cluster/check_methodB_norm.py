"""
check_methodB_norm.py
=====================

Sanity check for Method B.

Method B chooses the Lagrange multiplier ``lam`` as the value for which the
solution of ``(A - lam*I) theta = b`` already has unit norm, i.e. the root of

    g(lam) = ||theta(lam)|| - 1,     theta(lam) = (A - lam*I)^{-1} b.

This script plots ``||theta(lam)||`` as a function of ``lam`` so we can SEE
that it equals 1 somewhere -- and, in particular, that there is a clean,
unique crossing of 1 to the LEFT of the smallest eigenvalue of A, which is the
root the solver uses.

Why that region? Writing A = sum_k e_k u_k u_k^T (symmetric) and c_k = u_k . b,

    ||theta(lam)||^2 = sum_k c_k^2 / (e_k - lam)^2,

which has a pole at every eigenvalue e_k. On (-inf, e_min) the matrix
A - lam*I is positive definite and the norm rises smoothly and monotonically
from 0 (lam -> -inf) to +inf (lam -> e_min^-), so it crosses 1 exactly once.
In the bulk (between poles) it also crosses 1, but those roots are not the
well-defined global solution.

It also prints the overlaps of Method B vs Method A so you can judge whether B
is actually recovering the spikes, not just whether the solver converged.

Output:  results/methodB_norm_vs_lambda.png
Run:     python check_methodB_norm.py
"""

import os

import numpy as np

import spike_lib as sl


# ======================================================================
# CONFIG  -- match these to run_AB_comparison.py
# ======================================================================
N = 500
LAMBDA1 = 3.0
LAMBDA2 = 2.0
RHO = 0.5
MU = LAMBDA2 / (2 * (1 - LAMBDA1) ** 2)   # = mu_opt ; try others too
SEED = 0

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ======================================================================
# Build the exact A and b that Method B solves, for one model instance.
# ======================================================================
def build_A_b(N, lam1, lam2, rho, mu, seed):
    np.random.seed(seed)
    S = sl.TwoSpikes(N, lam1, lam2, rho, mu=mu)
    S.compute_theta1()
    F1 = S.fisher_MS(S.theta["1"], S.lam1)
    A = S.Y2 - np.eye(N) + 2 * mu * F1
    b = 2 * mu * F1 @ S.theta["1"]
    return S, A, b


def theta_norm(lam, evals, c):
    """||theta(lam)|| via the eigenbasis (vectorised over lam)."""
    lam = np.atleast_1d(lam).astype(float)
    # shape (len(lam), N): c_k / (e_k - lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = c[None, :] / (evals[None, :] - lam[:, None])
        out = np.sqrt(np.sum(terms ** 2, axis=1))
    return out


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S, A, b = build_A_b(N, LAMBDA1, LAMBDA2, RHO, MU, SEED)
    evals, evecs = np.linalg.eigh(A)        # ascending
    c = evecs.T @ b
    e_min, e_max = evals[0], evals[-1]

    # The root Method B actually uses (left of the smallest eigenvalue).
    thetaB, lamB = sl.solve_norm_constrained(A, b)

    # --- numbers: is B any good? ---------------------------------------
    S.compute_method(["A", "B"])
    oA = S.overlaps("A")
    oB = S.overlaps("B")
    print(f"params: N={N}, lambda1={LAMBDA1}, lambda2={LAMBDA2}, "
          f"rho={RHO}, mu={MU:.4f}")
    print(f"spectrum of A: [e_min, e_max] = [{e_min:.3f}, {e_max:.3f}]")
    print(f"Method B Lagrange multiplier lam = {lamB:.4f}  (< e_min)")
    print(f"||theta_B(lam)|| = {np.linalg.norm(thetaB):.6f}  (should be 1)")
    print(f"overlaps  A: |x1|={oA[0]:.3f} |x2|={oA[1]:.3f}  "
          f"mean={0.5*(oA[0]+oA[1]):.3f}")
    print(f"overlaps  B: |x1|={oB[0]:.3f} |x2|={oB[1]:.3f}  "
          f"mean={0.5*(oB[0]+oB[1]):.3f}")

    # ======================================================================
    # Plot
    # ======================================================================
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel 1: full range (shows the forest of poles) ---------------
    lam_full = np.linspace(e_min - 2.0, e_max + 2.0, 8000)
    nrm_full = theta_norm(lam_full, evals, c)
    ax_full.plot(lam_full, nrm_full, lw=0.8, color="steelblue")
    ax_full.axhline(1.0, color="k", ls="--", lw=1, label=r"$\|\theta\|=1$")
    # mark the extreme eigenvalues (first / last poles)
    ax_full.axvline(e_min, color="gray", ls=":", lw=1,
                    label=r"$\lambda_{\min}(A),\ \lambda_{\max}(A)$")
    ax_full.axvline(e_max, color="gray", ls=":", lw=1)
    ax_full.plot([lamB], [1.0], "o", color="tomato", ms=8, zorder=5,
                 label="Method B root")
    ax_full.set_ylim(0, 4)
    ax_full.set_xlabel(r"$\lambda$")
    ax_full.set_ylabel(r"$\|\theta(\lambda)\|$")
    ax_full.set_title("Full range (poles at every eigenvalue of A)")
    ax_full.legend(loc="upper right", fontsize=9)
    ax_full.grid(alpha=0.3)

    # ---- Panel 2: zoom on the Method-B region (lam < e_min) ------------
    lam_lo = min(lamB, e_min) - 1.5 * abs(e_min - lamB) - 0.5
    lam_zoom = np.linspace(lam_lo, e_min - 1e-6, 4000)
    nrm_zoom = theta_norm(lam_zoom, evals, c)
    ax_zoom.plot(lam_zoom, nrm_zoom, lw=1.5, color="steelblue")
    ax_zoom.axhline(1.0, color="k", ls="--", lw=1, label=r"$\|\theta\|=1$")
    ax_zoom.axvline(e_min, color="gray", ls=":", lw=1,
                    label=r"$\lambda_{\min}(A)$ (first pole)")
    ax_zoom.plot([lamB], [1.0], "o", color="tomato", ms=9, zorder=5,
                 label=fr"Method B root  $\lambda={lamB:.3f}$")
    ax_zoom.set_ylim(0, 4)
    ax_zoom.set_xlabel(r"$\lambda$")
    ax_zoom.set_ylabel(r"$\|\theta(\lambda)\|$")
    ax_zoom.set_title(r"Method B region: $\lambda < \lambda_{\min}(A)$"
                      "  (unique, monotone crossing)")
    ax_zoom.legend(loc="upper left", fontsize=9)
    ax_zoom.grid(alpha=0.3)

    fig.suptitle(
        rf"$\|\theta(\lambda)\|$ for Method B  "
        rf"($N={N}$, $\lambda_1={LAMBDA1}$, $\lambda_2={LAMBDA2}$, "
        rf"$\rho={RHO}$, $\mu={MU:.3f}$)  "
        rf"-- B mean overlap {0.5*(oB[0]+oB[1]):.2f} vs "
        rf"A {0.5*(oA[0]+oA[1]):.2f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = os.path.join(OUTDIR, "methodB_norm_vs_lambda.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()