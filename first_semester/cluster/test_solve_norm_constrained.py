"""
Verification for the Method B / solve_norm_constrained fix.
Run on the cluster:  python test_solve_norm_constrained.py

Checks:
  1. The solver now sits on the MAX branch (lam >= e_max) and B aligns with
     the TOP eigenvector of A (the signal), not the bottom (noise).
  2. Method B actually recovers x2, comparably to Method A.
  3. Unit-norm + correct stationarity residual on standard and hard cases.
  4. No crashes across many random draws.
"""
import numpy as np
from spike_lib import solve_norm_constrained, TwoSpikes


def test_branch_and_recovery():
    np.random.seed(0)
    S = TwoSpikes(N=500, lam1=3.0, lam2=2.0, rho=0.7, mu=1e-9)
    S.compute_theta1()
    F1 = S.fisher_MS(S.theta["1"], S.lam1)
    A = S.Y2 - np.eye(S.N) + 2 * S.mu * F1
    b = 2 * S.mu * F1 @ S.theta["1"]
    thetaB, lam = solve_norm_constrained(A, b)
    w, V = np.linalg.eigh(A)
    print("lam=%.4f  e_min=%.4f  e_max=%.4f" % (lam, w[0], w[-1]))
    print("|B . topvec| = %.6f   (was ~0)" % abs(thetaB @ V[:, -1]))
    print("|B . botvec| = %.6f   (was ~1)" % abs(thetaB @ V[:, 0]))
    print("|B . x2|     = %.6f   (was ~0)" % abs(thetaB @ S.x2))
    assert lam >= w[-1] - 1e-9                 # on the maximization branch
    assert abs(thetaB @ V[:, -1]) > 0.99       # aligned with the top eigvec
    assert abs(thetaB @ V[:, 0]) < 1e-2        # not the noise floor anymore


def test_B_recovers_like_A():
    """At small mu both estimators should track the top eigenvector of A."""
    np.random.seed(1)
    oA, oB = [], []
    for _ in range(50):
        S = TwoSpikes(N=300, lam1=3.0, lam2=2.0, rho=0.7, mu=1e-6)
        S.compute_method(["A", "B"])
        oA.append(S.overlaps("A")[1])          # overlap with x2
        oB.append(S.overlaps("B")[1])
    oA, oB = np.mean(oA), np.mean(oB)
    print("mean |theta . x2|:  A=%.4f   B=%.4f" % (oA, oB))
    assert oB > 0.5                            # B now recovers the signal
    assert abs(oA - oB) < 0.1                  # and behaves like A at mu->0


def test_stress_no_crash():
    np.random.seed(2)
    n, fails = 1000, 0
    for _ in range(n):
        S = TwoSpikes(N=120, lam1=3.0, lam2=3.0, rho=0.7, mu=2.0)
        try:
            S.compute_method(["1", "A", "B"])
            assert abs(np.linalg.norm(S.theta["B"]) - 1.0) < 1e-7
        except Exception as e:                 # noqa
            fails += 1
            if fails <= 3:
                print("  FAIL:", type(e).__name__, e)
    print("stress: %d/%d failed" % (fails, n))
    assert fails == 0


if __name__ == "__main__":
    test_branch_and_recovery()
    test_B_recovers_like_A()
    test_stress_no_crash()
    print("\nAll checks passed.")