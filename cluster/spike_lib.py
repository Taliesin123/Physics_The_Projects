"""
spike_lib.py
============

Reusable library for the two-spike signal-recovery study.

This file collects the *useful* code from the original ``model.py`` /
``nums_fisher.ipynb`` work: the ``TwoSpikes`` generative model, the two
Fisher-regularized estimators (Method A and Method B), and the helper
routines used to measure recovery quality (overlaps, cross-validation of mu,
and a 1D sweep engine).

It contains NO plotting and NO experiment-specific scripting, so it can be
imported safely on a compute cluster. All experiments live in the companion
driver file ``run_AB_comparison.py``.

------------------------------------------------------------------------
Model
------------------------------------------------------------------------
Two correlated planted signals x1, x2 (unit norm, correlation rho) are each
observed through their own spiked-Wigner matrix:

    Y1 = lambda1 * x1 x1^T + GOE noise
    Y2 = lambda2 * x2 x2^T + GOE noise

We first get a cheap estimate theta1 of x1 from Y1 (power iteration), build
the Fisher-information matrix F1 from theta1, and use it to *regularize* the
recovery of the second spike x2 from Y2 via the matrix

    A = Y2 - I + 2*mu*F1.

Two estimators are compared:

    Method A : top eigenvector of A (pure regularized PCA; ignores the
               linear prior-pull term).
    Method B : solution of the norm-constrained system (A - lam*I) theta = b,
               with b = 2*mu*F1@theta1 and lam the Lagrange multiplier chosen
               so that ||theta|| = 1.
"""

import numpy as np
from scipy.optimize import fsolve


# ----------------------------------------------------------------------
# Constraint used by Method B
# ----------------------------------------------------------------------
def norm_constraint(lam, A, b, N):
    """Residual ||theta(lam)|| - 1 for theta solving (A - lam*I) theta = b.

    Method B picks the Lagrange multiplier ``lam`` as the root of this
    function, i.e. the value for which the solution already has unit norm.
    """
    theta = np.linalg.solve(A - lam * np.eye(N), b)
    return np.linalg.norm(theta) - 1.0


# ----------------------------------------------------------------------
# Two-spike model + estimators
# ----------------------------------------------------------------------
class TwoSpikes:
    """Two correlated spikes observed through two spiked-Wigner matrices."""

    def __init__(self, N, lam1, lam2, rho, mu=1.0, alpha=np.sqrt(0.5)):
        self.N = N
        self.lam1 = lam1
        self.lam2 = lam2
        self.rho = rho
        self.alpha = alpha
        self.mu = mu

        self.Y1, self.Y2, self.x1, self.x2 = self.two_correlated_spikes()

        self.theta = {"1": None, "naive": None, "A": None, "B": None}

        # used only by the "naive" baseline estimator
        self.naive_matrix = (
            self.alpha * self.Y1 + np.sqrt(1 - self.alpha ** 2) * self.Y2
        )

    # ----- generative model -------------------------------------------
    def two_correlated_spikes(self):
        cov = [[1, self.rho], [self.rho, 1]]
        X = np.random.multivariate_normal([0, 0], cov, self.N)
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1 = x1 / np.linalg.norm(x1)
        x2 = x2 / np.linalg.norm(x2)
        Y1 = self.lam1 * np.outer(x1, x1) + self.symmetric_gaussian_matrix()
        Y2 = self.lam2 * np.outer(x2, x2) + self.symmetric_gaussian_matrix()
        return Y1, Y2, x1, x2

    def symmetric_gaussian_matrix(self):
        n = self.N
        G = np.random.normal(0, 1, (n, n))
        return (G + G.T) / np.sqrt(2 * n)

    # ----- utilities --------------------------------------------------
    def get_theta(self, method):
        return self.theta[method]

    def get_x1(self):
        return self.x1

    def set_mu(self, mu):
        self.mu = mu
        return self.mu

    @staticmethod
    def power_iteration(M, iterations=50, tol=1e-7):
        N = M.shape[0]
        x_hat = np.ones(N)
        for _ in range(iterations):
            x_new = M @ x_hat
            x_new /= np.linalg.norm(x_new)
            if np.linalg.norm(x_new - x_hat) < tol:
                break
            x_hat = x_new
        return x_hat

    def fisher_MS(self, x, lam):
        """Fisher information matrix F(x) used for the regularizer."""
        N = self.N
        return ((lam - 1) ** 2 + 1.0 / N) * np.outer(x, x) + np.eye(N) / N

    def Hess(self, theta):
        """Hessian of the Method-A objective at ``theta``."""
        return self.Y2 - np.eye(self.N) + 2 * self.mu * self.fisher_MS(theta, self.lam1)

    # ----- estimators -------------------------------------------------
    def compute_theta1(self):
        """Step-1 estimate of x1 from Y1 (top eigenvector)."""
        self.theta["1"] = self.power_iteration(self.Y1)
        return self.theta["1"]

    def compute_naive(self):
        self.theta["naive"] = self.power_iteration(self.naive_matrix)
        return self.theta["naive"]

    def x_fisher(self, method, solve=True):
        """Compute the Fisher-regularized estimators.

        Returns (thetaA, thetaB); the requested ones are filled, the others
        are left as zeros.
        """
        mu = self.mu
        N = self.N

        self.compute_theta1()
        F1 = self.fisher_MS(self.theta["1"], self.lam1)
        A = self.Y2 - np.eye(N) + 2 * mu * F1

        thetaA = np.zeros(N)
        thetaB = np.zeros(N)

        if "A" in method:
            # Method A: top eigenvector of A
            thetaA = self.power_iteration(A)

        if "B" in method:
            # Method B: norm-constrained solve (A - lam*I) theta = b
            b = 2 * mu * F1 @ self.theta["1"]
            if solve:
                lam = fsolve(norm_constraint, x0=-4, args=(A, b, N))[0]
                thetaB = np.linalg.solve(A - lam * np.eye(N), b)
            else:
                # direct (un-constrained) fallback; kept for reference
                if np.linalg.cond(A) < 1e12:
                    thetaB = np.linalg.inv(A - np.eye(N)) @ b
                else:
                    raise np.linalg.LinAlgError(
                        f"A nearly singular (cond={np.linalg.cond(A):.2e})"
                    )

        return thetaA, thetaB

    def compute_methodA(self):
        self.theta["A"], _ = self.x_fisher(method="A", solve=True)
        return self.theta["A"]

    def compute_methodB(self):
        _, self.theta["B"] = self.x_fisher(method="B", solve=True)
        return self.theta["B"]

    def compute_method(self, method):
        """Compute any subset of {"1", "naive", "A", "B"}."""
        if "1" in method:
            self.compute_theta1()
        if "naive" in method:
            self.compute_naive()
        if "A" in method:
            self.compute_methodA()
        if "B" in method:
            self.compute_methodB()

    # ----- quality measures -------------------------------------------
    def overlaps(self, method):
        """Absolute overlaps |theta . x1| and |theta . x2|."""
        theta = self.theta[method]
        overlap1 = abs(np.dot(theta, self.x1))
        overlap2 = abs(np.dot(theta, self.x2))
        return overlap1, overlap2

    def Loss_eval(self, method):
        """Loss used for cross-validation: minus the sum of the overlaps."""
        o1, o2 = self.overlaps(method)
        return -(o1 + o2)


# ----------------------------------------------------------------------
# Cross-validation of the regularization strength mu
# ----------------------------------------------------------------------
def cv_mu(MU, N, rho, lambda1, lambda2, M=10, method="A"):
    """Pick mu minimizing the mean recovery loss over M resamples.

    Returns (mu_opt, scores, scores_std).
    """
    scores = []
    scores_std = []
    for mu in MU:
        score = []
        for _ in range(M):
            S = TwoSpikes(N, lambda1, lambda2, rho, mu=mu)
            S.compute_method([method])
            score.append(S.Loss_eval(method))
        score = np.array(score)
        scores.append(np.mean(score))
        scores_std.append(np.std(score) / np.sqrt(M))
    mu_opt = MU[int(np.argmin(scores))]
    return mu_opt, np.array(scores), np.array(scores_std)


def mu_equilibrium(lambda1, lambda2):
    """Reference value mu_eq = lambda2 / (2 (1 - lambda1)^2)."""
    return lambda2 / (2 * (1 - lambda1) ** 2)


# ----------------------------------------------------------------------
# 1D sweep engine: overlaps of each method vs one varying parameter
# ----------------------------------------------------------------------
def comparison1D(vary_param, values, N, lambda1, lambda2, rho, mu,
                 M=20, method=("A", "B"), optimal_mu=False):
    """Sweep one parameter and measure the recovery quality of each method.

    Parameters
    ----------
    vary_param : str
        Which parameter to sweep: one of {"N", "lambda1", "lambda2",
        "rho", "mu"}.
    values : iterable
        Values taken by ``vary_param``.
    N, lambda1, lambda2, rho, mu : floats
        Default model parameters (the swept one is overwritten).
    M : int
        Number of independent resamples per point (for averaging).
    method : sequence of str
        Subset of {"A", "B", "naive"} to evaluate.
    optimal_mu : bool
        When sweeping something other than mu, if True cross-validate mu at
        each point; otherwise use the equilibrium value mu_eq.

    Returns
    -------
    dict with keys "values" and, per method m, the arrays
        f"{m}_overlap1", f"{m}_overlap1_std",
        f"{m}_overlap2", f"{m}_overlap2_std",
        f"{m}_sum",      f"{m}_sum_std".
    """
    method = list(method)
    out = {"values": np.asarray(values, dtype=float)}
    acc = {m: {"o1": [], "o1s": [], "o2": [], "o2s": [], "s": [], "ss": []}
           for m in method}

    for value in values:
        params = {"N": N, "lambda1": lambda1, "lambda2": lambda2,
                  "rho": rho, "mu": mu}
        params[vary_param] = value

        # choose mu when it is not the swept parameter
        if vary_param != "mu":
            mu_eq = mu_equilibrium(params["lambda1"], params["lambda2"])
            if optimal_mu:
                MU = np.linspace(mu_eq - mu_eq / 10, mu_eq + mu_eq / 10, 15)
                params["mu"], _, _ = cv_mu(
                    MU, int(params["N"]), params["rho"],
                    params["lambda1"], params["lambda2"], M=M, method="A")
            else:
                params["mu"] = mu_eq

        runs = {m: {"o1": [], "o2": []} for m in method}
        for _ in range(M):
            S = TwoSpikes(int(params["N"]), params["lambda1"],
                          params["lambda2"], params["rho"], params["mu"])
            S.compute_method(method)
            for m in method:
                o1, o2 = S.overlaps(m)
                runs[m]["o1"].append(o1)
                runs[m]["o2"].append(o2)

        for m in method:
            o1 = np.array(runs[m]["o1"])
            o2 = np.array(runs[m]["o2"])
            s = 0.5 * (o1 + o2)
            acc[m]["o1"].append(o1.mean())
            acc[m]["o1s"].append(o1.std() / np.sqrt(M))
            acc[m]["o2"].append(o2.mean())
            acc[m]["o2s"].append(o2.std() / np.sqrt(M))
            acc[m]["s"].append(s.mean())
            acc[m]["ss"].append(s.std() / np.sqrt(M))

    for m in method:
        out[f"{m}_overlap1"] = np.array(acc[m]["o1"])
        out[f"{m}_overlap1_std"] = np.array(acc[m]["o1s"])
        out[f"{m}_overlap2"] = np.array(acc[m]["o2"])
        out[f"{m}_overlap2_std"] = np.array(acc[m]["o2s"])
        out[f"{m}_sum"] = np.array(acc[m]["s"])
        out[f"{m}_sum_std"] = np.array(acc[m]["ss"])

    return out
