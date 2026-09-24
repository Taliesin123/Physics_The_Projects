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
from scipy.optimize import brentq


# ----------------------------------------------------------------------
# Constraint used by Method B
# ----------------------------------------------------------------------
def norm_constraint(lam, A, b, N):
    """Residual ||theta(lam)|| - 1 for theta solving (A - lam*I) theta = b.

    Kept for reference only. The robust solver below replaces the naive
    ``fsolve(norm_constraint, ...)`` call, which diverged through the poles
    of this function.
    """
    theta = np.linalg.solve(A - lam * np.eye(N), b)
    return np.linalg.norm(theta) - 1.0


def solve_norm_constrained(A, b, tol=1e-12):
    """Solve the unit-norm-constrained stationarity equation for Method B.

    Method B *maximizes* the regularized objective

        f(theta) = theta^T A theta + 2 b^T theta      subject to ||theta|| = 1,

    whose stationarity condition is ``(A - lam*I) theta = -b``, i.e.

        theta(lam) = (lam*I - A)^{-1} b = sum_k c_k / (lam - e_k) * u_k,

    with the symmetric eigendecomposition ``A = sum_k e_k u_k u_k^T`` and
    ``c_k = u_k . b``. The squared norm

        phi(lam) = ||theta(lam)||^2 = sum_k c_k^2 / (lam - e_k)^2

    has a double pole at every eigenvalue of ``A``.

    Which root to pick (this is the whole game)
    -------------------------------------------
    A constrained *maximizer* on the sphere requires the Lagrange multiplier
    to satisfy ``lam >= e_max`` so that ``(lam*I - A)`` is positive definite
    (the classic trust-region condition). On ``(e_max, +inf)`` the matrix is
    PD and ``phi`` is smooth and strictly *decreasing* from +inf (as
    ``lam -> e_max^+``) to 0 (as ``lam -> +inf``), so ``phi(lam) = 1`` has a
    *unique* root there -- the global maximizer, the top-eigenvector branch
    that actually recovers the signal.

    NOTE: an earlier version bracketed the root on ``(-inf, e_min)`` instead.
    That is the *minimization* branch: it returns the bottom eigenvector of
    ``A`` (pure noise), which is why Method B never recovered the signal --
    and why shrinking mu made it worse, since mu->0 sends theta straight to
    ``u_min(A)``. We now bracket the upper branch.

    The "hard case"
    ---------------
    If ``b`` has (almost) no component along the *top* eigenspace of ``A``,
    the pole of ``phi`` at ``e_max`` vanishes and ``phi(lam)`` stays bounded
    as ``lam -> e_max^+``. Its supremum ``L`` may be ``<= 1``, so there is no
    root above ``e_max``; the constrained maximizer then sits exactly at
    ``lam = e_max`` with a compensating component along the top eigenvector
    added so that ``||theta|| = 1``. Detected and handled explicitly.

    Returns
    -------
    (theta, lam) with ``theta`` of unit norm and ``lam >= e_max``.
    """
    # A is symmetric by construction (Y2, F1 and I are all symmetric).
    evals, evecs = np.linalg.eigh(A)            # ascending eigenvalues
    c = evecs.T @ b                             # b in the eigenbasis
    e_max = evals[-1]
    span = max(1.0, abs(e_max))

    def phi(lam):
        return np.sum((c / (lam - evals)) ** 2)

    # Top eigenspace (group near-degenerate eigenvalues) and how much of
    # b lives in it.
    deg = evals >= e_max - 1e-9 * span
    free = ~deg
    c_max_norm = np.linalg.norm(c[deg])

    def hard_case_solution():
        """Solution at ``lam = e_max`` plus a u_max component for unit norm."""
        if free.any():
            theta_perp = evecs[:, free] @ (c[free] / (e_max - evals[free]))
        else:
            theta_perp = np.zeros_like(b, dtype=float)
        n2 = float(theta_perp @ theta_perp)
        tau = np.sqrt(max(0.0, 1.0 - n2))       # fill the remaining norm
        theta = theta_perp + tau * evecs[:, -1]
        nrm = np.linalg.norm(theta)
        if nrm == 0.0:                          # b ~ 0: any unit vector works
            return evecs[:, -1], e_max
        return theta / nrm, e_max

    # ---- Hard case: phi cannot reach 1 above e_max -> no exterior root. ----
    if c_max_norm <= 1e-9 * max(1.0, np.linalg.norm(b)):
        L = (np.sum((c[free] / (e_max - evals[free])) ** 2)
             if free.any() else 0.0)
        if L <= 1.0:
            return hard_case_solution()
        # else a genuine exterior root exists; fall through to bracketing.

    # ---- Standard case: bracket the unique root in (e_max, +inf). ----------
    # Near end: approach the pole at e_max until phi > 1.
    lo = e_max + 1e-9 * span
    while phi(lo) <= 1.0:
        gap = lo - e_max
        if gap < 1e-15 * span:                  # pole too weak to reach 1
            return hard_case_solution()
        lo = e_max + gap * 0.1

    # Far end: far enough right that phi < 1.
    hi = e_max + span
    while phi(hi) >= 1.0:
        hi += span

    # Guaranteed sign change now: phi(hi) < 1 < phi(lo).
    lam = brentq(lambda l: phi(l) - 1.0, lo, hi,
                 xtol=tol, rtol=1e-14, maxiter=200)
    theta = evecs @ (c / (lam - evals))         # = (lam*I - A)^{-1} b
    return theta / np.linalg.norm(theta), lam


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
        self.a = alpha * lam1
        self.b = np.sqrt(1 - alpha ** 2) * lam2

        self.Y1, self.Y2, self.x1, self.x2 = self.two_correlated_spikes()

        self.theta = {"1": None, "naive": None, "A": None, "B": None}

        self.P = self.alpha *self.lam1 * np.outer(self.x1, self.x1) + np.sqrt(1 - self.alpha ** 2) *self.lam2 * np.outer(self.x2, self.x2)
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

    def get_x2(self):
            return self.x2

    def set_mu(self, mu):
        self.mu = mu
        return self.mu

    def get_P(self):
        return self.P

    #### Theory formulas
    def theory_eig(self):
        """Compute the theoretical eigenvalues of the matrix P."""
        lam1 = (self.a+self.b)/2 + np.sqrt((self.a-self.b)**2/4 + self.a*self.b*self.rho**2)
        lam2 = (self.a+self.b)/2 - np.sqrt((self.a-self.b)**2/4 + self.a*self.b*self.rho**2)
        # lam = (a+b)/2 +- sqrt((a-b)^2 /4 + ab rho^2)
        return lam1, lam2

    def theory_vp(self):
   
        a = self.a 
        b = self.b

        lam1, lam2 = self.theory_eig()

        beta1 = (lam1-a)/(a*self.rho)
        v1 = self.x1 + beta1 * self.x2
        v1 /= np.linalg.norm(v1)

        beta2 = (lam2-a)/(a*self.rho)
        v2 = self.x1 + beta2 * self.x2
        v2 /= np.linalg.norm(v2)

        return v1, v2

    #### theory analysis
    def eigen_P(self):  # déja trié décroissant : plus grand eigenval[0] et vect : eigenvect[:, 0]
        eigenvalues, eigenvectors =  np.linalg.eigh(self.P)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def eigen_Y(self):  # déja trié décroissant : plus grand eigenval[0] et vect : eigenvect[:, 0]
            eigenvalues, eigenvectors =  np.linalg.eigh(self.Y)
    
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    
            return eigenvalues, eigenvectors

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

    @staticmethod
    def top_eigenvector(M):
        """Eigenvector of the algebraically LARGEST eigenvalue of symmetric M.

        This is the spectral estimator we actually want for signal recovery
        (the planted spike sits at the most-positive eigenvalue). Plain
        ``power_iteration`` converges to the largest *magnitude* eigenvalue
        instead, so whenever the noise floor ``e_min`` is larger in magnitude
        than the signal ``e_max`` (e.g. Method A at small mu) it returns the
        bottom (noise) eigenvector. ``eigh`` avoids that trap entirely.
        """
        _, evecs = np.linalg.eigh(M)            # ascending eigenvalues
        return evecs[:, -1]

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
        # Naive spectral baseline: top eigenvector of a fixed blend of Y1, Y2
        # (independent of mu). Use the algebraically-largest eigenvector so it
        # tracks the signal rather than the larger-magnitude noise edge.
        self.theta["naive"] = self.top_eigenvector(self.naive_matrix)
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
            # Method A: algebraically-largest eigenvector of A (the signal
            # spike). power_iteration would grab the larger-magnitude noise
            # edge at small mu, so use eigh-based top_eigenvector instead.
            thetaA = self.top_eigenvector(A)

        if "B" in method:
            # Method B: norm-constrained solve (A - lam*I) theta = b
            b = 2 * mu * F1 @ self.theta["1"]
          
            thetaB, _ = solve_norm_constrained(A, b)
            # else:
            #     # direct (un-constrained) fallback; kept for reference
            #     if np.linalg.cond(A) < 1e12:
            #         thetaB = np.linalg.inv(A - np.eye(N)) @ b
            #     else:
            #         raise np.linalg.LinAlgError(
            #             f"A nearly singular (cond={np.linalg.cond(A):.2e})"
            #         )

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
        nrm = np.linalg.norm(theta)
        if nrm > 0:
            theta = theta / nrm          # compare unit vectors (both A and B)
        overlap1 = abs(np.dot(theta, self.x1))
        overlap2 = abs(np.dot(theta, self.x2))
        return overlap1, overlap2

    def Loss_eval(self, method):
        """Loss used for cross-validation: minus the sum of the overlaps."""
        o1, o2 = self.overlaps(method)
        return -(o1 + o2)

    def Loss_eval_2(self, method):
        """Rank-1 reconstruction loss of the estimate against BOTH observations.

        With ``P = theta theta^T`` (rank-1, unit-norm theta), this is the
        alpha-weighted squared Frobenius reconstruction error of Y1 and Y2:

            L = alpha * ||Y1 - P||_F^2 + sqrt(1 - alpha^2) * ||Y2 - P||_F^2.
        """
        theta = self.theta[method]
        L = (self.alpha * np.linalg.norm(self.Y1 - np.outer(theta, theta), 'fro') ** 2
             + np.sqrt(1 - self.alpha ** 2)
             * np.linalg.norm(self.Y2 - np.outer(theta, theta), 'fro') ** 2)
        return L


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
            score.append(S.Loss_eval_2(method))
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