"""
spike_lib_N.py
==============

Generalization of spike_lib.TwoSpikes (K=2) to an arbitrary number K of
correlated spikes/tasks ("N_task" below; lower-case n/N is reserved for the
SIGNAL dimension throughout, to avoid clashing with the original N).

------------------------------------------------------------------------
Model
------------------------------------------------------------------------
K = N_task unit-norm signals x_0, ..., x_{K-1} (dimension N) are drawn with
equicorrelation rho between every pair:

    Cov_ij = 1        if i == j
    Cov_ij = rho      if i != j

Each is observed through its own spiked-Wigner matrix, with the SAME
strength lam for every task:

    Y_t = lam * x_t x_t^T + GOE noise          for t = 0, ..., K-1.

Sequential (chain) recovery
----------------------------
theta_hat_0 = power_iteration(Y_0)                       (no regularizer yet)
For t = 1, ..., K-1, the Fisher matrix is evaluated AT THE PREVIOUS estimate
of the SAME method (this is the key generalization you asked for):

    F_{t-1}^m = fisher_MS(theta_hat_{t-1}^m, lam)         m in {"A", "B"}
    A_t^m     = Y_t - I + 2*mu^m*F_{t-1}^m

    Method A: theta_hat_t^A = top_eigenvector(A_t^A)
    Method B: b_t = 2*mu^B * F_{t-1}^B @ theta_hat_{t-1}^B
              theta_hat_t^B = solve_norm_constrained(A_t^B, b_t)

Loss (equal weights, as requested)
-----------------------------------
For cross-validating mu, the loss evaluates a candidate theta against ALL K
observations with EQUAL weight 1/sqrt(K) (generalizing the alpha-weighted
2-term loss of the original 2-spike code):

    L(theta) = (1/sqrt(K)) * sum_{t=0}^{K-1} ||Y_t - theta theta^T||_F^2
"""

import numpy as np
from scipy.optimize import brentq


# ----------------------------------------------------------------------
# Constraint used by Method B (dimension-agnostic, unchanged)
# ----------------------------------------------------------------------
def norm_constraint(lam, A, b, N):
    """Residual ||theta(lam)|| - 1 for theta solving (A - lam*I) theta = b.
    Kept for reference only -- see solve_norm_constrained for the real solver.
    """
    theta = np.linalg.solve(A - lam * np.eye(N), b)
    return np.linalg.norm(theta) - 1.0


def solve_norm_constrained(A, b, tol=1e-12):
    """Solve the unit-norm-constrained stationarity equation for Method B.

    Unchanged from the original 2-spike spike_lib.py -- the maths here is
    purely about a symmetric matrix A and a vector b, it does not depend on
    how many spikes/tasks the rest of the model has.
    """
    evals, evecs = np.linalg.eigh(A)            # ascending eigenvalues
    c = evecs.T @ b                             # b in the eigenbasis
    e_max = evals[-1]
    span = max(1.0, abs(e_max))

    def phi(lam):
        return np.sum((c / (lam - evals)) ** 2)

    deg = evals >= e_max - 1e-9 * span
    free = ~deg
    c_max_norm = np.linalg.norm(c[deg])

    def hard_case_solution():
        if free.any():
            theta_perp = evecs[:, free] @ (c[free] / (e_max - evals[free]))
        else:
            theta_perp = np.zeros_like(b, dtype=float)
        n2 = float(theta_perp @ theta_perp)
        tau = np.sqrt(max(0.0, 1.0 - n2))
        theta = theta_perp + tau * evecs[:, -1]
        nrm = np.linalg.norm(theta)
        if nrm == 0.0:
            return evecs[:, -1], e_max
        return theta / nrm, e_max

    if c_max_norm <= 1e-9 * max(1.0, np.linalg.norm(b)):
        L = (np.sum((c[free] / (e_max - evals[free])) ** 2)
             if free.any() else 0.0)
        if L <= 1.0:
            return hard_case_solution()

    lo = e_max + 1e-9 * span
    while phi(lo) <= 1.0:
        gap = lo - e_max
        if gap < 1e-15 * span:
            return hard_case_solution()
        lo = e_max + gap * 0.1

    hi = e_max + span
    while phi(hi) >= 1.0:
        hi += span

    lam = brentq(lambda l: phi(l) - 1.0, lo, hi,
                 xtol=tol, rtol=1e-14, maxiter=200)
    theta = evecs @ (c / (lam - evals))
    return theta / np.linalg.norm(theta), lam


# ----------------------------------------------------------------------
# N-task model + chain estimators
# ----------------------------------------------------------------------
class NSpikes:
    """K = N_task correlated spikes, equicorrelation rho, common strength lam.

    NOTE: kept the class name ``TwoSpikes`` for drop-in continuity with your
    code, but it now models K = N_task spikes (K=2 is just a special case).
    """

    def __init__(self, N, N_task, lam, rho, mu=1.0):
        self.N = N                  # signal dimension
        self.N_task = N_task        # number K of correlated spikes
        self.lam = lam              # SAME lambda for every task
        self.rho = rho
        self.muA = mu
        self.muB = mu

        self.Y, self.x = self.correlated_spikes()

        # theta[t][m] = estimate of x_t produced at chain-step t by method m
        self.theta = [{"naive": None, "A": None, "B": None}
                      for _ in range(N_task)]

        # Base case: no regularizer available yet at t=0, plain spectral
        # estimate of x_0 seeds the chain for every method.
        theta0 = self.power_iteration(self.Y[:, :, 0])
        for m in ("naive", "A", "B"):
            self.theta[0][m] = theta0

        # Single naive baseline: equal-weight (1/sqrt(K)) sum of ALL Y_t,
        # independent of mu and of the chain -- same theta reused at every t.
        

    # ----- generative model -------------------------------------------
    def correlated_spikes(self):
        """K = N_task unit-norm signals (dim N) with equicorrelation rho.

        IMPORTANT: the covariance is over the K TASKS (shape (K, K)), and
        we draw N i.i.d. samples (size=N) -- this is the K-task analogue of
        the original 2-spike ``cov = [[1, rho], [rho, 1]]; size=self.N``.
        """
        K = self.N_task
        cov = np.full((K, K), self.rho)
        np.fill_diagonal(cov, 1.0)

        X = np.random.multivariate_normal(
            mean=np.zeros(K), cov=cov, size=self.N
        )                                          # shape (N, K), no .T needed
        X = X / np.linalg.norm(X, axis=0, keepdims=True)   # unit-norm columns

        Y = np.empty((self.N, self.N, K))
        for t in range(K):
            Y[:, :, t] = self.lam * np.outer(X[:, t], X[:, t]) \
                + self.symmetric_gaussian_matrix()

        return Y, X

    def symmetric_gaussian_matrix(self):
        n = self.N
        G = np.random.normal(0, 1, (n, n))
        return (G + G.T) / np.sqrt(2 * n)

    def naive_matrix(self,t):
        """Equal-weight (1/sqrt(K)) sum of ALL Y_t -- matches the equal-
        weighted Loss requested, used as a fixed (mu-independent) baseline."""
        return np.sum(self.Y[:, :, :t], axis=2) / np.sqrt(t + 1)

    # ----- utilities --------------------------------------------------
    def get_theta(self, method, t):
        return self.theta[t][method]

    def get_x(self):
        return self.x

    def set_mu(self, muA, muB):
        self.muA = muA
        self.muB = muB
        return self.muA, self.muB

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
        """Eigenvector of the algebraically LARGEST eigenvalue of symmetric M."""
        _, evecs = np.linalg.eigh(M)
        return evecs[:, -1]

    def fisher_MS(self, x, lam):
        """Fisher information matrix F(x) used for the regularizer."""
        N = self.N
        return ((lam - 1) ** 2 + 1.0 / N) * np.outer(x, x) + np.eye(N) / N

    # ----- chain estimators ---------------------------------------------
    def x_fisher(self, method, t):
        """One chain step: Fisher matrix evaluated at theta_hat_{t-1} of the
        SAME method, regularizing recovery of x_t from Y_t.

        Returns (thetaA, thetaB); only the requested ones are filled.
        """
        N = self.N
        thetaA = np.zeros(N)
        thetaB = np.zeros(N)

        if "A" in method:
            F_prev = self.fisher_MS(self.theta[t - 1]["A"], self.lam)
            A_t = self.Y[:, :, t] - np.eye(N) + 2 * self.muA * F_prev
            thetaA = self.top_eigenvector(A_t)

        if "B" in method:
            F_prev = self.fisher_MS(self.theta[t - 1]["B"], self.lam)
            A_t = self.Y[:, :, t] - np.eye(N) + 2 * self.muB * F_prev
            b = 2 * self.muB * F_prev @ self.theta[t - 1]["B"]
            thetaB, _ = solve_norm_constrained(A_t, b)

        return thetaA, thetaB
    
    def compute_naive(self, t):
        self.theta[t]["naive"] = self.top_eigenvector(self.naive_matrix(t))
        return self.theta[t]["naive"]

    def compute_methodA(self, t):
        self.theta[t]["A"], _ = self.x_fisher(method="A", t=t)
        return self.theta[t]["A"]

    def compute_methodB(self, t):
        _, self.theta[t]["B"] = self.x_fisher(method="B", t=t)
        return self.theta[t]["B"]

    def compute_method(self, method):
        """Run the chain (t = 1 .. N_task-1) for any subset of {"A", "B"}.

        "naive" needs no per-step computation -- it was already filled once
        in __init__ (see ``_theta_naive`` / ``naive_matrix``).
        """
        for t in range(1, self.N_task):
            if "A" in method:
                self.compute_methodA(t)
            if "B" in method:
                self.compute_methodB(t)

    # ----- quality measures -------------------------------------------
    def overlaps(self, method, time=0):
        """Overlap of the CURRENT estimator theta_hat_time against ALL
        targets seen so far (x_0, ..., x_time):

            overlap[m][s] = |theta_hat_time^m . x_s|     for s = 0 .. time

        Returns dict method -> array of shape (time + 1,).
        """
        overlap = {m: np.zeros(time+1) for m in method} # time +1
        for m in method:
            theta = self.theta[time][m]
            nrm = np.linalg.norm(theta)
            theta_unit = theta / nrm if nrm > 0 else theta
            for s in range(time+1): # time +1
                overlap[m][s] = abs(np.dot(theta_unit, self.x[:, s]))**2
        return overlap

    def overlaps_cumulative(self, method):
        """Overlap of the CURRENT estimator theta_hat_t against ALL targets
        seen so far (x_0, ..., x_t), averaged:
 
            overlap[m][t] = mean_{s=0}^{t} |theta_hat_t^m . x_s|
 
        This is the quantity that is expected to DECREASE with t when rho
        is small: a single vector theta_hat_t cannot stay well-aligned with
        many weakly-correlated targets at once. Contrast with ``overlaps``,
        which only checks theta_hat_t against its OWN target x_t (and so
        stays high as long as each Y_t alone is informative, regardless of
        the chain history).
 
        "naive" uses the single shared naive estimator at every t (so this
        differs from ``overlaps`` only in which targets are checked).
 
        Returns dict method -> array of shape (N_task,).
        """
        overlap = {m: np.zeros(self.N_task) for m in method}
        for m in method:
            for t in range(self.N_task):
                theta = self.theta[t][m]
                nrm = np.linalg.norm(theta)
                theta_unit = theta / nrm if nrm > 0 else theta
                # overlap against EVERY target seen so far (s = 0..t)
                o = np.abs(theta_unit @ self.x[:, :t])   # shape (t+1,)
                overlap[m][t] = o.mean()
        return overlap
    
    def Loss_eval_2(self,method, time = 0):
        """Equal-weight (1/sqrt(K)) reconstruction loss against ALL K
        observations, for a single estimate ``theta``:

            L(theta) = (1/sqrt(K)) * sum_t ||Y_t - theta theta^T||_F^2
        """
        theta = self.theta[time][method]
        nrm = np.linalg.norm(theta)
        theta_unit = theta / nrm if nrm > 0 else theta
        P = np.outer(theta, theta)
        total = sum(
            np.linalg.norm(self.Y[:, :, t] - P, "fro") ** 2
            for t in range(time +1)
        )
        return total / np.sqrt(time +1)


# ----------------------------------------------------------------------
# Cross-validation of the (single, shared) regularization strength mu
# ----------------------------------------------------------------------
def mu_equilibrium_N(lam):
    """Reference value for constant lambda across all tasks:
    mu_eq = lam / (2 (1 - lam)^2)  (generalizes lambda2/(2(1-lambda1)^2)
    to lambda1 = lambda2 = lam)."""
    return lam / (2 * (1 - lam) ** 2)


def cv_mu_N(MU, N, N_task, rho, lam, M=10, method="A"):
    """Pick mu (used as muA=muB=mu at EVERY chain step) minimizing the mean
    final-step reconstruction loss over M resamples.

    The candidate grid MU is typically built around mu_equilibrium_N(lam),
    e.g.:
        mu_eq = mu_equilibrium_N(lam)
        MU = mu_eq * np.linspace(0.25, 1.75, 15)

    Returns (mu_opt, scores, scores_std).
    """
    scores = []
    scores_std = []
    for mu in MU:
        score = []
        for _ in range(M):
            S = NSpikes(N, N_task, lam, rho, mu=mu)
            S.compute_method([method])
            score.append(S.Loss_eval_2(method, time=N_task - 1))
        score = np.array(score)
        scores.append(score.mean())
        scores_std.append(score.std() / np.sqrt(M))
    mu_opt = MU[int(np.argmin(scores))]
    return mu_opt, np.array(scores), np.array(scores_std)


# ----------------------------------------------------------------------
# Per-step cross-validation: a DIFFERENT mu_t for every chain step t
# ----------------------------------------------------------------------
def cv_mu_chain(N, N_task, rho, lam, n_mu_grid=15, mu_window=0.75, M=10,
                 methods=("A", "B"), seed=None):
    """Sequential, per-step CV: find mu_t (t=1..N_task-1), SEPARATELY for
    method A and method B, that minimizes the average reconstruction loss
    of theta_hat_t against Y_t ALONE:

        L_t(theta) = ||Y_t - theta theta^T||_F^2

    This is the only criterion available "locally" at step t in a
    sequential setting (it does not peek at future data). The grid of
    candidate mu is centered on mu_equilibrium_N(lam):

        MU_grid = mu_eq * linspace(1 - mu_window, 1 + mu_window, n_mu_grid)

    Efficiency note: for a fixed resample, the chain up to step t-1 does
    NOT depend on the candidate mu_t, so it is built ONCE per resample and
    reused for every candidate in MU_grid -- only the last step is redone
    for each candidate (cheap), instead of recomputing the whole chain
    n_mu_grid times.

    Returns
    -------
    mu_opt : dict method -> array of length N_task
             mu_opt[m][0] is unused (np.nan): step 0 has no regularizer.
             mu_opt[m][t] is the CV-selected mu for step t (t=1..N_task-1).
    """
    if seed is not None:
        np.random.seed(seed)

    mu_eq = mu_equilibrium_N(lam)
    MU_grid = mu_eq * np.linspace(max(1e-6, 1 - mu_window),
                                   1 + mu_window, n_mu_grid)
    MU_grid = MU_grid[MU_grid > 0]

    mu_opt = {m: np.full(N_task, np.nan) for m in methods}

    for m in methods:
        for t in range(1, N_task):
            scores = np.zeros(len(MU_grid))
            for _ in range(M):
                S = NSpikes(N, N_task, lam, rho, mu=mu_eq)

                # Build the chain up to t-1 using the ALREADY-DETERMINED
                # mu's from previous (earlier-t) CV iterations.
                for tp in range(1, t):
                    if m == "A":
                        S.muA = mu_opt["A"][tp]
                        S.compute_methodA(tp)
                    else:
                        S.muB = mu_opt["B"][tp]
                        S.compute_methodB(tp)

                # Try every candidate mu_t, reusing the SAME chain so far
                # (only the last step is recomputed each time).
                for k, mu in enumerate(MU_grid):
                    if m == "A":
                        S.muA = mu
                        theta_t, _ = S.x_fisher(method="A", t=t)
                    else:
                        S.muB = mu
                        _, theta_t = S.x_fisher(method="B", t=t)
                    L = np.linalg.norm(
                        S.Y[:, :, t] - np.outer(theta_t, theta_t), "fro"
                    ) ** 2
                    scores[k] += L

            scores /= M
            mu_opt[m][t] = MU_grid[int(np.argmin(scores))]

    return mu_opt