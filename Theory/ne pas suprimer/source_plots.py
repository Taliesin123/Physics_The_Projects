import numpy as np
from class_twospike import TwoSpikes
import source_twospike as sts

def semi_circle(z): #takes an array of values, return the semi circle for those values
    z = np.asarray(z)
    result = np.zeros_like(z, dtype=float)

    mask = np.abs(z) < 2 - 1e-12
    result[mask] = np.sqrt(4 - z[mask]**2) / (2 * np.pi) 

    return result

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


def comparison2D(vary_param1, values1, vary_param2, values2,
                  N, lambda1, lambda2, rho, mu,
                  M=20, method=("A", "B"), optimal_mu=False):
    """Sweep two parameters and measure the recovery quality of each method.
    Returns
    -------
    dict avec "values1", "values2" (1D), et par méthode m les tableaux 2D
    (shape (len(values1), len(values2))) :
        f"{m}_overlap1", f"{m}_overlap2"
    (moyennes sur M tirages, pas de std).
    """
    if isinstance(method, str):
        method = [method]
    else:
        method = list(method)

    values1 = np.asarray(values1, dtype=float)
    values2 = np.asarray(values2, dtype=float)

    out = {"values1": values1, "values2": values2}

    shape = (len(values1), len(values2))
    grids = {m: {"o1": np.zeros(shape), "o2": np.zeros(shape)} for m in method}

    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            params = {"N": N, "lambda1": lambda1, "lambda2": lambda2,
                      "rho": rho, "mu": mu}
            params[vary_param1] = v1
            params[vary_param2] = v2

            # choix de mu s'il n'est balayé sur aucun des deux axes
            if vary_param1 != "mu" and vary_param2 != "mu":
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
                grids[m]["o1"][i, j] = np.mean(runs[m]["o1"])
                grids[m]["o2"][i, j] = np.mean(runs[m]["o2"])

    for m in method:
        out[f"{m}_overlap1"] = grids[m]["o1"]
        out[f"{m}_overlap2"] = grids[m]["o2"]

    return out


def plot_comparison2D(result, vary_param1, vary_param2, method=("A", "B"),
                       functions=None, cmap="viridis", vmin=0, vmax=1,
                       figsize=(11, 4.5)):
    """Plot the colormaps produced by comparison2D.

    Parameters
    functions : list of (callable, str, dict) or None
        prend [f, label, kwargs]
            f: array de valeurs de param1 ou fonction de param1
            param2 = f(param1) 
            kwargs: ex: {"color": "k"} 
        -> plot la fonction sur la fig 2D
    """
    values1 = result["values1"]
    values2 = result["values2"]
    figs = {}
    if isinstance(method, str):
        method = [method]
    else:
        method = list(method)
        
    for m in method:
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        for ax, key, title in zip(axes,
                                   [f"{m}_overlap1", f"{m}_overlap2"],
                                   ["overlap 1", "overlap 2"]):
            Z = result[key]  # shape (len(values1), len(values2))
            pc = ax.pcolormesh(values2, values1, Z, shading="auto",
                                cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(pc, ax=ax, label=title)
            ax.set_xlabel(vary_param2)
            ax.set_ylabel(vary_param1)
            ax.set_title(f"Method {m} — {title}")
            ax.set_ylim(values1[0],values1[-1])

            if functions is not None:
                for item in functions:
                    f_or_x, y_or_label, *rest = item
                    if callable(f_or_x):
                        label = rest[0] if rest else ""
                        kwargs = rest[1] if len(rest) > 1 else {}
                        x_plot = values1
                        y_plot = f_or_x(values1)
                    else:
                        # f_or_x = x_array (param2), y_or_label = y_array (param1)
                        x_plot = f_or_x
                        y_plot = y_or_label
                        label = rest[0] if rest else ""
                        kwargs = rest[1] if len(rest) > 1 else {}
                    kwargs = dict(kwargs) if kwargs else {}
                    kwargs.setdefault("color", "white")
                    kwargs.setdefault("lw", 2)
                    ax.plot(x_plot, y_plot, label=label, **kwargs)
                ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"Method {m}")
        figs[m] = fig

    return figs


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


### norm contraint methode B
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


# ---------------------------------------------------------------- overlap of x1 with the top eigenvector of A
# used by overlap_x1_vs_n, overlap_x1_vs_rho, overlap_x1_vs_lam1 in plot_maker_tali.py




def overlaps_A_one_draw(n, lam1, lam2, rho, mu, alpha=np.sqrt(0.5)):
    """One draw. Builds A = Y2 - I + 2 mu F1(x) for the three choices of x
         1) the true x_hat (top eigenvector of Y1)
         2) the BBP surrogate m1 x1 + sqrt(1 - m1^2) g_perp
         3) m1 x1
    and returns |<x1, v>| for each of them, v = top eigenvector of A.
    All three use the same Y1, Y2."""
    S = TwoSpikes(n, lam1, lam2, rho, mu=mu, alpha=alpha)
    x1 = S.get_x1()
    m1 = sts.m1_theory(lam1)

    # random unit vector orthogonal to x1
    g = np.random.normal(0, 1, n)
    g_perp = g - (g @ x1) * x1
    g_perp = g_perp / np.linalg.norm(g_perp)

    # the three choices of x
    _, x_hat = S.top_eigenvector(S.Y1)
    x_bbp = m1 * x1 + np.sqrt(1 - m1 ** 2) * g_perp
    x_m1x1 = m1 * x1

    # top eigenvector of A for each x (eigh sorts eigenvalues ascending, so the last column is the top one)
    _, V = np.linalg.eigh(S.A(x_hat, mu))
    overlap_xhat = abs(np.dot(x1, V[:, -1]))

    _, V = np.linalg.eigh(S.A(x_bbp, mu))
    overlap_bbp = abs(np.dot(x1, V[:, -1]))

    _, V = np.linalg.eigh(S.A(x_m1x1, mu))
    overlap_m1x1 = abs(np.dot(x1, V[:, -1]))

    return overlap_xhat, overlap_bbp, overlap_m1x1


def overlap_A_theory(lam1, lam2, rho, mu):
    """Theoretical |<x1, v>|, v = top eigenvector of A, for x = x_hat or the BBP surrogate.
    beta is the same 3x3 matrix as in source_twospike.beta_value, in the basis
    (x1, (x2 - rho x1)/sqrt(1-rho^2), g_perp). With theta, u its top eigenvalue / eigenvector,
    BBP gives |<x1, v>| = sqrt(1 - 1/theta^2) * |u[0]| if theta > 1, else 0."""
    m1 = sts.m1_theory(lam1)
    c = 2 * mu * (lam1 - 1) ** 2
    s_rho = np.sqrt(max(0.0, 1 - rho ** 2))
    s_m = np.sqrt(max(0.0, 1 - m1 ** 2))
    beta = np.array([
        [c * m1 ** 2 + lam2 * rho ** 2, lam2 * rho * s_rho,    c * m1 * s_m],
        [lam2 * rho * s_rho,            lam2 * (1 - rho ** 2), 0.0],
        [c * m1 * s_m,                  0.0,                   c * (1 - m1 ** 2)],
    ])
    w, U = np.linalg.eigh(beta)
    theta = w[-1]
    u = U[:, -1]
    if theta > 1:
        return np.sqrt(1 - 1 / theta ** 2) * abs(u[0])
    else:
        return 0.0
    

def _comp_x_hat_draw(n, lam1, lam2, rho, mu):
    """One draw: top eigenvalue of A = Y2 - I + 2 mu F1(x) for the three choices of x
    (true x_hat, BBP surrogate, m1 x1). All three use the same Y2."""
    S = TwoSpikes(n, lam1, lam2, rho, mu=mu, alpha=np.sqrt(0.5))
    with np.errstate(invalid="ignore"):          # np.where evaluates sqrt(<0) for lam1<1 (result is 0 anyway)
        m1 = float(sts.overlap_Y_P(lam1))    # BBP overlap of top eigvec of Y1 with x1
    x1 = S.get_x1()
    g = np.random.normal(0, 1, n)
    g_perp = g - (g @ x1) * x1                  # project out x1
    g_perp /= np.linalg.norm(g_perp)
    _, xhat = S.top_eigenvector(S.Y1)           # exact top eigvec of Y1 (eigh). NB S.compute_theta1()
                                                # (50 power iterations) is not converged for lam1 < ~1.3
    xs = [xhat,                                          # le vrai x_hat
          m1 * x1 + np.sqrt(1 - m1**2) * g_perp,         # x_hat BBP (proj + g)
          m1 * x1,                                        # m1 x1 (norm m1, not 1)
          x1]
    return xs
