import numpy as np
import matplotlib.pyplot as plt
from class_twospike import TwoSpikes, cv_mu, mu_equilibrium
import source_twospike as sts
# Functions used while plotting (simulations, sweeps), so that the plot_maker files stay lean.
# Theory formulas are in source_twospike.py, the class and the method solvers in class_twospike.py.


# ---------------------------------------------------------------- naive estimator (top eigvec of Y)
# used by fig_eig, fig_lambda, heat, fig_phase in plot_maker_tali.py

def make(n, a, b, rho, alpha=np.sqrt(0.5)):
    """TwoSpikes instance parametrised by a = alpha lam1 and b = sqrt(1 - alpha^2) lam2 (the weights in P)."""
    return TwoSpikes(n, a / alpha, b / np.sqrt(1 - alpha**2), rho, alpha=alpha)


def top2(Y):
    w, V = np.linalg.eigh(Y)
    return w[-1:-3:-1], V[:, -1:-3:-1]        # 2 largest, descending


def sample_overlaps(n, a, b, rho, reps, alpha=np.sqrt(0.5)):
    """Mean over reps of (|<x_hat,x1>|, |<x_hat,x2>|), x_hat = top eigenvector of Y = naive_matrix."""
    o = np.zeros(2)
    for _ in range(reps):
        S = make(n, a, b, rho, alpha)
        S.compute_naive()                    # x_hat = top eigenvector of Y = alpha Y1 + sqrt(1-alpha^2) Y2
        o += S.overlaps("naive")
    return o / reps



# ---------------------------------------------------------------- comparison of the methods A, B, naive

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



# ---------------------------------------------------------------- the three choices of x in A = Y2 - I + 2 mu F1(x)
# overlaps_A_one_draw: used by overlap_x1_vs_n, overlap_x1_vs_rho, overlap_x1_vs_lam1 in plot_maker_tali.py
# overlaps_xhat_one_draw: used by overlap_xhat_vs_n, overlap_xhat_vs_rho, overlap_xhat_vs_lam1 in plot_maker_tali.py
# top_eig_A_one_draw:  used by fig_comp_x_hat in plot_maker_tali.py

def overlaps_A_one_draw(n, lam1, lam2, rho, mu, alpha=np.sqrt(0.5)):
    """One draw. Builds A = Y2 - I + 2 mu F1(x) for the four choices of x
         1) the true x_hat (top eigenvector of Y1)
         2) x_pg1  = m1 x1 + sqrt(1 - m1^2) g_perp    (g_perp orthogonal to x1)
         3) x_m1   = m1 x1
         4) x_pg12 = m1 x1 + sqrt(1 - m1^2) g_perp12  (g_perp12 orthogonal to x1 and x2)
    and returns |<x1, v>| for each of them, v = top eigenvector of A.
    All four use the same Y1, Y2."""
    S = TwoSpikes(n, lam1, lam2, rho, mu=mu, alpha=alpha)
    x1 = S.get_x1()
    x2 = S.get_x2()
    m1 = sts.m1_theory(lam1)

    # random unit vector orthogonal to x1
    g = np.random.normal(0, 1, n)
    g_perp = g - (g @ x1) * x1
    g_perp = g_perp / np.linalg.norm(g_perp)

    # random unit vector orthogonal to x1 and x2 (Gram-Schmidt)
    e2 = x2 - (x2 @ x1) * x1                  # part of x2 orthogonal to x1
    g_perp12 = g - (g @ x1) * x1
    if np.linalg.norm(e2) > 1e-8:             # if rho = 1, x2 = x1: nothing more to remove
        e2 = e2 / np.linalg.norm(e2)
        g_perp12 = g_perp12 - (g_perp12 @ e2) * e2
    g_perp12 = g_perp12 / np.linalg.norm(g_perp12)

    # the four choices of x
    _, x_hat = S.top_eigenvector(S.Y1)
    x_bbp = m1 * x1 + np.sqrt(1 - m1 ** 2) * g_perp        # x_pg1
    x_m1x1 = m1 * x1                                       # x_m1
    x_pg12 = m1 * x1 + np.sqrt(1 - m1 ** 2) * g_perp12

    # top eigenvector of A for each x (eigh sorts eigenvalues ascending, so the last column is the top one)
    _, V = np.linalg.eigh(S.A(x_hat, mu))
    overlap_xhat = abs(np.dot(x1, V[:, -1]))

    _, V = np.linalg.eigh(S.A(x_bbp, mu))
    overlap_bbp = abs(np.dot(x1, V[:, -1]))

    _, V = np.linalg.eigh(S.A(x_m1x1, mu))
    overlap_m1x1 = abs(np.dot(x1, V[:, -1]))

    _, V = np.linalg.eigh(S.A(x_pg12, mu))
    overlap_pg12 = abs(np.dot(x1, V[:, -1]))

    return overlap_xhat, overlap_bbp, overlap_m1x1, overlap_pg12

def overlaps_xhat_one_draw(n, lam1, lam2, rho, alpha=np.sqrt(0.5)):
    """One draw. Returns |<x_hat, x_pg1>| and |<x_hat, x_pg12>|, with
         x_hat  = top eigenvector of Y1
         x_pg1  = m1 x1 + sqrt(1 - m1^2) g_perp    (g_perp orthogonal to x1)
         x_pg12 = m1 x1 + sqrt(1 - m1^2) g_perp12  (g_perp12 orthogonal to x1 and x2)"""
    S = TwoSpikes(n, lam1, lam2, rho, alpha=alpha)
    x1 = S.get_x1()
    x2 = S.get_x2()
    m1 = sts.m1_theory(lam1)

    # random unit vector orthogonal to x1
    g = np.random.normal(0, 1, n)
    g_perp = g - (g @ x1) * x1
    g_perp = g_perp / np.linalg.norm(g_perp)

    # random unit vector orthogonal to x1 and x2 (Gram-Schmidt)
    e2 = x2 - (x2 @ x1) * x1                  # part of x2 orthogonal to x1
    g_perp12 = g - (g @ x1) * x1
    if np.linalg.norm(e2) > 1e-8:             # if rho = 1, x2 = x1: nothing more to remove
        e2 = e2 / np.linalg.norm(e2)
        g_perp12 = g_perp12 - (g_perp12 @ e2) * e2
    g_perp12 = g_perp12 / np.linalg.norm(g_perp12)

    _, x_hat = S.top_eigenvector(S.Y1)
    x_pg1 = m1 * x1 + np.sqrt(1 - m1 ** 2) * g_perp
    x_pg12 = m1 * x1 + np.sqrt(1 - m1 ** 2) * g_perp12

    overlap_pg1 = abs(np.dot(x_hat, x_pg1))
    overlap_pg12 = abs(np.dot(x_hat, x_pg12))
    return overlap_pg1, overlap_pg12


def top_eig_A_one_draw(n, lam1, lam2, rho, mu, alpha=np.sqrt(0.5)):
    """One draw. Same three choices of x as overlaps_A_one_draw
         1) the true x_hat (top eigenvector of Y1)
         2) the BBP surrogate m1 x1 + sqrt(1 - m1^2) g_perp
         3) m1 x1
    but returns the top eigenvalue of A = Y2 - I + 2 mu F1(x) for each of them.
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

    # top eigenvalue of A for each x (eigvalsh sorts ascending, so [-1] is the top one)
    top_xhat = np.linalg.eigvalsh(S.A(x_hat, mu))[-1]
    top_bbp = np.linalg.eigvalsh(S.A(x_bbp, mu))[-1]
    top_m1x1 = np.linalg.eigvalsh(S.A(x_m1x1, mu))[-1]

    return top_xhat, top_bbp, top_m1x1


