"""
scatter_overlap_vs_rho.py
==========================

Pour K = 2 taches (spike_lib_N.TwoSpikes avec N_task=2), trace l'overlap
MOYEN SUR LES 2 TACHES en fonction de rho :

    y = mean( overlap[0], overlap[1] )     (= mean( |theta_hat_0 . x_0|,
                                                       |theta_hat_1 . x_1| ))

Pour chaque valeur de rho dans rho_values, on fait M_points resamples
independants (donnees collectees point par point dans run_scatter_overlap_
vs_rho), puis on les agrege par rho : moyenne + barre d'erreur (erreur-type
sur les M_points resamples), trace avec plt.errorbar.

3 courbes (memes couleurs que les autres scripts) :
  - "naive" : estimateur naif unique (pas de CV necessaire)
  - "A"     : chaine Fisher-regularisee, methode A, mu choisi par CV
              (cv_mu_N, RECALCULE A CHAQUE rho)
  - "B"     : idem, methode B

Depend de spike_lib_N.py (doit etre dans le meme dossier) :
    from spike_lib_N import TwoSpikes, cv_mu_N, mu_equilibrium_N

Usage:
    python scatter_overlap_vs_rho.py
"""

import numpy as np
import matplotlib.pyplot as plt

from spike_lib_Ntasks import NSpikes, cv_mu_N, mu_equilibrium_N


def run_scatter_overlap_vs_rho(rho_values, N, lam,
                                n_mu_grid=15, mu_window=0.75,
                                M_cv=10, M_points=30, seed=0):
    """
    Pour chaque rho dans rho_values (K=2 taches fixe) :
      1) CV separee pour A et B (cv_mu_N) -> mu_opt_A(rho), mu_opt_B(rho).
      2) M_points resamples a ce mu -> overlap moyen sur les 2 taches par
         resample (1 point scatter par resample).

    Returns
    -------
    data : dict method -> dict avec
           "rho"     : array (len(rho_values) * M_points,) -- rho repete
           "overlap" : array de meme taille -- overlap moyen par resample
    mu_opt : dict method -> array (len(rho_values),) -- mu retenu par rho
    """
    if seed is not None:
        np.random.seed(seed)

    N_task = 2
    rho_values = np.asarray(rho_values, dtype=float)
    mu_eq = mu_equilibrium_N(lam)
    MU_grid = mu_eq * np.linspace(max(1e-6, 1 - mu_window),
                                   1 + mu_window, n_mu_grid)
    MU_grid = MU_grid[MU_grid > 0]

    methods = ("naive", "A", "B")
    data = {m: {"rho": [], "overlap": []} for m in methods}
    mu_opt = {"A": np.zeros(len(rho_values)),
              "B": np.zeros(len(rho_values))}

    for ridx, rho in enumerate(rho_values):
        # ---- CV (une fois par methode, a ce rho) ------------------------
        print("rho", rho)
        mu_A, _, _ = cv_mu_N(MU_grid, N, N_task, rho, lam, M=M_cv, method="A")
        mu_B, _, _ = cv_mu_N(MU_grid, N, N_task, rho, lam, M=M_cv, method="B")
        mu_opt["A"][ridx] = mu_A
        mu_opt["B"][ridx] = mu_B

        # ---- M_points resamples, un point scatter chacun -----------------
        for _ in range(M_points):
            S = NSpikes(N, N_task, lam, rho, mu=1.0)
            S.compute_naive(1)
            S.muA = mu_A
            S.compute_methodA(1)
            S.muB = mu_B
            S.compute_methodB(1)

            ov = S.overlaps(["naive", "A", "B"])
            for m in methods:
                data[m]["rho"].append(rho)
                data[m]["overlap"].append(ov[m].mean())   # moyenne sur les 2 taches

    for m in methods:
        data[m]["rho"] = np.array(data[m]["rho"])
        data[m]["overlap"] = np.array(data[m]["overlap"])

    return data, mu_opt


def plot_errorbar_overlap_vs_rho(data, rho_values, N, lam, fname=None,
                                  figsize=(7, 5), dpi=150):
    """Meme donnees que plot_scatter_overlap_vs_rho, mais agregees par rho :
    pour chaque rho, on affiche la MOYENNE des points (sur les M_points
    resamples) avec une barre d'erreur = erreur-type (SEM)."""
    style = {
        "naive": dict(color="seagreen", marker="^", linestyle="--",
                      label="Naive"),
        "A": dict(color="steelblue", marker="o", linestyle="-",
                  label="Method A"),
        "B": dict(color="tomato", marker="s", linestyle="-",
                  label="Method B"),
    }

    fig, ax = plt.subplots(figsize=figsize)

    for m in ("naive", "A", "B"):
        rho_arr = data[m]["rho"]
        ov_arr = data[m]["overlap"]
        means, sems = [], []
        for rho in rho_values:
            vals = ov_arr[rho_arr == rho]
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))
        ax.errorbar(rho_values, means, yerr=sems, capsize=3,
                    linewidth=1.8, markersize=6, **style[m])

    ax.set_xlabel(r"Correlation $\rho$", fontsize=12)
    ax.set_ylabel(r"Mean overlap over the 2 tasks", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.suptitle(
        rf"Mean overlap vs $\rho$  ($N={N}$, $K=2$, $\lambda={lam}$)",
        fontsize=12)
    fig.tight_layout()

    if fname:
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    N = 500
    lam = 2.0
    rho_values = np.linspace(0.0, 1.0, 20)

    data, mu_opt = run_scatter_overlap_vs_rho(
        rho_values, N, lam,
        n_mu_grid=20, mu_window=0.75,
        M_cv=10, M_points=5,
        seed=0,
    )

    print("mu_opt A:", np.round(mu_opt["A"], 3))
    print("mu_opt B:", np.round(mu_opt["B"], 3))

    plot_errorbar_overlap_vs_rho(data, rho_values, N, lam,
                                  fname="errorbar_overlap_vs_rho.png")
