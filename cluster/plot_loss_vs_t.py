"""
plot_overlap_vs_t.py
=====================

Pour le modele a N_task spikes correles (spike_lib_N.TwoSpikes), trace
l'overlap du DERNIER estimateur theta_hat_t contre TOUTES les cibles deja
vues x_0, ..., x_t :

    y(t) = mean_{s=0}^{t} |theta_hat_t . x_s|

pour t = 0, ..., N_task-1 (cf. spike_lib_N.TwoSpikes.overlaps_cumulative).

C'est differente de l'ancienne quantite |theta_hat_s . x_s| moyennee sur s
(qui reste haute meme a grand t car chaque etape "reussit" sur SA PROPRE
cible, independamment de l'historique) : ici un SEUL vecteur theta_hat_t
doit rester aligne avec un nombre croissant de cibles faiblement correlees
-- on s'attend a une decroissance avec t quand rho est petit, ce qui est le
comportement recherche.

3 courbes :
  - "naive"  : estimateur naif unique (pas de CV necessaire)
  - "A"      : chaine Fisher-regularisee, methode A, mu UNIQUE choisi par CV
               (cv_mu_N), applique a CHAQUE etape de la chaine
  - "B"      : idem, methode B

Depend de spike_lib_N.py (doit etre dans le meme dossier) :
    from spike_lib_N import TwoSpikes, cv_mu_N, mu_equilibrium_N

Usage:
    python plot_loss_vs_t.py
"""

import numpy as np
import matplotlib.pyplot as plt

from spike_lib_Ntasks import NSpikes, cv_mu_chain, mu_equilibrium_N


def run_overlap_vs_t(N, N_task, rho, lam,
                      n_mu_grid=15, mu_window=0.75, M_cv=10,
                      M_final=30, seed=0):
    """
    1) Cross-validation UNIQUE par methode (cv_mu_N) -> mu_opt["A"], mu_opt["B"]
       (un seul mu, applique a CHAQUE etape de la chaine -- pas de CV par etape).
    2) M_final resamples du modele complet ; a chaque resample, on fait
       tourner la chaine A et la chaine B avec ce mu fixe, et a chaque t on
       calcule l'overlap du DERNIER estimateur contre TOUTES les cibles vues
       jusque-la (overlaps_cumulative -- voir spike_lib_N.py).
    3) Moyenne (+ erreur-type) de cette quantite sur les M_final resamples.

    Returns
    -------
    t_values : array (N_task,)
    curves   : dict method -> dict avec "mean" (array N_task) et
               "sem" (array N_task) -- erreur-type sur les M_final resamples
    mu_opt   : dict method -> float, le mu unique retenu par la CV
    """
    rng_seed = seed
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # ---- 1) CV unique par methode (PAS par etape) ----------------------
    mu_eq = mu_equilibrium_N(lam)
    MU_grid = mu_eq * np.linspace(max(1e-6, 1 - mu_window),
                                   1 + mu_window, n_mu_grid)
    MU_grid = MU_grid[MU_grid > 0]

    mu_opt = cv_mu_chain(N, N_task, rho, lam,
                              n_mu_grid=n_mu_grid, mu_window=mu_window,
                              M=M_cv, methods=("A", "B"), seed=None)

    # ---- 2) M_final resamples, overlap CUMULATIF (vs toutes les cibles) ---
    t_values = np.arange(N_task)
    raw_naive = np.zeros((M_final, N_task))
    raw_A = np.zeros((M_final, N_task))
    raw_B = np.zeros((M_final, N_task))

    for r in range(M_final):
        S = NSpikes(N, N_task, lam, rho, mu=1.0)  # mu de construction
        # ecrase a chaque etape -- non utilise directement

        for t in range(1, N_task):
            S.muA = mu_opt["A"]
            S.compute_methodA(t)
            S.muB = mu_opt["B"]
            S.compute_methodB(t)

        ov = S.overlaps_cumulative(["naive", "A", "B"])
        raw_naive[r] = ov["naive"]
        raw_A[r] = ov["A"]
        raw_B[r] = ov["B"]

    curves = {}
    for name, raw in (("naive", raw_naive), ("A", raw_A), ("B", raw_B)):
        curves[name] = {
            "mean": raw.mean(axis=0),
            "sem": raw.std(axis=0) / np.sqrt(M_final),
        }

    return t_values, curves, mu_opt


def plot_overlap_vs_t(t_values, curves, N, N_task, rho, lam, fname=None,
                       figsize=(7, 5), dpi=150):
    style = {
        "naive": dict(color="seagreen", marker="^", linestyle="--",
                      label="Naive"),
        "A": dict(color="steelblue", marker="o", linestyle="-",
                  label="Method A (CV)"),
        "B": dict(color="tomato", marker="s", linestyle="-",
                  label="Method B (CV)"),
    }

    fig, ax = plt.subplots(figsize=figsize)
    for m in ("naive", "A", "B"):
        mean = curves[m]["mean"]
        sem = curves[m]["sem"]
        ax.errorbar(t_values, mean, yerr=sem, capsize=3, linewidth=1.8,
                    markersize=6, **style[m])

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(r"Overlap of $\hat\theta_t$ vs all targets $x_0..x_t$",
                  fontsize=12)
    ax.set_xlim(t_values[0], t_values[-1])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.suptitle(
        rf"Overlap of $\hat\theta_t$ vs $x_0..x_t$  "
        rf"($N={N}$, $K={N_task}$, $\rho={rho}$, $\lambda={lam}$)",
        fontsize=12)
    fig.tight_layout()

    if fname:
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_loss_vs_t(N, N_task, rho, lam,
                   n_mu_grid=15, mu_window=0.75, M_cv=10,
                   M_final=30, seed=0):
    """
    Meme structure que run_overlap_vs_t, mais trace la LOSS cumulee :

        L_t(theta_hat_t) = (1/sqrt(t+1)) * sum_{s=0}^{t} ||Y_s - theta theta^T||_F^2

    (cf. Loss_eval_2(method, time=t) dans spike_lib_N).

    Avantage vs l'overlap : metrique entierement "non supervisee" (pas besoin
    des vrais x_s), la meme pour toutes les methodes, et c'est exactement ce
    que la CV optimise.

    Returns
    -------
    t_values : array (N_task - 1,)   -- commence a t=1 (t=0 trivial)
    curves   : dict method -> dict avec "mean" et "sem" (arrays)
    mu_opt   : dict method -> float
    """
    if seed is not None:
        np.random.seed(seed)

    # ---- 1) CV unique par methode --------------------------------------
    
    mu_opt = cv_mu_chain(N, N_task, rho, lam,
                              n_mu_grid=n_mu_grid, mu_window=mu_window,
                              M=M_cv, methods=("A", "B"), seed=None)
    # ---- 2) M_final resamples, loss par etape --------------------------
    # on commence a t=1 (a t=0 : un seul terme, identique pour tous)
    t_values = np.arange(1, N_task)
    raw = {m: np.zeros((M_final, N_task - 1)) for m in ("naive", "A", "B")}

    for r in range(M_final):
        S = NSpikes(N, N_task, lam, rho, mu=1.0)

        for t in range(1, N_task):
            S.compute_naive(t)
            S.muA = mu_opt["A"][t]
            S.compute_methodA(t)
            S.muB = mu_opt["B"][t]
            S.compute_methodB(t)

            for m in ("naive", "A", "B"):
                raw[m][r, t - 1] = S.Loss_eval_2(m, time=t)

    curves = {}
    for m in ("naive","A", "B"):
        curves[m] = {
            "mean": raw[m].mean(axis=0)-raw["naive"].mean(axis=0),
            "sem": (raw[m]-raw["naive"]).std(axis=0) / np.sqrt(M_final),
        }

    return t_values, curves, mu_opt


def plot_loss_vs_t(t_values, curves, N, N_task, rho, lam, fname=None,
                    figsize=(7, 5), dpi=150):
    style = {
        "naive": dict(color="seagreen", marker="^", linestyle="--",
                       label="Naive"),
        "A": dict(color="steelblue", marker="o", linestyle="-",
                  label="Method A "),
        "B": dict(color="tomato", marker="s", linestyle="-",
                  label="Method B"),
    }

    fig, ax = plt.subplots(figsize=figsize)
    for m in ("naive","A", "B"):
        ax.errorbar(t_values, curves[m]["mean"], yerr=curves[m]["sem"],
                    capsize=3, linewidth=1.8, markersize=6, **style[m])

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(
        r"Difference between score loss eval method and naive: $L_{eval}^t(method) - L_{eval}^t(naive)$",
        fontsize=11)
    ax.set_xlim(t_values[0], t_values[-1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.suptitle(
        rf"Cumulative loss vs $t$  "
        rf"($N={N}$, $K={N_task}$, $\rho={rho}$, $\lambda={lam}$)",
        fontsize=12)
    fig.tight_layout()

    if fname:
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    
    N =1000
    rho = 0.2
    N_task = 8
    lam = np.random.uniform(2, 6, N_task)
    # --- overlap cumulatif ---
    

    # --- loss cumulee ---
    t_loss, curves_loss, _ = run_loss_vs_t(
        N, N_task, rho, lam,
        n_mu_grid=15, mu_window=0.75,
        M_cv=10, M_final=30, seed=0,
    )
    plot_loss_vs_t(t_loss, curves_loss, N, N_task, rho, lam,
                    fname="loss_vs_t.png")

