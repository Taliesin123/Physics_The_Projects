"""
plot_overlap_vs_t.py
=====================

Pour le modele a N_task spikes correles (spike_lib_N.TwoSpikes), trace
l'overlap MOYEN CUMULE en fonction de t :

    y(t) = mean( overlap[0], overlap[1], ..., overlap[t] )

pour t = 0, ..., N_task-1, soit la moyenne de l'alignement |theta_hat_s . x_s|
sur toutes les taches s deja vues jusqu'a l'instant t.

3 courbes :
  - "naive"  : estimateur naif unique (pas de CV necessaire)
  - "A"      : chaine Fisher-regularisee, methode A, AVEC CV PAR ETAPE
               (un mu_t different a chaque step t, choisi par cv_mu_chain)
  - "B"      : idem, methode B

Depend de spike_lib_N.py (doit etre dans le meme dossier) :
    from spike_lib_N import TwoSpikes, cv_mu_chain

Usage:
    python cluster/plot_overlap_vs_t.py
"""

import numpy as np
import matplotlib.pyplot as plt

from spike_lib_Ntasks import NSpikes, cv_mu_chain


def run_overlap_vs_t(N, N_task, rho, lam,
                      n_mu_grid=15, mu_window=0.75, M_cv=10,
                      M_final=30, seed=0):
    """
    1) Cross-validation par etape (cv_mu_chain) -> mu_opt["A"][t], mu_opt["B"][t]
    2) M_final resamples du modele complet ; a chaque resample, on fait
       tourner la chaine A et la chaine B en utilisant CES mu_t (un mu
       different par etape), et on calcule l'overlap par etape (naive/A/B).
    3) Moyenne cumulee de l'overlap (axe x = t, axe y = mean(overlap[0..t]))
       moyennee sur les M_final resamples (+ erreur-type).

    Returns
    -------
    t_values : array (N_task,)
    curves   : dict method -> dict avec "mean" (array N_task) et
               "sem" (array N_task) -- erreur-type sur les M_final resamples
    mu_opt   : dict method -> array (N_task,) des mu retenus par la CV
    """
    rng_seed = seed
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # ---- 1) CV par etape, separee pour A et B --------------------------
    mu_opt = cv_mu_chain(N, N_task, rho, lam,
                          n_mu_grid=n_mu_grid, mu_window=mu_window,
                          M=M_cv, methods=("A", "B"), seed=None)

    # ---- 2) M_final resamples, chaines A/B avec mu_t variable ----------
    t_values = np.arange(N_task)
    cum_naive = np.zeros((M_final, N_task))
    cum_A = np.zeros((M_final, N_task))
    cum_B = np.zeros((M_final, N_task))

    for r in range(M_final):
        S = NSpikes(N, N_task, lam, rho, mu=1.0)  # mu de construction
        #  ecrase a chaque etape -- non utilise directement
        ov = S.overlaps(["naive", "A", "B"],time = 0)
        cum_naive[r][0] = np.sum(ov["naive"])
        cum_A[r][0] = np.sum(ov["A"])
        cum_B[r][0] = np.sum(ov["B"])
        for t in range(1, N_task):
             S.compute_naive(t)
             S.muA = mu_opt["A"][t]
             S.compute_methodA(t)
             S.muB = mu_opt["B"][t]
             S.compute_methodB(t)
 
             ov = S.overlaps(["naive", "A", "B"],time = t)
             cum_naive[r][t] = np.sum(ov["naive"])/(t+1)
             cum_A[r][t] = np.sum(ov["A"])/(t+1)
             cum_B[r][t] = np.sum(ov["B"])/(t+1)
    

    curves = {}
    for name, cum in (("naive", cum_naive), ("A", cum_A), ("B", cum_B)):
        curves[name] = {
            "mean": cum.mean(axis=0),
            "sem": cum.std(axis=0) / np.sqrt(M_final),
        }

    return t_values, curves, mu_opt


def plot_overlap_vs_t(t_values, curves, N, N_task, rho, lam, fname=None,
                       figsize=(7, 5), dpi=150):
    style = {
        "naive": dict(color="seagreen", marker="^", linestyle="--",
                      label="Naive"),
        "A": dict(color="steelblue", marker="o", linestyle="-",
                  label="Method A "),
        "B": dict(color="tomato", marker="s", linestyle="-",
                  label="Method B "),
    }

    fig, ax = plt.subplots(figsize=figsize)
    for m in ("naive", "A", "B"):
        mean = curves[m]["mean"]
        sem = curves[m]["sem"]
        ax.errorbar(t_values, mean, yerr=sem, capsize=3, linewidth=1.8,
                    markersize=6, **style[m])

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(r"$m_t$", fontsize=12)
    ax.set_xlim(t_values[0], t_values[-1])
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.suptitle(
        rf"Cumulative mean overlap vs $t$  "
        rf"($N={N}$, $K={N_task}$, $\rho={rho}$, $\lambda={lam}$)",
        fontsize=12)
    fig.tight_layout()

    if fname:
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    N =1000
    rho = 0.1
    N_task = 8
    lam = np.random.uniform(1, 6, N_task)
    
    


    t_values, curves, mu_opt = run_overlap_vs_t(
        N, N_task, rho, lam,
        n_mu_grid=15, mu_window=0.75,
        M_cv=5, M_final=5,
        seed=0,
    )

    print("mu_opt A:", np.round(mu_opt["A"], 3))
    print("mu_opt B:", np.round(mu_opt["B"], 3))

    plot_overlap_vs_t(t_values, curves, N, N_task, rho, lam,
                       fname="overlap_vs_t_sum_lam2.png")
    