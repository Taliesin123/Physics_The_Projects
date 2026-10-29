# to run : python N_tasks\comparison_ABnaive.py
"""
comparison_ABnaive.py
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
    python comparison_ABnaive.py
"""

import numpy as np
import matplotlib.pyplot as plt

from spike_lib_Ntasks import NSpikes, cv_mu_chain, mu_equilibrium_N

def run_comp_vs_t(N, N_task, rho, lam,
                      n_mu_grid=15, mu_window=0.75, M_cv=10,
                      M_final=30, seed=0,
                      ov=True, Loss=True):
    """
    Fusion de run_overlap_vs_t et run_loss_vs_t.

    1) Cross-validation par etape (cv_mu_chain) -> mu_opt["A"][t], mu_opt["B"][t]
    2) M_final resamples du modele complet ; a chaque resample, on fait
       tourner la chaine A et la chaine B avec ces mu_t, et on calcule,
       selon les flags demandes :
         - ov=True   : l'overlap moyenne cumulee (naive/A/B), axe x = t
         - Loss=True : la loss cumulee (naive/A/B - naive), axe x = t (a partir de t=1)

    Parameters
    ----------
    ov, Loss : bool
        Au moins un des deux doit etre True.

    Returns
    -------
    results : dict
        - results["mu_opt"]            : dict method -> array (N_task,)
        - results["overlap"] (si ov)   : {"t_values": array(N_task,), "curves": {...}}
        - results["loss"]    (si Loss) : {"t_values": array(N_task-1,), "curves": {...}}
    """
    assert ov or Loss, "au moins une des deux metriques (ov ou Loss) doit etre demandee"

    if seed is not None:
        np.random.seed(seed)

    # ---- 1) CV par etape, commune aux deux metriques -------------------
    mu_opt = cv_mu_chain(N, N_task, rho, lam,
                          n_mu_grid=n_mu_grid, mu_window=mu_window,
                          M=M_cv, methods=("A", "B"), seed=None)

    t_values_ov = np.arange(N_task)
    t_values_loss = np.arange(1, N_task)

    if ov:
        cum = {m: np.zeros((M_final, N_task)) for m in ("naive", "A", "B")}
    if Loss:
        raw = {m: np.zeros((M_final, N_task - 1)) for m in ("naive", "A", "B")}

    # ---- 2) M_final resamples, une seule boucle pour les deux ----------
    for r in range(M_final):
        S = NSpikes(N, N_task, lam, rho, mu=1.0)  # mu de construction

        if ov:
            ov0 = S.overlaps(["naive", "A", "B"], time=0)
            for m in ("naive", "A", "B"):
                cum[m][r, 0] = np.sum(ov0[m])

        for t in range(1, N_task):
            S.compute_naive(t)
            S.muA = mu_opt["A"][t]
            S.compute_methodA(t)
            S.muB = mu_opt["B"][t]
            S.compute_methodB(t)

            if ov:
                ov_t = S.overlaps(["naive", "A", "B"], time=t)
                for m in ("naive", "A", "B"):
                    cum[m][r, t] = np.sum(ov_t[m]) / (t + 1)

            if Loss:
                for m in ("naive", "A", "B"):
                    raw[m][r, t - 1] = S.Loss_eval_2(m, time=t)

    # ---- 3) Assemblage des resultats -----------------------------------
    results = {"mu_opt": mu_opt}

    if ov:
        curves_ov = {}
        for m in ("naive", "A", "B"):
            curves_ov[m] = {
                "mean": cum[m].mean(axis=0),
                "sem": cum[m].std(axis=0) / np.sqrt(M_final),
            }
        results["overlap"] = {"t_values": t_values_ov, "curves": curves_ov}

    if Loss:
        curves_loss = {}
        for m in ("naive", "A", "B"):
            curves_loss[m] = {
                "mean": raw[m].mean(axis=0) - raw["naive"].mean(axis=0),
                "sem": (raw[m] - raw["naive"]).std(axis=0) / np.sqrt(M_final),
            }
        results["loss"] = {"t_values": t_values_loss, "curves": curves_loss}

    return results

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
    print (N_task)
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
    ax.set_ylim(0, 1)
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
    
    N =500
    rho = 0.8
    N_task = 8
    lam = np.random.uniform(2, 6, N_task)

    ov = True 
    Loss = True

    res = run_comp_vs_t(N, N_task, rho, lam, ov=ov, Loss=Loss)
    # --- overlap cumulatif ---
    
    t_ov, curves_ov = res["overlap"]["t_values"], res["overlap"]["curves"]
    t_loss, curves_loss = res["loss"]["t_values"], res["loss"]["curves"]
    mu_opt = res["mu_opt"]
    
    if ov:
        print("mu_opt A:", np.round(mu_opt["A"], 3))
        print("mu_opt B:", np.round(mu_opt["B"], 3))
        
        plot_overlap_vs_t(t_ov, curves_ov, N, N_task, rho, lam,
                               f"N_tasks/comparison_results/overlap_rho_{rho}.png")

     # --- loss cumulee ---
    if Loss:
   
        plot_loss_vs_t(t_loss, curves_loss, N, N_task, rho, lam,
                        fname=f"N_tasks/comparison_results/loss_rho_{rho}.png")

