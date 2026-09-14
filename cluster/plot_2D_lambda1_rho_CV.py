"""
plot_2D_lambda1_rho.py
=======================

Heatmap 2D : overlap (Z) en fonction de lambda1 (X) et rho (Y),
pour les methodes A et B, et separement pour le signal x1 et le signal x2.

Pour CHAQUE point de grille (rho, lambda1) et CHAQUE methode, mu n'est plus
pris egal a la formule fermee mu_equilibrium -- on fait une VRAIE
cross-validation (sl.cv_mu) sur une grille de mu centree autour de
mu_equilibrium(lambda1, lambda2), et on garde le mu qui minimise la perte
(sl.Loss_eval_2). Le mu retenu est ensuite utilise pour M_final resamples
qui donnent l'overlap moyen reporte dans la heatmap.

-> Pour CHAQUE methode (A, B, ...), une figure avec 2 panneaux cote a cote :
   (a) overlap avec x1   (b) overlap avec x2
   Style "presentation" : pas de titre, lettres (a)/(b) sous chaque panneau,
   colorbar viridis 0->1 avec legende m1 / m2 (comme une figure de papier).
   Les figures sont sauvees dans le dossier "overlap_results/" :
       overlap_results/overlap_A.png
       overlap_results/overlap_B.png

ATTENTION COUT DE CALCUL : pour chaque (rho, lambda1, methode) on fait
  len(MU_grid) * M_cv  simulations pour la CV, PLUS M_final simulations
  finales. Avec les valeurs par defaut ci-dessous (n_mu_grid=15, M_cv=5,
  M_final=20) ca fait 95 simulations par tache. Une grille 25x25x2 methodes
  = 1250 taches -> ~119 000 simulations au total. Reduis la resolution de
  la grille et/ou n_mu_grid/M_cv pour un premier test rapide.

Reprend le style de run_AB_comparison.py (process pool, 1 thread BLAS par
worker).

Usage:
    python plot_2D_lambda1_rho.py
ou, depuis un notebook / autre script:
    from plot_2D_lambda1_rho import run_experiment_2D_lambda_rho
    Z1, Z2, MU = run_experiment_2D_lambda_rho(lambda1_values, rho_values, ...)
"""

import os

# --- 1 thread BLAS par worker, AVANT d'importer numpy --------------------
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor

import spike_lib as sl


# ======================================================================
# Worker (doit etre top-level pour etre picklable par ProcessPoolExecutor)
# Une tache = UN point de grille (rho, lambda1) pour UNE methode.
# ======================================================================
def _simulate_method_point(task):
    """
    task = (i, j, method, rho, lambda1, lambda2, n,
            n_mu_grid, mu_window, M_cv, M_final, seed)

    1) Construit une grille de mu centree sur mu_equilibrium(lambda1, lambda2):
           MU_grid = mu_eq * linspace(1 - mu_window, 1 + mu_window, n_mu_grid)
       (valeurs <= 0 retirees -- mu_eq est toujours > 0 ici).
    2) Cherche mu_opt par cross-validation (sl.cv_mu), M_cv resamples / valeur.
    3) Lance M_final resamples a mu_opt et mesure les overlaps avec x1, x2.

    Retourne (i, j, method, mu_opt, o1_list, o2_list)
    """
    (i, j, method, rho, lambda1, lambda2, n,
     n_mu_grid, mu_window, M_cv, M_final, seed) = task
    np.random.seed(seed)

    mu_eq = sl.mu_equilibrium(lambda1, lambda2)
    MU_grid = mu_eq * np.linspace(max(1e-6, 1 - mu_window),
                                   1 + mu_window, n_mu_grid)
    MU_grid = MU_grid[MU_grid > 0]

    mu_opt, _scores, _scores_std = sl.cv_mu(
        MU_grid, n, rho, lambda1, lambda2, M=M_cv, method=method
    )

    o1_list, o2_list = [], []
    for _ in range(M_final):
        S = sl.TwoSpikes(n, lambda1, lambda2, rho, mu=mu_opt)
        S.compute_method([method])
        o1, o2 = S.overlaps(method)
        o1_list.append(o1)
        o2_list.append(o2)

    return i, j, method, float(mu_opt), o1_list, o2_list


# ======================================================================
# Experience 2D : lambda1 (x) vs rho (y), mu choisi par CV a chaque point
# ======================================================================
def run_experiment_2D_lambda_rho(lambda1_values, rho_values,
                                  n=200, lambda2=2.0,
                                  n_mu_grid=15, mu_window=0.75,
                                  M_cv=5, M_final=20,
                                  methods=("A", "B"),
                                  seed=0, n_workers=None, plot=True,
                                  save_dir="overlap_results"):
    """
    Balaye lambda1 (axe X) et rho (axe Y). Pour chaque point et chaque
    methode, choisit mu par cross-validation (sl.cv_mu) sur une grille
    centree autour de mu_equilibrium(lambda1, lambda2), puis mesure
    l'overlap moyen (sur M_final resamples) avec x1 et avec x2.

    Parameters
    ----------
    lambda1_values : array-like, valeurs de lambda1 a balayer (X)
    rho_values     : array-like, valeurs de rho balayees (Y)
    n              : dimension du signal
    lambda2        : force du spike 2 (fixee)
    n_mu_grid      : nb de valeurs de mu testees lors de la CV
    mu_window      : fenetre relative autour de mu_eq, ex. 0.75 ->
                     MU_grid = mu_eq * linspace(0.25, 1.75, n_mu_grid)
    M_cv           : nb de resamples Monte-Carlo par valeur de mu pendant
                     la CV (cf. sl.cv_mu)
    M_final        : nb de resamples Monte-Carlo, a mu_opt, pour mesurer
                     l'overlap final reporte dans la heatmap
    methods        : tuple/list parmi {"A", "B", "naive"}
    seed           : graine de base (SeedSequence) pour reproductibilite
    n_workers      : nb de process workers (defaut: tous les coeurs visibles)
    plot           : si True, affiche ET sauvegarde les figures (une figure
                     par methode, 2 panneaux : overlap x1 | overlap x2)
    save_dir       : dossier (cree si besoin) ou sont sauvees les figures

    Returns
    -------
    Z1, Z2 : dict method -> array (len(rho_values), len(lambda1_values))
             overlap moyen avec x1 / x2 respectivement
    MU     : dict method -> array (len(rho_values), len(lambda1_values))
             mu retenu par la CV a chaque point de grille
    """
    lambda1_values = np.asarray(lambda1_values, dtype=float)
    rho_values = np.asarray(rho_values, dtype=float)
    methods = tuple(methods)
    n_workers = n_workers or (os.cpu_count() or 1)

    n_rho, n_l1 = len(rho_values), len(lambda1_values)
    Z1 = {m: np.zeros((n_rho, n_l1)) for m in methods}
    Z2 = {m: np.zeros((n_rho, n_l1)) for m in methods}
    MU = {m: np.zeros((n_rho, n_l1)) for m in methods}

    # --- construire toutes les taches independantes ----------------------
    # Une tache = un point de grille (rho, lambda1) pour une methode.
    n_tasks = n_rho * n_l1 * len(methods)
    seeds = np.random.SeedSequence(seed).generate_state(n_tasks)

    tasks = []
    k = 0
    for i, rho in enumerate(rho_values):
        for j, l1 in enumerate(lambda1_values):
            for m in methods:
                tasks.append((i, j, m, rho, float(l1), lambda2, n,
                              n_mu_grid, mu_window, M_cv, M_final,
                              int(seeds[k])))
                k += 1

    n_sims_per_task = n_mu_grid * M_cv + M_final
    print(f"Running {n_tasks} grid-tasks on {n_workers} worker(s) "
          f"(~{n_sims_per_task} sims/task, ~{n_tasks * n_sims_per_task} "
          f"sims total; n={n}, |rho|={n_rho}, |lambda1|={n_l1}) ...",
          flush=True)

    chunk = max(1, n_tasks // (n_workers * 8))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, j, m, mu_opt, o1_list, o2_list in ex.map(
                _simulate_method_point, tasks, chunksize=chunk):
            Z1[m][i, j] = np.mean(o1_list)
            Z2[m][i, j] = np.mean(o2_list)
            MU[m][i, j] = mu_opt

    if plot:
        os.makedirs(save_dir, exist_ok=True)
        for m in methods:
            fname = os.path.join(save_dir, f"overlap_{m}.png")
            _plot_method_panel(Z1[m], Z2[m], lambda1_values, rho_values,
                                fname)
            print(f"    wrote {fname}", flush=True)

    return Z1, Z2, MU


# ======================================================================
# Plot helper : 1 figure par methode, 2 panneaux (overlap x1 | overlap x2)
# Style "papier" -- pas de titre, lettres (a)/(b), colorbar 0->1 viridis.
# ======================================================================
def _plot_method_panel(Z1_m, Z2_m, lambda1_values, rho_values, fname,
                        figsize=(10, 4.2), dpi=300):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    extent = [lambda1_values[0], lambda1_values[-1],
              rho_values[0], rho_values[-1]]

    panels = [
        (axes[0], Z1_m, r"Overlap $m_1 = |\hat{x}\cdot x_1|$"," "),
        (axes[1], Z2_m, r"Overlap $m_2 = |\hat{x}\cdot x_2|$", " "),
    ]

    for ax, Z, cbar_label, letter in panels:
        im = ax.imshow(
            Z, origin="lower", aspect="auto", cmap="viridis",
            extent=extent, vmin=0, vmax=1,
        )
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel(r"$\lambda_1$", fontsize=12)
        ax.set_ylabel(r"Correlation $\rho$", fontsize=12)
        # lettre (a)/(b) sous le panneau, comme dans la figure de reference
        ax.text(0.5, -0.18, letter, transform=ax.transAxes,
                 ha="center", va="top", fontsize=12)

    fig.tight_layout()
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ======================================================================
# Exemple d'utilisation
# ======================================================================
if __name__ == "__main__":
    # Grille modeste pour un premier test rapide -- augmente la resolution
    # une fois que tu as verifie que tout tourne correctement.
    lambda1_values = np.linspace(0, 5.1, 20)   # eviter lambda1=1 (mu_eq diverge)
    rho_values = np.linspace(0.0, 1.0, 20)

    run_experiment_2D_lambda_rho(
        lambda1_values, rho_values,
        n=200, lambda2=3.0,
        n_mu_grid=10, mu_window=0.75,   # MU_grid = mu_eq * [0.25, 1.75]
        M_cv=10, M_final=15,
        methods=("B"),
        seed=0,
        plot=True,
    )
