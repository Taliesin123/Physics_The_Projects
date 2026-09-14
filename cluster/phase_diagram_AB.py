"""
phase_diagram_AB.py
====================

Phase diagram dans le plan (lambda1, lambda2), a rho fixe, pour les
methodes A et B (estimateurs Fisher-regularises de spike_lib.TwoSpikes),
avec mu choisi par cross-validation (sl.cv_mu) autour de
mu_equilibrium(lambda1, lambda2) -- comme dans les heatmaps precedentes.

Logique de vote (reprend exactement ta fonction ``phase_diagram`` de
reference, adaptee a TwoSpikes) :
  - pour chaque point de grille (lambda1, lambda2) et chaque methode,
    on tire N_average resamples independants ;
  - a chaque resample on calcule l'overlap avec x1 et x2 et on "vote"
    overlap > threshold ;
  - le signal i est dit "recupere" au point de grille si plus de la
    moitie des votes le confirment (vote majoritaire, comme dans ton
    code) ;
  - couleur du point, MEME PALETTE que ta figure de reference :
        ni x1 ni x2 recuperes  -> noir   ("Neither recovered")
        seulement x1 recupere  -> rouge  ("Only m1 recovered")
        seulement x2 recupere  -> vert   ("Only m2 recovered")
        x1 ET x2 recuperes     -> jaune  ("Both recovered")

Format de figure identique a celui que tu as montre :
  - meme palette de 4 couleurs + legende dans le meme ordre
  - lignes pointillees optionnelles (verticale noire / horizontale teal)
  - limites des axes = [min(lambda1), max(lambda1)] x [min(lambda2), max(lambda2)]
  - titre "Recovery phase diagram in (lambda1, lambda2), rho = ..."

Une figure PAR METHODE (A, B), sauvegardee dans "phase_diagrams/" :
    phase_diagrams/phase_diagram_A.png
    phase_diagrams/phase_diagram_B.png

ATTENTION COUT DE CALCUL : pour chaque (lambda1, lambda2, methode) on fait
  n_mu_grid * M_cv simulations pour la CV de mu, PLUS N_average simulations
  pour le vote final. Une grille 30x30 x 2 methodes avec les valeurs par
  defaut (n_mu_grid=15, M_cv=5, N_average=20) -> 1800 taches x 95 sims/tache
  = ~171 000 simulations. Reduis la resolution de grille / n_mu_grid / M_cv
  pour un premier essai rapide.

Usage:
    python phase_diagram_AB.py
ou:
    from phase_diagram_AB import run_phase_diagram_AB
    cats = run_phase_diagram_AB(lambda1_values, lambda2_values, rho=0.2, ...)
"""

import os

# --- 1 thread BLAS par worker, AVANT d'importer numpy --------------------
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor

import spike_lib as sl


# ----------------------------------------------------------------------
# Categories (mêmes 4 cas que ta figure de reference)
# ----------------------------------------------------------------------
NEITHER, ONLY_M1, ONLY_M2, BOTH = 0, 1, 2, 3

# Couleurs et libelles dans le MEME ordre que la legende de la figure de
# reference : noir / rouge / vert / jaune.
CATEGORY_COLORS = ["black", "red", "green", "yellow"]
CATEGORY_LABELS = [
    "Neither recovered",
    r"Only $m_1$ recovered",
    r"Only $m_2$ recovered",
    "Both recovered",
]


# ======================================================================
# Worker (top-level pour etre picklable) -- un point de grille, une methode
# ======================================================================
def _vote_point(task):
    """
    task = (i, j, method, lambda1, lambda2, rho, n,
            n_mu_grid, mu_window, M_cv, N_average, threshold, seed)

    1) mu_opt par CV (sl.cv_mu) autour de mu_equilibrium(lambda1, lambda2).
    2) N_average resamples a mu_opt -> vote overlap > threshold pour x1, x2.
    3) Vote majoritaire (> N_average/2) -> categorie (NEITHER/ONLY_M1/
       ONLY_M2/BOTH), exactement comme ta fonction ``phase_diagram``.

    Retourne (i, j, method, category, mu_opt)
    """
    (i, j, method, lambda1, lambda2, rho, n,
     n_mu_grid, mu_window, M_cv, N_average, threshold, seed) = task
    np.random.seed(seed)

    mu_eq = sl.mu_equilibrium(lambda1, lambda2)
    MU_grid = mu_eq * np.linspace(max(1e-6, 1 - mu_window),
                                   1 + mu_window, n_mu_grid)
    MU_grid = MU_grid[MU_grid > 0]

    mu_opt, _scores, _scores_std = sl.cv_mu(
        MU_grid, n, rho, lambda1, lambda2, M=M_cv, method=method
    )

    votes_1, votes_2 = 0, 0
    for _ in range(N_average):
        S = sl.TwoSpikes(n, lambda1, lambda2, rho, mu=mu_opt)
        S.compute_method([method])
        o1, o2 = S.overlaps(method)
        if o1 > threshold:
            votes_1 += 1
        if o2 > threshold:
            votes_2 += 1

    recovered_1 = votes_1 > N_average / 2
    recovered_2 = votes_2 > N_average / 2

    if recovered_1 and recovered_2:
        category = BOTH
    elif not recovered_1 and not recovered_2:
        category = NEITHER
    elif recovered_1:
        category = ONLY_M1
    else:
        category = ONLY_M2

    return i, j, method, category, float(mu_opt)


# ======================================================================
# Run : grille (lambda1, lambda2) a rho fixe, pour chaque methode
# ======================================================================
def run_phase_diagram_AB(lambda1_values, lambda2_values, rho,
                          n=500, threshold=0.1,
                          n_mu_grid=15, mu_window=0.75,
                          M_cv=5, N_average=15,
                          methods=("A", "B"),
                          seed=0, n_workers=None, plot=True,
                          save_dir="phase_diagrams",
                          vline=None, hline=None):
    """
    Balaye lambda1 (X) et lambda2 (Y) a rho fixe. Pour chaque point et
    chaque methode : mu choisi par CV autour de mu_equilibrium, puis vote
    majoritaire sur N_average resamples (overlap > threshold) pour decider
    si x1 / x2 sont "recuperes".

    Parameters
    ----------
    lambda1_values, lambda2_values : grilles 1D pour les axes X, Y
    rho            : correlation entre les deux spikes (fixee)
    n              : dimension du signal
    threshold      : seuil de recouvrement sur l'overlap (defaut 0.5)
    n_mu_grid, mu_window, M_cv : parametres de la CV de mu (cf. script
                     heatmap precedent)
    N_average      : nb de resamples pour le vote majoritaire
    methods        : tuple/list parmi {"A", "B", "naive"}
    seed           : graine de base
    n_workers      : nb de process workers (defaut: tous les coeurs)
    plot           : si True, sauvegarde une figure par methode
    save_dir       : dossier de sortie des figures
    vline, hline   : si fournis (float), trace une ligne verticale /
                     horizontale pointillee a cette valeur (comme dans la
                     figure de reference). None -> pas de ligne.

    Returns
    -------
    categories : dict method -> array (len(lambda2_values), len(lambda1_values))
                 valeurs dans {0,1,2,3} = {NEITHER, ONLY_M1, ONLY_M2, BOTH}
    MU         : dict method -> array de meme forme, mu retenu par la CV
    """
    lambda1_values = np.asarray(lambda1_values, dtype=float)
    lambda2_values = np.asarray(lambda2_values, dtype=float)
    methods = tuple(methods)
    n_workers = n_workers or (os.cpu_count() or 1)

    n_l2, n_l1 = len(lambda2_values), len(lambda1_values)
    categories = {m: np.zeros((n_l2, n_l1), dtype=int) for m in methods}
    MU = {m: np.zeros((n_l2, n_l1)) for m in methods}

    # --- construire toutes les taches independantes ----------------------
    n_tasks = n_l1 * n_l2 * len(methods)
    seeds = np.random.SeedSequence(seed).generate_state(n_tasks)

    tasks = []
    k = 0
    for j, l1 in enumerate(lambda1_values):       # X
        for i, l2 in enumerate(lambda2_values):   # Y
            for m in methods:
                tasks.append((i, j, m, float(l1), float(l2), rho, n,
                              n_mu_grid, mu_window, M_cv, N_average,
                              threshold, int(seeds[k])))
                k += 1

    n_sims_per_task = n_mu_grid * M_cv + N_average
    print(f"Running {n_tasks} grid-tasks on {n_workers} worker(s) "
          f"(~{n_sims_per_task} sims/task, ~{n_tasks * n_sims_per_task} "
          f"sims total; n={n}, rho={rho}, |lambda1|={n_l1}, "
          f"|lambda2|={n_l2}) ...", flush=True)

    chunk = max(1, n_tasks // (n_workers * 8))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, j, m, category, mu_opt in ex.map(
                _vote_point, tasks, chunksize=chunk):
            categories[m][i, j] = category
            MU[m][i, j] = mu_opt

    if plot:
        os.makedirs(save_dir, exist_ok=True)
        for m in methods:
            fname = os.path.join(save_dir, f"phase_diagram_{m}.png")
            _plot_phase_diagram(categories[m], lambda1_values,
                                 lambda2_values, rho, fname,
                                 vline=vline, hline=hline)
            print(f"    wrote {fname}", flush=True)

    return categories, MU


# ======================================================================
# Plot helper -- meme format que la figure de reference
# ======================================================================
def _plot_phase_diagram(category_grid, lambda1_values, lambda2_values, rho,
                         fname, vline=None, hline=None,
                         figsize=(6, 5.5), dpi=300):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    cmap = ListedColormap(CATEGORY_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        category_grid, origin="lower", aspect="auto",
        cmap=cmap, norm=norm,
        extent=[lambda1_values[0], lambda1_values[-1],
                lambda2_values[0], lambda2_values[-1]],
    )

    if vline is not None:
        ax.axvline(vline, color="black", linestyle="--", linewidth=1.5)
    if hline is not None:
        ax.axhline(hline, color="teal", linestyle="--", linewidth=1.5)

    ax.set_xlim(lambda1_values[0], lambda1_values[-1])
    ax.set_ylim(lambda2_values[0], lambda2_values[-1])
    ax.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax.set_ylabel(r"$\lambda_2$", fontsize=13)

    legend_handles = [Patch(facecolor=c, label=l)
                       for c, l in zip(CATEGORY_COLORS, CATEGORY_LABELS)]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=9, framealpha=0.9)

    fig.suptitle(
        rf"Recovery phase diagram in $(\lambda_1, \lambda_2)$, "
        rf"$\rho = {rho}$.", y=0.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ======================================================================
# Exemple d'utilisation
# ======================================================================
if __name__ == "__main__":
    # Grille modeste pour un premier test rapide.
    lambda1_values = np.linspace(0, 5.0, 49)
    lambda2_values = np.linspace(0, 5.0, 49)

    categories, MU = run_phase_diagram_AB(
        lambda1_values, lambda2_values, rho=0.2,
        n=200, threshold=0.5,
        n_mu_grid=10, mu_window=0.75,
        M_cv=5, N_average=20,
        methods=("A", "B"),
        seed=0,
        plot=True,
        vline=1.0,    # adapte ces valeurs au seuil theorique qui t'interesse
        hline=1.0,
    )
