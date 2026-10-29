"""
plot_2D_lambda1_rho.py
=======================

Heatmap 2D : overlap (Z) en fonction de lambda1 (X) et rho (Y),
pour les methodes A et B, et separement pour le signal x1 et le signal x2.

-> 2 figures produites :
   - Figure 1 : overlap avec x1   (sous-figures : Methode A | Methode B)
   - Figure 2 : overlap avec x2   (sous-figures : Methode A | Methode B)

Reprend la logique / le style de run_AB_comparison.py :
  - calcul parallelise (ProcessPoolExecutor), 1 thread BLAS par worker
  - M resamples Monte-Carlo par point de grille -> on moyenne sur les seeds
  - mu peut etre fixe, ou calcule automatiquement comme mu_opt(lambda1) si
    mu=None (mu_opt = lambda2 / (2*(1-lambda1)**2), cf. sl.mu_equilibrium)

Usage:
    python plot_2D_lambda1_rho.py
ou, depuis un notebook / autre script:
    from plot_2D_lambda1_rho import run_experiment_2D_lambda_rho
    Z1, Z2 = run_experiment_2D_lambda_rho(lambda1_values, rho_values, ...)
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
# ======================================================================
def _simulate_point(task):
    """
    task = (i, j, rho, lambda1, lambda2, n, mu, methods, seed)
    mu=None  -> on utilise mu_opt(lambda1) = lambda2 / (2*(1-lambda1)**2)
    mu=valeur-> on utilise cette valeur fixe pour tous les points

    Retourne (i, j, {method: (overlap1, overlap2)})
    """
    i, j, rho, lambda1, lambda2, n, mu, methods, seed = task
    np.random.seed(seed)

    if mu is None:
        # mu optimal predit, recalcule a chaque lambda1 (cf. mu_equilibrium)
        mu_use = lambda2 / (2 * (1 - lambda1) ** 2)
    else:
        mu_use = mu

    S = sl.TwoSpikes(n, lambda1, lambda2, rho, mu_use)
    S.compute_method(list(methods))
    return i, j, {m: S.overlaps(m) for m in methods}


# ======================================================================
# Experience 2D : lambda1 (x) vs rho (y)
# ======================================================================
def run_experiment_2D_lambda_rho(lambda1_values, rho_values,
                                  n=500, lambda2=2.0, mu=None,
                                  M=20, methods=("A", "B"),
                                  seed=0, n_workers=None, plot=True):
    """
    Balaye lambda1 (axe X) et rho (axe Y), calcule l'overlap moyen
    (sur M resamples) avec x1 et avec x2, pour chaque methode demandee.

    Parameters
    ----------
    lambda1_values : array-like, valeurs de lambda1 a balayer (X)
    rho_values     : array-like, valeurs de rho balayees (Y)
    n              : dimension du signal
    lambda2        : force du spike 2 (fixee)
    mu             : None -> mu_opt(lambda1) recalcule a chaque colonne ;
                     sinon valeur de mu fixee pour toute la grille
    M              : nb de resamples Monte-Carlo par point (rho, lambda1)
    methods        : tuple/list parmi {"A", "B", "naive"}
    seed           : graine de base (SeedSequence) pour reproductibilite
    n_workers      : nb de process workers (defaut: tous les coeurs visibles)
    plot           : si True, affiche les 2 figures (x1 et x2)

    Returns
    -------
    Z1, Z2 : dict method -> array (len(rho_values), len(lambda1_values))
             overlap moyen avec x1 / x2 respectivement
    """
    lambda1_values = np.asarray(lambda1_values, dtype=float)
    rho_values = np.asarray(rho_values, dtype=float)
    methods = tuple(methods)
    n_workers = n_workers or (os.cpu_count() or 1)

    n_rho, n_l1 = len(rho_values), len(lambda1_values)
    Z1 = {m: np.zeros((n_rho, n_l1)) for m in methods}
    Z2 = {m: np.zeros((n_rho, n_l1)) for m in methods}

    # --- construire toutes les taches independantes ----------------------
    n_tasks = n_rho * n_l1 * M
    seeds = np.random.SeedSequence(seed).generate_state(n_tasks)

    tasks = []
    k = 0
    for i, rho in enumerate(rho_values):
        for j, l1 in enumerate(lambda1_values):
            for _ in range(M):
                tasks.append((i, j, rho, float(l1), lambda2, n, mu,
                              methods, int(seeds[k])))
                k += 1

    print(f"Running {n_tasks} simulations on {n_workers} worker(s) "
          f"(n={n}, M={M}, |rho|={n_rho}, |lambda1|={n_l1}) ...", flush=True)

    # store[i][j][m] = {"o1": [...], "o2": [...]}
    store = [[{m: {"o1": [], "o2": []} for m in methods}
              for _ in range(n_l1)] for _ in range(n_rho)]

    chunk = max(1, n_tasks // (n_workers * 8))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, j, ov in ex.map(_simulate_point, tasks, chunksize=chunk):
            for m in methods:
                o1, o2 = ov[m]
                store[i][j][m]["o1"].append(o1)
                store[i][j][m]["o2"].append(o2)

    # --- moyenne sur les M resamples -> grille finale ---------------------
    for i in range(n_rho):
        for j in range(n_l1):
            for m in methods:
                Z1[m][i, j] = np.mean(store[i][j][m]["o1"])
                Z2[m][i, j] = np.mean(store[i][j][m]["o2"])

    if plot:
        _plot_heatmaps(Z1, lambda1_values, rho_values, methods,
                        title=r"Overlap with $x_1$  $|\theta\cdot x_1|$",
                        suptitle_extra=f"n={n}, $\\lambda_2$={lambda2}, M={M}")
        _plot_heatmaps(Z2, lambda1_values, rho_values, methods,
                        title=r"Overlap with $x_2$  $|\theta\cdot x_2|$",
                        suptitle_extra=f"n={n}, $\\lambda_2$={lambda2}, M={M}")

    return Z1, Z2


# ======================================================================
# Plot helper : une figure, un sous-plot par methode
# ======================================================================
def _plot_heatmaps(Z, lambda1_values, rho_values, methods, title="",
                    suptitle_extra=""):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(methods),
                              figsize=(6 * len(methods), 5), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, methods):
        im = ax.imshow(
            Z[m], origin="lower", aspect="auto", cmap="viridis",
            extent=[lambda1_values[0], lambda1_values[-1],
                    rho_values[0], rho_values[-1]],
            vmin=0, vmax=1,
        )
        fig.colorbar(im, ax=ax, label="overlap")
        ax.set_xlabel(r"$\lambda_1$", fontsize=12)
        ax.set_ylabel(r"$\rho$", fontsize=12)
        ax.set_title(f"Method {m}", fontsize=13)

    fig.suptitle(f"{title}   ({suptitle_extra})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ======================================================================
# Exemple d'utilisation
# ======================================================================
if __name__ == "__main__":
    lambda1_values = np.linspace(1.2, 5.0, 25)   # eviter lambda1=1 (mu_opt diverge)
    rho_values = np.linspace(0.0, 1.0, 25)

    Z1, Z2 = run_experiment_2D_lambda_rho(
        lambda1_values, rho_values,
        n=500, lambda2=2.0,
        mu=None,            # None -> mu_opt(lambda1) recalcule a chaque colonne
        M=20,
        methods=("A", "B"),
        seed=0,
        plot=True,
    )
