"""
run_AB_comparison.py
====================

Driver script: compare Method A vs Method B at recovering the second spike.

The recovery quality is the overlap of each estimator with the true signals
x1 and x2 (and their average). We sweep the regularization strength ``mu``
for several spike correlations ``rho`` -- this is the core A-vs-B comparison.

PERFORMANCE / CLUSTER NOTES
---------------------------
* The work is embarrassingly parallel: every Monte-Carlo resample and every
  (rho, mu) grid point is independent. This script fans them out across a
  process pool, so it uses ALL allocated CPU cores (set N_WORKERS, or it
  auto-reads SLURM_CPUS_PER_TASK).
* It is CPU-only NumPy. It does NOT use a GPU; for N in the hundreds a GPU
  would not help (matrices are ~MB, transfer overhead dominates). The lever
  that matters is the number of CPU cores.
* Memory footprint is tiny (an N x N float64 matrix is ~N^2 * 8 bytes, e.g.
  ~2 MB at N=500, ~200 MB at N=5000). You do not need hundreds of GB of RAM.
* Each worker process is pinned to a SINGLE BLAS thread (see the env vars set
  at the very top, before numpy is imported) so the many small linear-algebra
  calls don't oversubscribe cores -- parallelism comes from the pool instead.

OUTPUT (always written INSIDE this script's own folder, in ``results/``)
------------------------------------------------------------------------
* ``results/AB_comparison.csv`` -- tidy numerics, one row per (rho, mu,
  method). Pick this up and plot locally with anything.
* ``results/AB_comparison.npz`` -- the same numbers as numpy arrays.
* ``results/AB_vs_mu_rho_*.png`` -- one figure per rho (only if MAKE_PLOTS).

Run with:
    python run_AB_comparison.py
"""

import os

# --- Pin BLAS to one thread PER PROCESS, before numpy is imported. ----
# Parallelism comes from the process pool below, not from threaded BLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import brentq

import spike_lib as sl


# ======================================================================
# CONFIG  -- edit these for the cluster run
# ======================================================================
N = 500                          # signal dimension
LAMBDA1 = 3.0                    # strength of spike 1
LAMBDA2 = 2.0                    # strength of spike 2
M = 200                          # resamples per point (averaging precision)

# Predicted optimal regularization strength for Method B (friend's tip):
#   mu_opt = lambda2 / (2 (1 - lambda1)^2)        (== sl.mu_equilibrium)
#MU_OPT = LAMBDA2 / (2 * (1 - LAMBDA1) ** 2)

# Sweep mu in a window *centred* on MU_OPT so we test the predicted optimum
# directly and see B rise to its best (and overtake A) around it. With 25
# points spanning [0.25, 1.75] * MU_OPT, the exact centre point == MU_OPT.
MU = np.linspace(0, 0.4, 100)
#MU = np.linspace(0.01, 1.0, 10)  # (old) uniform sweep, for reference
RHO = [0.0, 0.1, 0.2, 0.4, 0.5, 0.75, 1.0]   # spike correlations to compare

METHODS = ["A", "B", "naive"]    # estimators under comparison (+naive baseline)

MAKE_PLOTS = True                # False -> data only, no matplotlib needed
SEED = 0                         # base seed; set to None for non-reproducible

# Number of parallel worker processes. Defaults to the cores SLURM gave us
# (SLURM_CPUS_PER_TASK), else all visible cores. Override by hand if needed.
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK",
                               os.cpu_count() or 1))

# Output ALWAYS lands next to this file -> inside the "cluster" folder,
# regardless of the directory you launch the job from.
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CSV_FIELDS = ["rho", "mu", "method",
              "overlap1", "overlap1_std",
              "overlap2", "overlap2_std",
              "sum", "sum_std"]


# ======================================================================
# Worker: one independent simulation (must be top-level to be picklable)
# ======================================================================
def _simulate(task):
    """Run a single resample and return overlaps for each method.

    task = (i, j, rho, mu, N, lambda1, lambda2, methods, seed)
    returns (i, j, {method: (overlap1, overlap2)})
    """
    i, j, rho, mu, n, l1, l2, methods, seed = task
    np.random.seed(seed)
    S = sl.TwoSpikes(n, l1, l2, rho, mu)
    S.compute_method(list(methods))
    return i, j, {m: S.overlaps(m) for m in methods}


# ======================================================================
# Plotting helper (only used when MAKE_PLOTS is True)
# ======================================================================
def plot_one_rho(res, rho, outpath):
    """3-panel figure (overlap x1, overlap x2, mean) comparing A and B."""
    import matplotlib.pyplot as plt
    style = {
        "A": dict(fmt="o-", color="steelblue", label="Method A"),
        "B": dict(fmt="s-", color="tomato", label="Method B"),
        "naive": dict(fmt="^--", color="seagreen", label="Naive spectral"),
    }
    mu_vals = res["values"]
    panels = [
        ("overlap1", r"Overlap with $x_1$  $|\theta\cdot x_1|$"),
        ("overlap2", r"Overlap with $x_2$  $|\theta\cdot x_2|$"),
        ("sum", r"Mean overlap  $\frac{1}{2}(|\theta x_1|+|\theta x_2|)$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, ylabel) in zip(axes, panels):
        for m in METHODS:
            ax.errorbar(
                mu_vals, res[f"{m}_{key}"], yerr=res[f"{m}_{key}_std"],
                capsize=3, linewidth=1.5, markersize=5, **style[m])
        ax.set_xlabel(r"$\mu$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
    fig.suptitle(
        rf"A vs B recovery  ($N={N}$, $\lambda_1={LAMBDA1}$, "
        rf"$\lambda_2={LAMBDA2}$, $\rho={rho}$, $M={M}$)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ======================================================================
# Aggregation + output helpers
# ======================================================================
def _res_for_rho(mu_vals, per_point):
    """Build a result dict (means + standard errors) for one rho.

    per_point[j][m] = {"o1": [...], "o2": [...]} across the M resamples.
    """
    res = {"values": np.asarray(mu_vals, dtype=float)}
    for m in METHODS:
        o1m, o1s, o2m, o2s, sm, ss = [], [], [], [], [], []
        for j in range(len(mu_vals)):
            o1 = np.array(per_point[j][m]["o1"])
            o2 = np.array(per_point[j][m]["o2"])
            s = 0.5 * (o1 + o2)
            o1m.append(o1.mean()); o1s.append(o1.std() / np.sqrt(len(o1)))
            o2m.append(o2.mean()); o2s.append(o2.std() / np.sqrt(len(o2)))
            sm.append(s.mean());   ss.append(s.std() / np.sqrt(len(s)))
        res[f"{m}_overlap1"] = np.array(o1m)
        res[f"{m}_overlap1_std"] = np.array(o1s)
        res[f"{m}_overlap2"] = np.array(o2m)
        res[f"{m}_overlap2_std"] = np.array(o2s)
        res[f"{m}_sum"] = np.array(sm)
        res[f"{m}_sum_std"] = np.array(ss)
    return res


def _rows_from_res(rho, res):
    rows = []
    for j, mu in enumerate(res["values"]):
        for m in METHODS:
            rows.append({
                "rho": rho, "mu": float(mu), "method": m,
                "overlap1": float(res[f"{m}_overlap1"][j]),
                "overlap1_std": float(res[f"{m}_overlap1_std"][j]),
                "overlap2": float(res[f"{m}_overlap2"][j]),
                "overlap2_std": float(res[f"{m}_overlap2_std"][j]),
                "sum": float(res[f"{m}_sum"][j]),
                "sum_std": float(res[f"{m}_sum_std"][j]),
            })
    return rows


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if MAKE_PLOTS:
        import matplotlib
        matplotlib.use("Agg")    # headless: set before pyplot is imported

    # --- build the full list of independent tasks -----------------------
    n_tasks = len(RHO) * len(MU) * M
    seeds = np.random.SeedSequence(SEED).generate_state(n_tasks)
    tasks = []
    k = 0
    for i, rho in enumerate(RHO):
        for j, mu in enumerate(MU):
            for _ in range(M):
                tasks.append((i, j, rho, float(mu), N, LAMBDA1, LAMBDA2,
                              tuple(METHODS), int(seeds[k])))
                k += 1

    print(f"Running {n_tasks} simulations on {N_WORKERS} worker(s) "
          f"(N={N}, M={M}, |RHO|={len(RHO)}, |MU|={len(MU)}) ...", flush=True)

    # storage: store[i][j][m] = {"o1": [...], "o2": [...]}
    store = [[{m: {"o1": [], "o2": []} for m in METHODS}
              for _ in MU] for _ in RHO]

    t0 = time.time()
    chunk = max(1, n_tasks // (N_WORKERS * 8))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, j, ov in ex.map(_simulate, tasks, chunksize=chunk):
            for m in METHODS:
                o1, o2 = ov[m]
                store[i][j][m]["o1"].append(o1)
                store[i][j][m]["o2"].append(o2)
    print(f"Simulations done in {time.time() - t0:.1f} s.", flush=True)

    # --- aggregate, write outputs --------------------------------------
    all_rows, all_arrays = [], {}
    for i, rho in enumerate(RHO):
        res = _res_for_rho(MU, store[i])
        all_rows.extend(_rows_from_res(rho, res))
        for key, val in res.items():
            all_arrays[f"rho_{rho:.2f}__{key}"] = val
        if MAKE_PLOTS:
            fig_path = os.path.join(OUTDIR, f"AB_vs_mu_rho_{rho:.2f}.png")
            plot_one_rho(res, rho, fig_path)
            print(f"    wrote {fig_path}", flush=True)

    csv_path = os.path.join(OUTDIR, "AB_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved numerics -> {csv_path}  ({len(all_rows)} rows)")

    npz_path = os.path.join(OUTDIR, "AB_comparison.npz")
    np.savez_compressed(npz_path, **all_arrays)
    print(f"Saved numerics -> {npz_path}")
    print(f"Total wall time {time.time() - t0:.1f} s. Output in: {OUTDIR}")


if __name__ == "__main__":
    main()