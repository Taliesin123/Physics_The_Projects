"""
run_loss_sweeps.py
==================

Driver: pick the loss-optimal regularization strength mu, then compare the
estimators' recovery quality across lambda1 and rho at that best mu.

Two phases
----------
1. LOSS vs MU.  At a fixed reference point (N, lambda1, lambda2, rho), sweep
   mu and evaluate the rank-1 reconstruction loss ``Loss_eval_2`` for Method A
   and Method B, averaged over M resamples. The mu minimizing the mean loss is
   recorded per method (mu*_A, mu*_B) and drawn as a vertical line on the loss
   plot.

2. OVERLAP SWEEPS with mu RE-OPTIMIZED at every point.  Sweep lambda1 (rho
   fixed) and rho (lambda1 fixed). At EACH grid value we first re-run a mu
   sweep and pick the mu that MAXIMIZES the mean overlap
   0.5*(|theta.x1| + |theta.x2|), separately for Method A and Method B (naive
   needs no mu), then measure overlap1 = |theta.x1|, overlap2 = |theta.x2|,
   and their mean at those per-point optima. Each sweep -> one figure with
   exactly 3 panels. The per-point mu*(lambda1) and mu*(rho) are saved to the
   npz.

The work is embarrassingly parallel and fans out over SLURM_CPUS_PER_TASK
cores, exactly like run_AB_comparison.py.

OUTPUT (written inside this script's folder, in ``results/``)
-------------------------------------------------------------
* results/loss_vs_mu.png             -- loss(mu) for A & B, + mu* lines  (1 panel)
* results/overlaps_vs_lambda1.png    -- overlap1/overlap2/mean vs lambda1 (3 panels)
* results/overlaps_vs_rho.png        -- overlap1/overlap2/mean vs rho     (3 panels)
* results/loss_sweeps.npz            -- all numbers as numpy arrays

Run with:
    python run_loss_sweeps.py
"""

import os

# --- Pin BLAS to one thread PER PROCESS, before numpy is imported. ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import spike_lib as sl


# ======================================================================
# CONFIG  -- edit these for the cluster run
# ======================================================================
N = 500                          # signal dimension
LAMBDA2 = 2.0                    # strength of spike 2 (held fixed throughout)
M = 1000                          # resamples per overlap point (final precision)
# Resamples used to ESTIMATE the mean overlap when re-selecting the optimal mu
# at each sweep point. This inner selection dominates the cost
# (|grid| x |MU| x M_SEL sims per sweep), so it can be lowered independently if
# runs get too long; mu selection tolerates more noise than the final overlaps.
# Defaults to M for a faithful, consistent estimate.
M_SEL = M

# Reference point at which the loss-optimal mu is chosen (phase 1) and the
# fixed values used by the sweeps (phase 2).
LAM1_REF = 3.0                   # lambda1 used for the mu sweep & the rho sweep
RHO_REF = 0.2                    # rho used for the mu sweep & the lambda1 sweep

# mu grid for the loss sweep. 0 plus a log span so we see both the small
# optimum (mu_eq = lambda2 / (2(1-lambda1)^2) ~ 0.25 here) and large-mu decay.
#MU = np.unique(np.concatenate([[0.0], np.logspace(-3, 2, 40)]))
MU = np.logspace(-3, 2, 40)   
# parameter grids for the overlap sweeps
L1_VALUES = np.linspace(0.1, 5.0, 50)        # lambda1 sweep (rho = RHO_REF)
RHO_VALUES = np.linspace(0.0, 1.0, 50)      # rho sweep    (lambda1 = LAM1_REF)

METHODS = ["A", "B", "naive"]    # estimators on the overlap plots
SEED = 0                         # base seed; set None for non-reproducible

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

STYLE = {
    "A": dict(fmt="o-", color="steelblue", label="Method A"),
    "B": dict(fmt="s-", color="tomato", label="Method B"),
    "naive": dict(fmt="^--", color="seagreen", label="Naive spectral"),
}


# ======================================================================
# Workers (top-level so they pickle for the process pool)
# ======================================================================
def _sim_loss(task):
    """Phase 1: loss of A and B for one resample at a given mu."""
    n, l1, l2, rho, mu, seed = task
    np.random.seed(seed)
    S = sl.TwoSpikes(n, l1, l2, rho, mu)
    S.compute_method(["A", "B"])
    return {"A": S.Loss_eval_2("A"), "B": S.Loss_eval_2("B")}


def _sim_overlaps(task):
    """Phase 2: overlaps for A (at mu_A), B (at mu_B), naive -- same draw."""
    n, l1, l2, rho, mu_A, mu_B, seed = task
    np.random.seed(seed)
    S = sl.TwoSpikes(n, l1, l2, rho, mu_A)   # Y1, Y2 drawn once
    S.set_mu(mu_A); S.compute_methodA()
    S.set_mu(mu_B); S.compute_methodB()
    S.compute_naive()
    return {m: S.overlaps(m) for m in METHODS}


def _sim_select(task):
    """mu-selection: mean overlap of A and B for one resample at a given mu."""
    n, l1, l2, rho, mu, seed = task
    np.random.seed(seed)
    S = sl.TwoSpikes(n, l1, l2, rho, mu)
    S.compute_method(["A", "B"])
    out = {}
    for m in ("A", "B"):
        o1, o2 = S.overlaps(m)
        out[m] = 0.5 * (o1 + o2)             # mean overlap
    return out


def run_pool(fn, tasks):
    """Map fn over tasks across the process pool, preserving input order."""
    chunk = max(1, len(tasks) // (N_WORKERS * 8))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        return list(ex.map(fn, tasks, chunksize=chunk))


# ======================================================================
# Aggregation helpers
# ======================================================================
def _seeds(n, salt):
    return np.random.SeedSequence([0 if SEED is None else SEED, salt]).generate_state(n)


def best_mu_per_point(vary, values, fixed_l1, fixed_rho, salt):
    """For each grid value, sweep MU and return the mu that MAXIMIZES the mean
    overlap for A and B (mean over M_SEL resamples). One pooled run over all
    points."""
    P = len(values)
    seeds = _seeds(P * len(MU) * M_SEL, salt)
    tasks, k = [], 0
    for v in values:
        l1 = float(v) if vary == "lambda1" else fixed_l1
        rho = float(v) if vary == "rho" else fixed_rho
        for mu in MU:
            for _ in range(M_SEL):
                tasks.append((N, l1, LAMBDA2, rho, float(mu), int(seeds[k])))
                k += 1
    res = run_pool(_sim_select, tasks)

    per = len(MU) * M_SEL
    mu_A = np.zeros(P)
    mu_B = np.zeros(P)
    for p in range(P):
        block = res[p * per:(p + 1) * per]
        ovA = np.array([[block[mi * M_SEL + r]["A"] for r in range(M_SEL)]
                        for mi in range(len(MU))]).mean(1)
        ovB = np.array([[block[mi * M_SEL + r]["B"] for r in range(M_SEL)]
                        for mi in range(len(MU))]).mean(1)
        mu_A[p] = MU[int(np.argmax(ovA))]    # mu maximizing mean overlap
        mu_B[p] = MU[int(np.argmax(ovB))]
    return mu_A, mu_B


def aggregate_overlaps(results, n_points):
    """results is a flat list of {m:(o1,o2)} ordered point-major, M per point."""
    data = {m: {k: [] for k in ("o1", "o1s", "o2", "o2s", "s", "ss")}
            for m in METHODS}
    for p in range(n_points):
        block = results[p * M:(p + 1) * M]
        for m in METHODS:
            o1 = np.array([d[m][0] for d in block])
            o2 = np.array([d[m][1] for d in block])
            s = 0.5 * (o1 + o2)
            data[m]["o1"].append(o1.mean()); data[m]["o1s"].append(o1.std() / np.sqrt(M))
            data[m]["o2"].append(o2.mean()); data[m]["o2s"].append(o2.std() / np.sqrt(M))
            data[m]["s"].append(s.mean());   data[m]["ss"].append(s.std() / np.sqrt(M))
    for m in METHODS:
        for k in data[m]:
            data[m][k] = np.array(data[m][k])
    return data


# ======================================================================
# Plotting
# ======================================================================
def plot_loss(mu_vals, meanA, semA, meanB, semB, mu_A, mu_B, outpath):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(mu_vals, meanA, yerr=semA, capsize=2, ms=4, lw=1.4,
                **STYLE["A"])
    ax.errorbar(mu_vals, meanB, yerr=semB, capsize=2, ms=4, lw=1.4,
                **STYLE["B"])
    ax.axvline(mu_A, ls="--", color=STYLE["A"]["color"],
               label=rf"$\mu^*_A={mu_A:.3g}$")
    ax.axvline(mu_B, ls="--", color=STYLE["B"]["color"],
               label=rf"$\mu^*_B={mu_B:.3g}$")
    ax.set_xscale("symlog", linthresh=1e-3)   # mu grid includes 0
    ax.set_xlabel(r"$\mu$", fontsize=12)
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylabel(r"Loss  $\alpha\|Y_1-\theta\theta^\top\|_F^2"
                  r"+\sqrt{1-\alpha^2}\,\|Y_2-\theta\theta^\top\|_F^2$",
                  fontsize=11)
    ax.set_title("")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_overlaps(xvals, xlabel, data, title, outpath):
    import matplotlib.pyplot as plt
    panels = [
        ("o1", "o1s", r"Overlap with $x_1$  $|\theta\cdot x_1|$"),
        ("o2", "o2s", r"Overlap with $x_2$  $|\theta\cdot x_2|$"),
        ("s",  "ss",  r"Mean overlap  $\frac{1}{2}(|\theta x_1|+|\theta x_2|)$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (mk, sk, ylabel) in zip(axes, panels):
        for m in METHODS:
            ax.errorbar(xvals, data[m][mk], yerr=data[m][sk],
                        capsize=3, lw=1.5, ms=5, **STYLE[m])
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")        # headless; set before pyplot import

    t0 = time.time()

    # ---------- Phase 1: loss vs mu -> best mu per method ----------------
    seeds = _seeds(len(MU) * M, salt=1)
    tasks, k = [], 0
    for mu in MU:
        for _ in range(M):
            tasks.append((N, LAM1_REF, LAMBDA2, RHO_REF, float(mu), int(seeds[k])))
            k += 1
    print(f"[phase 1] loss vs mu: {len(tasks)} sims on {N_WORKERS} worker(s) "
          f"(|MU|={len(MU)}, M={M}) ...", flush=True)
    res = run_pool(_sim_loss, tasks)

    lossA = np.array([[res[p * M + r]["A"] for r in range(M)] for p in range(len(MU))])
    lossB = np.array([[res[p * M + r]["B"] for r in range(M)] for p in range(len(MU))])
    meanA, semA = lossA.mean(1), lossA.std(1) / np.sqrt(M)
    meanB, semB = lossB.mean(1), lossB.std(1) / np.sqrt(M)
    mu_A = float(MU[int(np.argmin(meanA))])
    mu_B = float(MU[int(np.argmin(meanB))])
    print(f"[phase 1] best mu: mu*_A={mu_A:.4g}  mu*_B={mu_B:.4g}", flush=True)

    plot_loss(MU, meanA, semA, meanB, semB, mu_A, mu_B,
              os.path.join(OUTDIR, "loss_vs_mu.png"))
    print(f"    wrote {os.path.join(OUTDIR, 'loss_vs_mu.png')}", flush=True)

    # ---------- Phase 2a: sweep lambda1, mu re-optimized per point ------
    print(f"[phase 2a] sweep lambda1: re-selecting mu (max mean overlap) over "
          f"{len(MU)} values at each of {len(L1_VALUES)} points "
          f"({len(L1_VALUES) * len(MU) * M_SEL} selection sims + "
          f"{len(L1_VALUES) * M} overlap sims) ...", flush=True)
    muA_l1, muB_l1 = best_mu_per_point("lambda1", L1_VALUES, LAM1_REF, RHO_REF, salt=10)
    seeds = _seeds(len(L1_VALUES) * M, salt=11)
    tasks, k = [], 0
    for pi, l1 in enumerate(L1_VALUES):
        for _ in range(M):
            tasks.append((N, float(l1), LAMBDA2, RHO_REF,
                          float(muA_l1[pi]), float(muB_l1[pi]), int(seeds[k])))
            k += 1
    data_l1 = aggregate_overlaps(run_pool(_sim_overlaps, tasks), len(L1_VALUES))
    plot_overlaps(L1_VALUES, r"$\lambda_1$", data_l1,
                  rf"Recovery vs $\lambda_1$  ($N={N}$, $\lambda_2={LAMBDA2}$, "
                  rf"$\rho={RHO_REF}$, $M={M}$)",
                  os.path.join(OUTDIR, "overlaps_vs_lambda1.png"))
    print(f"    wrote {os.path.join(OUTDIR, 'overlaps_vs_lambda1.png')}", flush=True)

    # ---------- Phase 2b: sweep rho, mu re-optimized per point ----------
    print(f"[phase 2b] sweep rho: re-selecting mu (max mean overlap) over "
          f"{len(MU)} values at each of {len(RHO_VALUES)} points "
          f"({len(RHO_VALUES) * len(MU) * M_SEL} selection sims + "
          f"{len(RHO_VALUES) * M} overlap sims) ...", flush=True)
    muA_rho, muB_rho = best_mu_per_point("rho", RHO_VALUES, LAM1_REF, RHO_REF, salt=20)
    seeds = _seeds(len(RHO_VALUES) * M, salt=21)
    tasks, k = [], 0
    for pi, rho in enumerate(RHO_VALUES):
        for _ in range(M):
            tasks.append((N, LAM1_REF, LAMBDA2, float(rho),
                          float(muA_rho[pi]), float(muB_rho[pi]), int(seeds[k])))
            k += 1
    data_rho = aggregate_overlaps(run_pool(_sim_overlaps, tasks), len(RHO_VALUES))
    plot_overlaps(RHO_VALUES, r"$\rho$", data_rho,
                  rf"Recovery vs $\rho$  ($N={N}$, $\lambda_1={LAM1_REF}$, "
                  rf"$\lambda_2={LAMBDA2}$, $M={M}$)",
                  os.path.join(OUTDIR, "overlaps_vs_rho.png"))
    print(f"    wrote {os.path.join(OUTDIR, 'overlaps_vs_rho.png')}", flush=True)

    # ---------- save raw numbers ---------------------------------------
    arrays = {"MU": MU, "lossA_mean": meanA, "lossA_sem": semA,
              "lossB_mean": meanB, "lossB_sem": semB,
              "mu_A_ref": mu_A, "mu_B_ref": mu_B,
              "L1_VALUES": L1_VALUES, "RHO_VALUES": RHO_VALUES,
              "mu_A_of_lambda1": muA_l1, "mu_B_of_lambda1": muB_l1,
              "mu_A_of_rho": muA_rho, "mu_B_of_rho": muB_rho}
    for tag, data in (("l1", data_l1), ("rho", data_rho)):
        for m in METHODS:
            for k_ in ("o1", "o1s", "o2", "o2s", "s", "ss"):
                arrays[f"{tag}__{m}__{k_}"] = data[m][k_]
    npz_path = os.path.join(OUTDIR, "loss_sweeps.npz")
    np.savez_compressed(npz_path, **arrays)
    print(f"Saved numerics -> {npz_path}")
    print(f"Total wall time {time.time() - t0:.1f} s. Output in: {OUTDIR}")


if __name__ == "__main__":
    main()