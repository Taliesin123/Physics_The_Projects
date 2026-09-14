# Physics_The_Projects

Semester project on spiked random matrix models (BBP transition, two correlated
spikes) and Fisher/EWC-based analysis. Continued in semester 2.

Authors: Taliesin Perez, Celia Budelot.

---

## Repository layout

### `Spike_code/`
Core modules and notebooks.

| File | Author | Contents |
|---|---|---|
| `spiked_matrices.py` | Tali | Spiked Wigner / BBP transition: `gaussian_matrix`, `symmetric_gaussian_matrix`, `spiked_gaussian_matrix`, `two_correlated_spikes`, `spec`, `spectral_matrix`, `overlap`, `BBP`, `max_eigen_val`, plotting helpers |
| `fisher_model.py` | Celia | Fisher/Hessian analysis: `analyse_Hess_Fisher`, `cv_mu`, `ecart_F_xhat_x1`, `norm_constraint`, `run_experiment_2D` |
| `model_2.py` | Tali | Imported by 11 files across the repo — the most depended-on module |

Notebooks: `Overlap_2spike.ipynb`, `Plots_1.ipynb`, `naive_vs_fisher.ipynb`,
`nums_fisher_yours.ipynb`, `nums_fisher_celia.ipynb`.

Scripts: `phase_diagram.py`, `phase_transi_and_bulk_video[_2].py`,
`plot_corrolated_spikes[_2].py`, `plots_2.py`.

### `spike_fisher/`
Celia's Fisher / EWC regularization work: `fisher_regularization.py`,
`ewc_plots.py`, `l2_comparison[_plots].py`, `phase_diagram_ewc[_plots].py`,
`rho_sweep[_plots].py`, `two_spike_recovery.py`, plus
`Overlap_2spike.ipynb` and `Overlap_2spike_yours.ipynb`.

Contains a copy of `spiked_matrices.py` so its notebooks can import it.
**This is a duplicate, not a second version** — keep in sync or replace with a
path import.

### `cluster/`
SLURM jobs and EWC experiments run on the cluster: `run_AB.sbatch`,
`run_EWC.sbatch`, `run_AB_comparison.py`, `run_loss_sweeps.py`, `plot_ewc.py`,
`EWC_plots.py`, `spike_lib.py`, `check_methodB_norm.py`,
`test_solve_norm_constrained.py`.

Also contains `venv-ewc/` (a Linux aarch64 virtualenv), the raw MNIST dataset,
and SLURM logs. These are committed but should not be — the venv only runs on
the cluster and can be rebuilt with pip.

### `Plots/`
Figures and videos from semester 1.

---

## Module naming — read this before you import anything

There were originally two different files both called `model.py`, one on each
branch, with **zero function names in common**. The merge would have silently
replaced one with the other, so they were split into `spiked_matrices.py` and
`fisher_model.py`. `model.py` no longer exists; nothing should import it.

All imports were repointed but keep the `md` alias, so call sites are unchanged:

```python
import spiked_matrices as md   # was: import model as md
import fisher_model as md      # was: import model as md
```

`model_2.py` was never affected and is imported as `import model_2 as m`.

---

## Open questions for semester 2

1. **`cv_mu` and `norm_constraint` exist in both `Spike_code/fisher_model.py`
   and `cluster/spike_lib.py`.** Unknown whether the implementations are
   identical or have diverged. If they differ, two notebooks are producing
   different numbers for the same quantity. Check this first.
2. **`Spike_code/` and `spike_fisher/` both contain `Overlap_2spike.ipynb`,**
   and there are two versions of `nums_fisher`. Which directory was the live
   workspace is not recorded.
3. **`spike_lib_Ntasks`** is referenced from memory but exists nowhere in the
   repo or on Tali's machine. May be on Celia's clone or the cluster.
4. Which `cluster/` scripts are live vs. one-off runs has not been determined.
5. No `requirements.txt`. Compiled caches show the code was run under Python
   3.10, 3.11, 3.12 and 3.13 — pin one version.

---

## Git history

| Ref | What it is |
|---|---|
| `main` | Both branches merged, nothing deleted |
| `celia` | Celia's branch as it stood at end of semester 1 |
| `semester1-main` | Tag: `main` before the merge |
| `semester1-celia` | Tag: `celia` before the merge |

The tags are the recovery point. Anything removed since then is still
reachable through them.

Note: `main` was force-moved forward during cleanup. Run `git fetch` before
working from an older clone.

---

## Known issues

- `spiked_matrices.py:101` — `f'$\lambda$'` raises a `SyntaxWarning`; should be
  a raw string, `rf'$\lambda$'`.
- `.DS_Store` files are tracked and cause spurious merge conflicts.