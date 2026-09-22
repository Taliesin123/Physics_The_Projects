# Physics_The_Projects

Semester project on spiked random matrix models (BBP transition, two correlated
spikes) and Fisher/EWC-based analysis. Continued in semester 2.

Authors: Taliesin Perez, Celia Budelot.

---

## Repository layout

### `Spike_code/`
Core modules and notebooks.

| File | Contents |
|---|---|
| `spiked_matrices.py` | Spiked Wigner / BBP transition: `gaussian_matrix`, `symmetric_gaussian_matrix`, `spiked_gaussian_matrix`, `two_correlated_spikes`, `spec`, `spectral_matrix`, `overlap`, `BBP`, `max_eigen_val`, plotting helpers |
| `fisher_model.py` | Fisher/Hessian analysis: `analyse_Hess_Fisher`, `cv_mu`, `ecart_F_xhat_x1`, `norm_constraint`, `run_experiment_2D` |
| `model_2.py` | Imported by 11 files across the repo — the most depended-on module |

### `Results/`
Lots of results, including plots and data files. Organized by topic.

### `EWC_code/`
EWC (Elastic Weight Consolidation) code for reproduction of the results of the EWC paper everything is based on. 

### `cluster/`
Clone of the project on the ALPS cluster to run huge simulations on GPUs.

### `Plots/`
Some plots.

### `spike_fisher/`
Some code.

### `Theory/`
Theoretical notes and derivations, plots, and other materials.