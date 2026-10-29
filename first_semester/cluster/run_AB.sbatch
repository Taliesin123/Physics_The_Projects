#!/bin/bash
#SBATCH --job-name=AB_compare
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80          # <-- # of CPU cores; the script scales
                                    #     with this. Set it to what you allocate.
#SBATCH --mem=200000                   # plenty: matrices are only a few MB each.
                                    #     (800G is unnecessary; raising it will
                                    #     NOT make the run faster.)
#SBATCH --time=4:00:00             # walltime; raise for big N / M / grids.
#SBATCH --output=slurm-%j.out       # stdout/stderr -> submit dir
#SBATCH --partition=normal
#SBATCH --account=infra01
# NOTE: this code is CPU-only. Do NOT request a GPU -- it goes unused and
# only makes you wait longer in the queue. (No --gres=gpu line on purpose.)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# One BLAS thread per process; parallelism comes from the process pool, which
# uses cpus-per-task cores. (The python script also sets these defensively.)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ---- load your python environment: edit for your cluster --------------
# module load python/3.11
# source activate spike_env          # or: conda activate / venv activate
# (numpy, scipy, and -- only if MAKE_PLOTS=True -- matplotlib must be present)

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "CPUs allocated (cpus-per-task): ${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"

# The script auto-reads SLURM_CPUS_PER_TASK for its worker count.
#srun python run_AB_comparison.py
#srun python check_methodB_norm.py
srun python run_loss_sweeps.py

echo "Finished: $(date)"