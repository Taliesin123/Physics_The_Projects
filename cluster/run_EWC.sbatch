#!/bin/bash
#SBATCH --job-name=EWC_comparison
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # this is a PyTorch/CUDA job -> request a GPU
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00              # safety margin; raise for big grids
#SBATCH --partition=debug           # 'debug' is capped short; use it only for quick tests
#SBATCH --output=logs/slurm-%x-%j.out
#
# ---- CSCS PyTorch environment (uenv) ---------------------------------------
# The pytorch uenv already provides torch, torchvision and numpy.
# matplotlib is added in a venv on top (see SETUP_cluster.md, one-time step).
#SBATCH --uenv=pytorch/v2.8.0:/user-environment
#SBATCH --view=default

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Recommended CSCS PyTorch env vars
export OMP_NUM_THREADS=8
export CUDA_CACHE_DISABLE=1
export MPICH_GPU_SUPPORT_ENABLED=0

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"

# Activate the matplotlib venv INSIDE srun (it lives on top of the uenv).
srun -ul bash -c "
    source \"${SLURM_SUBMIT_DIR}/venv-ewc/bin/activate\"
    python EWC_plots.py
"

echo "Finished: $(date)"