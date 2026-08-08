#!/bin/bash
#SBATCH --job-name=ofc-88sr-n100-u
#SBATCH --output=logs/88-Sr-N100-u-max-%j.out
#SBATCH --error=logs/88-Sr-N100-u-max-%j.err
#SBATCH --partition=zen5,epyc
#SBATCH --time=4-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
mkdir -p logs

readonly OFC_PYTHON=/home/gyqyan/charlie/.conda/envs/optical-feshbach-control/bin/python
test -x "${OFC_PYTHON}"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun --cpu-bind=cores "${OFC_PYTHON}" run.py \
    --queue-id "${SLURM_JOB_ID}" \
    88_Sr_N_100_u_max_sweep.yaml
