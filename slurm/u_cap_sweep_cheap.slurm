#!/bin/bash
#SBATCH --job-name=ofc-u-cheap
#SBATCH --output=logs/u-cap-cheap-%j.out
#SBATCH --error=logs/u-cap-cheap-%j.err
#SBATCH --partition=zen5,epyc
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
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
    u_cap_sweep_cheap.yaml
