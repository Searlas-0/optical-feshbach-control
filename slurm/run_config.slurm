#!/bin/bash
#SBATCH --job-name=ofc-config
#SBATCH --output=logs/ofc-config-%j.out
#SBATCH --error=logs/ofc-config-%j.err
#SBATCH --partition=zen5,epyc
#SBATCH --time=4-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch slurm/run_config.slurm CONFIG.yaml" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
mkdir -p logs

CONFIG_PATH="$1"
if [[ "${CONFIG_PATH}" != /* ]]; then
    CONFIG_PATH="${SLURM_SUBMIT_DIR}/${CONFIG_PATH}"
fi
readonly CONFIG_PATH
test -f "${CONFIG_PATH}"

OFC_PYTHON="${OFC_PYTHON:-/home/gyqyan/charlie/.conda/envs/optical-feshbach-control/bin/python}"
readonly OFC_PYTHON
test -x "${OFC_PYTHON}"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

QUEUE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
RUN_ARGUMENTS=(--queue-id "${QUEUE_ID}")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_ARGUMENTS+=(--batch-index "${SLURM_ARRAY_TASK_ID}")
fi
readonly QUEUE_ID

echo "Slurm job ${SLURM_JOB_ID}: ${CONFIG_PATH} ${RUN_ARGUMENTS[*]}"
srun --cpu-bind=cores "${OFC_PYTHON}" run.py \
    "${RUN_ARGUMENTS[@]}" \
    "${CONFIG_PATH}"
