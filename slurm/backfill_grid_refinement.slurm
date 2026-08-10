#!/bin/bash
#SBATCH --job-name=ofc-grid-backfill
#SBATCH --output=logs/ofc-grid-backfill-%A_%a.out
#SBATCH --error=logs/ofc-grid-backfill-%A_%a.err
#SBATCH --partition=zen5,epyc
# Serialize SQLite writers: concurrent nodes can trigger filesystem-level I/O
# errors against the same large WAL-backed database on shared storage.
#SBATCH --array=0-15%1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -uo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch slurm/backfill_grid_refinement.slurm [DATABASE_MANIFEST]" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
mkdir -p logs

readonly OFC_PYTHON=/home/gyqyan/charlie/.conda/envs/optical-feshbach-control/bin/python
test -x "${OFC_PYTHON}"

MANIFEST="${1:-run_config/grid_refinement_backfill_databases.txt}"
if [[ "${MANIFEST}" != /* ]]; then
    MANIFEST="${SLURM_SUBMIT_DIR}/${MANIFEST}"
fi
readonly MANIFEST
test -f "${MANIFEST}"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

readonly SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array}"
readonly SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:?Missing Slurm array task count}"
failures=0
processed=0

while IFS= read -r database || [[ -n "${database}" ]]; do
    [[ -z "${database}" || "${database}" == \#* ]] && continue
    if [[ "${database}" != /* ]]; then
        database="${SLURM_SUBMIT_DIR}/${database}"
    fi
    if [[ ! -f "${database}" ]]; then
        echo "Missing database ${database}; continuing" >&2
        failures=$((failures + 1))
        continue
    fi
    echo "Backfilling ${database} shard $((SHARD_INDEX + 1))/${SHARD_COUNT}"
    if srun --cpu-bind=cores "${OFC_PYTHON}" -m ofc.grid_refinement_backfill \
        --database "${database}" \
        --shard-index "${SHARD_INDEX}" \
        --shard-count "${SHARD_COUNT}" \
        --batch-size 32 </dev/null; then
        processed=$((processed + 1))
    else
        failures=$((failures + 1))
        echo "Backfill failed for ${database}; continuing" >&2
    fi
done < "${MANIFEST}"

echo "Grid backfill shard complete: databases=${processed}, failures=${failures}"
exit "${failures}"
