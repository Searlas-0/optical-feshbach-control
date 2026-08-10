#!/bin/bash
# Run one cap's resolution ladder serially against its isolated database.
# Submission supplies resources and an afterany dependency on N=100 refinement.

set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch slurm/run_resolution_ladder.slurm CAP" >&2
    exit 2
fi

readonly CAP="$1"
case "${CAP}" in
    40|160|1280) ;;
    *) echo "CAP must be 40, 160, or 1280." >&2; exit 2 ;;
esac

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

failures=0
for resolution in 200 300 400 500; do
    loose="run_config/N${resolution}_u${CAP}_resolution_ladder_loose_cpu.yaml"
    strict="run_config/N${resolution}_u${CAP}_resolution_ladder_strict_cpu.yaml"

    echo "u_max=${CAP} N=${resolution}: starting 5 promoted + 10 random loose refinements"
    if srun --cpu-bind=cores "${OFC_PYTHON}" run.py \
        --queue-id "${SLURM_JOB_ID}" "${loose}"; then
        echo "u_max=${CAP} N=${resolution}: loose stage complete"
    else
        echo "u_max=${CAP} N=${resolution}: loose stage failed or lacked five stable predecessors; continuing safely" >&2
        failures=$((failures + 1))
    fi

    echo "u_max=${CAP} N=${resolution}: starting strict refinement of the best stable loose result"
    if srun --cpu-bind=cores "${OFC_PYTHON}" run.py \
        --queue-id "${SLURM_JOB_ID}" "${strict}"; then
        echo "u_max=${CAP} N=${resolution}: strict stage complete"
    else
        echo "u_max=${CAP} N=${resolution}: strict stage failed or lacked a stable loose result; continuing safely" >&2
        failures=$((failures + 1))
    fi
done

echo "u_max=${CAP}: resolution ladder finished with ${failures} failed/skipped stage(s)"
exit "${failures}"
