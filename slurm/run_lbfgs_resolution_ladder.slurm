#!/bin/bash
# Run one cap's N=100→500 L-BFGS-B ladder serially.

set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch slurm/run_lbfgs_resolution_ladder.slurm CAP" >&2
    exit 2
fi

readonly CAP="$1"
case "${CAP}" in
    40|160) ;;
    *) echo "CAP must be 40 or 160." >&2; exit 2 ;;
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
handoff="run_config/N100_u${CAP}_lbfgs_handoff_cpu.yaml"
echo "u_max=${CAP} N=100: starting strict L-BFGS-B from five saved checkpoints"
if srun --cpu-bind=cores "${OFC_PYTHON}" run.py \
    --queue-id "${SLURM_JOB_ID}" "${handoff}"; then
    echo "u_max=${CAP} N=100: L-BFGS-B handoff complete"
else
    echo "u_max=${CAP} N=100: L-BFGS-B handoff failed; later stages will validate their own sources" >&2
    failures=$((failures + 1))
fi

for resolution in 200 300 400 500; do
    loose="run_config/N${resolution}_u${CAP}_lbfgs_ladder_loose_cpu.yaml"
    strict="run_config/N${resolution}_u${CAP}_lbfgs_ladder_strict_cpu.yaml"

    echo "u_max=${CAP} N=${resolution}: starting 5 promoted + 10 random loose L-BFGS-B refinements"
    if srun --cpu-bind=cores "${OFC_PYTHON}" run.py \
        --queue-id "${SLURM_JOB_ID}" "${loose}"; then
        echo "u_max=${CAP} N=${resolution}: loose L-BFGS-B stage complete"
    else
        echo "u_max=${CAP} N=${resolution}: loose stage failed or lacked five saved predecessors; continuing safely" >&2
        failures=$((failures + 1))
    fi

    echo "u_max=${CAP} N=${resolution}: starting strict L-BFGS-B refinement"
    if srun --cpu-bind=cores "${OFC_PYTHON}" run.py \
        --queue-id "${SLURM_JOB_ID}" "${strict}"; then
        echo "u_max=${CAP} N=${resolution}: strict L-BFGS-B stage complete"
    else
        echo "u_max=${CAP} N=${resolution}: strict stage failed or lacked a saved loose result; continuing safely" >&2
        failures=$((failures + 1))
    fi
done

echo "u_max=${CAP}: L-BFGS-B resolution ladder finished with ${failures} failed/skipped stage(s)"
exit "${failures}"
