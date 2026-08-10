#!/usr/bin/env bash
# One-command RTX 4060 worker setup and resumable local experiment launcher.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly VENV_DIR="${SCRIPT_DIR}/.venv"
readonly PYTHON_BIN="${VENV_DIR}/bin/python"
readonly SETUP_MARKER="${VENV_DIR}/.ofc-setup-commit"
readonly RUN_LOG="${PROJECT_ROOT}/logs/local_rtx4060.log"
readonly JAX_VERSION="0.6.2"

cd "${PROJECT_ROOT}"
mkdir -p logs results

case "$(uname -s)" in
    Linux*) ;;
    *)
        echo "This GPU launcher requires Linux. On Windows, run it inside WSL2; native Windows JAX does not support NVIDIA CUDA." >&2
        exit 2
        ;;
esac

command -v git >/dev/null || { echo "git is required." >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 >= 3.10 is required." >&2; exit 2; }
command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable. Install/update the NVIDIA driver and enable GPU access in WSL2." >&2
    exit 2
}

if [[ -n "$(git status --porcelain)" ]]; then
    echo "The checkout has uncommitted files. Commit, discard, or move them before starting the production queue." >&2
    git status --short >&2
    exit 2
fi

git fetch origin main
readonly LOCAL_COMMIT="$(git rev-parse HEAD)"
readonly REMOTE_COMMIT="$(git rev-parse origin/main)"
if [[ "${LOCAL_COMMIT}" != "${REMOTE_COMMIT}" ]]; then
    if git merge-base --is-ancestor "${LOCAL_COMMIT}" "${REMOTE_COMMIT}"; then
        git merge --ff-only "${REMOTE_COMMIT}"
        exec "${BASH_SOURCE[0]}" "$@"
    fi
    echo "The checkout is ahead of or diverged from origin/main. Resolve it before running production work." >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

readonly SETUP_KEY="${LOCAL_COMMIT}:jax-${JAX_VERSION}-cuda12"
if [[ ! -f "${SETUP_MARKER}" || "$(<"${SETUP_MARKER}")" != "${SETUP_KEY}" ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install --upgrade "jax[cuda12]==${JAX_VERSION}"
    "${PYTHON_BIN}" -m pip install --editable "${PROJECT_ROOT}"
    echo "${SETUP_KEY}" > "${SETUP_MARKER}"
fi

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" -c '
import jax
gpus = [device for device in jax.devices() if device.platform == "gpu"]
if not gpus:
    raise SystemExit("JAX did not discover a CUDA GPU. Use WSL2/Linux and check the NVIDIA driver.")
print("JAX", jax.__version__, "using", gpus[0])
'

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_local_queue.py" 2>&1 | tee -a "${RUN_LOG}"

