from pathlib import Path

from ofc.device import configure_jax_environment, effective_device, is_slurm
from ofc.runner import _device


ROOT = Path(__file__).resolve().parents[1]


def test_slurm_overrides_gpu_config_and_cuda_environment():
    environment = {
        "SLURM_JOB_ID": "123",
        "JAX_PLATFORMS": "cuda",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    configure_jax_environment(environment)

    assert is_slurm(environment)
    assert effective_device("gpu", environment) == "cpu"
    assert environment["JAX_PLATFORMS"] == "cpu"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] == "1"


def test_local_device_toggle_is_unchanged():
    environment = {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0"}

    configure_jax_environment(environment)

    assert not is_slurm(environment)
    assert effective_device("auto", environment) == "auto"
    assert effective_device("cpu", environment) == "cpu"
    assert effective_device("gpu", environment) == "gpu"
    assert environment == {
        "JAX_PLATFORMS": "cuda",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def test_runner_selects_cpu_for_gpu_config_inside_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    assert _device("gpu").platform == "cpu"


def test_every_slurm_launcher_disables_cuda_discovery():
    scripts = sorted((ROOT / "slurm").glob("*.slurm"))
    assert scripts

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "export JAX_PLATFORMS=cpu" in text, script
        assert 'export CUDA_VISIBLE_DEVICES=""' in text, script
        assert "export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1" in text, script
        assert "#SBATCH --gres" not in text, script
        assert "#SBATCH --gpus" not in text, script


def test_generic_slurm_launcher_accepts_and_runs_one_config_path():
    text = (ROOT / "slurm" / "run_config.slurm").read_text(encoding="utf-8")

    assert 'CONFIG_PATH="$1"' in text
    assert 'QUEUE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"' in text
    assert 'RUN_ARGUMENTS=(--queue-id "${QUEUE_ID}")' in text
    assert '--batch-index "${SLURM_ARRAY_TASK_ID}"' in text
    assert '"${CONFIG_PATH}"' in text
