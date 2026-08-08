"""Process-level device policy that must run before JAX is imported."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import os


def is_slurm(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the current process belongs to a Slurm allocation."""

    environment = os.environ if environ is None else environ
    return bool(environment.get("SLURM_JOB_ID"))


def configure_jax_environment(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Force CPU-only JAX discovery for Slurm while leaving local runs alone."""

    environment = os.environ if environ is None else environ
    if not is_slurm(environment):
        return
    environment["JAX_PLATFORMS"] = "cpu"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"


def effective_device(
    configured_device: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the config toggle under the process-level Slurm CPU policy."""

    return "cpu" if is_slurm(environ) else configured_device
