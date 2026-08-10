# RTX 4060 laptop worker

This directory contains the independent half of the underexplored-cap funnel
assigned to the user's 8 GB NVIDIA GeForce RTX 4060 Laptop GPU. It covers both
regularization endpoints for `u_max = 80, 20, 10`. The server continues with
`u_max = 2560, 640, 160`, so the two machines do not duplicate work.

From a clean clone on Linux, run exactly:

```bash
bash scripts/local/run_rtx4060.sh
```

Before that one command, manually download
`results/auto_fourier_intensity_priors-transfer.zip` from the server into the
laptop clone's `results/` directory. The launcher extracts its compact SQLite
database automatically. This is the only input data transfer; it contains 650
scalar solution summaries (not full histories or controls).

For a Windows laptop, run that command inside WSL2. JAX does not support NVIDIA
CUDA on native Windows; WSL2 support is currently experimental. The launcher
checks that the checkout exactly matches `origin/main`, creates its own virtual
environment, installs the same JAX 0.6.2 CUDA 12 version used on the server,
verifies that JAX sees the GPU, and starts or resumes the queue. CUDA 12 was
chosen for broader laptop-driver compatibility.

Every laptop lane starts from fresh, deterministic Fourier controls. Their
intensity center uses the downloaded cross-cap summaries plus results already
completed locally; all dependent stages use the dedicated output pair:

- `results/local_rtx4060_underexplored_v1.sqlite3`
- `results/local_rtx4060_underexplored_v1.parameters.sqlite3`

If the process is interrupted, run the same shell script again. Completed
configs are skipped; an incomplete current config is removed from this
dedicated scratch database and safely retried before any descendant runs.

After all configs complete, the launcher checkpoints both SQLite WAL files and
creates `results/local_rtx4060_underexplored_v1-transfer.zip`. Upload that ZIP
back to the server. It contains both required databases and metadata identifying
the exact Git commit and manifest.

Official JAX platform and installation guidance:
<https://docs.jax.dev/en/latest/installation.html>.
