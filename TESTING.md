# Testing protocol

Run the complete CPU suite before changing physics, optimization, persistence,
config expansion, or process boundaries:

```bash
JAX_PLATFORMS=cpu pytest -q
```

The test layers are:

1. **Configuration/unit tests** validate defaults, input errors, Cartesian
   expansion, deterministic ordering, and embedded ID/seed round trips.
2. **Initialization tests** require bitwise batch-size reproducibility,
   grid-independent continuous Fourier curves, and fixed RMS/mean statistics.
3. **Golden physics tests** lock the signed-background scattering convention,
   inelastic recurrence, objective, and penalty with independently recalculated
   fixed values. Any intended equation change requires a written reason and an
   independently recalculated golden fixture.
4. **JAX tests** compile differentiation paths and require finite values and
   gradients.
5. **Integration tests** use tiny schedules and temporary SQLite databases to
   cover stage commits, histories, controls, tolerances, gradients, exact/range
   search, and deterministic overwrite-on-rerun behavior.
6. **Architecture tests** parse imports to prevent config generation, results,
   and plotting from becoming coupled to the runner or to each other.

Tests must never write to the checked-in `results/` or `figures/` directories.
Use pytest's `tmp_path` for all generated artifacts. Real default-size runs do
not belong in the automated suite; validate those through a named YAML config
after the full CPU suite passes.

CI repeats the same suite from a clean editable install on every push and pull
request. GPU performance is an execution concern rather than a numerical
correctness oracle; CPU x64 golden tests remain the reference.
