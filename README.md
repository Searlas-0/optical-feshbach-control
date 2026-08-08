# Optical Feshbach control

A clean, reproducible JAX implementation of the **dimensionless inelastic**
optical Feshbach optimization. The old `optical-feshbach-control-ver-0`
directory is retained as a physics reference; this repository contains the
current implementation and does not depend on or modify it.

The workflow is deliberately one-way:

```text
create YAML config → run numerical work → query SQLite results → plot selection
```

Data production and plotting are separate processes. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the enforced module-isolation contract.

## Quick start

Create the environment and run the tests:

```bash
conda env create -f environment.yml
conda activate optical-feshbach-control
pytest -q
```

Running the config maker without arguments always uses fresh canonical defaults
and creates a unique timestamped filename:

```bash
python run_config/make_config.py
# The command prints the created path, for example:
python run.py default_20260806T120000000000Z.yaml
```

Bare config names resolve inside `run_config/`. Multiple configs are queued in
the order supplied:

```bash
python run.py first.yaml second.yaml third.yaml
```

The production sweeps have dedicated CPU job files. Submit them from this
repository so relative config, database, and log paths resolve consistently:

```bash
sbatch slurm/u_cap_sweep_cheap.slurm
sbatch slurm/u_cap_sweep_expensive.slurm
```

The runner validates `SLURM_CPUS_PER_TASK` before starting numerical work and
fails clearly if the allocation is smaller than a config's worker count. The
expensive job maps its 50 stable `(N, T)` batches onto a 50-task Slurm array;
each array task writes the same transactional SQLite database under the shared
array job ID. Every supplied Slurm launcher forces CPU-only JAX discovery before
Python starts. The runner also overrides `runtime.device` to `cpu` whenever
`SLURM_JOB_ID` is present, so a config copied from a GPU laptop remains safe on
this CPU-only cluster.

Outside Slurm, `runtime.device` remains a local execution toggle: use `auto` to
let JAX choose, `cpu` to force CPU, or `gpu` to require a configured GPU backend.
Each run stores both the requested `device` and the resolved
`execution_device` in its methodology records.

## Configuration and sweeps

The checked-in production configs are generated from `make_config.py`. The
fixed time scale defines `l* = sqrt(t* hbar / m)`, while the signed, dimensionless
background length `r_bg = a_bg/l*` remains an explicit sweepable physical
parameter. The config omits dimensional interpretation fields (`gamma`,
`Gamma_max`, `a_bg`, `m`, and `detuning_max`) and all elastic fields (`loss`,
`a`, `a_min`, `a_max`, and related tolerances/penalties).

For an experiment, call `make_config()` with only the values that differ from
the defaults; there is no mutable default dictionary to restore afterward:

```python
path = make_config(
    name="u_max_sweep",
    description="Determine a useful intensity cap",
    parameters={
        "N": [100, 200],
        "r_bg": [-1.5, 0.75],
        "u_max": [25.0, 50.0, 100.0],
        "optimizer": "adam",
        "adam_learning_rate": 1e-2,
        "smoothness": 1e-5,
        "sharpness": 1e-7,
    },
    runtime={"concurrent_workers": 4},
    query=None,
)
```

This expands to the full Cartesian product. Parameters with compatible shapes
are flattened with all random initializations and evaluated as one vectorized
JAX batch. `N`, complete schedules, and `t_interval` create separate persisted
batches so a time sweep can be distributed safely over a Slurm array. Batch
planning orders `N`, then schedule, then time, giving stable array indices.

A particular zero-based batch can be run independently while retaining the
same immutable config and batch IDs:

```bash
python run.py --batch-index 7 --queue-id 12345 u_cap_sweep_expensive.yaml
```

A single schedule is a list of stages. A schedule sweep is a list of complete
schedules:

```python
# One schedule
"schedule": [(5_000, 1.0), (5_000, 0.5)]

# Two schedule choices
"schedule": [
    [(5_000, 1.0)],
    [(5_000, 1.0), (5_000, 0.5)],
]
```

For `optimizer: adam`, schedule multipliers compound and Adam moments and its
step counter remain continuous across learning-rate changes. For
`optimizer: lbfgs`, every schedule multiplier must be `1.0`; the L-BFGS line
search chooses the step size instead. Adam settings use the `adam_*` prefix,
while L-BFGS settings use the `lbfgs_*` prefix, so their tuning variables are
kept separate.

### Query-based initializations

An optional top-level `query` selects stored controls from the config's results
database and appends them to the configured random Fourier initializations:

```yaml
query:
  where:
    config_name: earlier_sweep
    r_bg: 0.01
    best_score: "0.1:"
  limit: 5
  order_by: best_score
  descending: true
  control_kind: best
```

`runtime.initialisations` remains the number of new Fourier starts. If this
query matches five runs, those five stored controls are added after the Fourier
starts for every current sweep case. `status: complete` is applied automatically
unless `where` specifies another status. String ranges use the same `MIN:MAX`,
`:MAX`, and `MIN:` notation as `query.py`.

Queried controls are resampled over normalized time when `N` changes and then
mapped through the inverse of the current sigmoid/tanh bounds. Each resulting
run stores `initialization_source: query`, its `source_run_id`, and the selected
`source_control_kind`; Fourier runs store `initialization_source: fourier`.
No-match queries fail before optimization rather than silently falling back to
random starts. Use `query: null` to disable stored-control initializations.

`u_isbound`, `v_isbound`, and `block_size` are intentionally scalar because
they change compiled or diagnostic structure. The maker validates all values
before writing YAML and refuses to overwrite an existing named config,
preserving the file required to reproduce earlier runs. `default_parameters()`
and `default_runtime()` return new mappings on every call.

## Reproducibility and identifiers

`make_config.py` uses cryptographic randomness once to embed:

- one signed 64-bit `config_id` per config;
- one signed 64-bit `batch_id` for every fixed-`N`/schedule batch.

The JAX initialization seed is

```text
(config_id + batch_id) mod (2³² − 1)
```

Initialization index and control name are then folded into that key. Thus every
sweep case inside a batch receives the same corresponding raw starting curves,
different batches receive different curves, and rerunning the same YAML is
identical. A random `queue_id` is generated by each `run.py` invocation, so
configs and batches launched together remain searchable as a group.

All initialization details are stored in `initialization.py` and copied into
each YAML config: ten default starts, five Fourier modes, a `1/m²` envelope, raw-space RMS
0.3, `u` centered at 30% of its cap, and `v` centered at zero. The continuous
coefficients make corresponding curves grid-independent across `N`.

## Physics and optimization

Only two controls remain:

- `u ≥ 0`: dimensionless optical width;
- `v`: signed dimensionless detuning.

The model uses a fixed reference time and its associated positive length,

```text
t* = m l*² / hbar,    l* = sqrt(t* hbar / m),    r_bg = a_bg/l*.
```

Writing `s = sign(r_bg)`, the implemented dimensionless scattering length is:

```text
a_s/l* = r_bg [1 + s u / (-s u - v + i/2)]
        = |r_bg| [s + u / (-s u - v + i/2)]
```

`r_bg` is a finite nonzero signed number (default `1.0`) and is an ordinary
sweepable case parameter, so candidate background lengths and masses can share
a fixed-shape JAX batch. Its sign is derived inside the physics kernel; there is
no separate sign parameter. The same sign multiplies the optical width
`Gamma`, while the zero-width limit remains exactly `a_s/l* = r_bg`. Under the
`+i/2` convention this gives `Im(a_s/l*) <= 0`, and hence
`Im(1/(a_s/l*)) >= 0`, for either background sign. No additional background
sign is applied to the inelastic contact.

This ratio drives the same discretized Volterra recurrence for the pair
amplitude. The molecular objective is the trapezoid-integrated inelastic
contact, and the maximized score is:

```text
score = molecular objective - smoothness penalty - sharpness penalty
```

The optimizer therefore uses the actual molecular-objective and penalty values;
the unpenalized molecular objective is also saved and plotted. The penalty is
evaluated on the absolute bounded controls over dimensionless time:
the first-difference terms scale as `sum(diff(control)**2) / dt`, and the new
second-difference (sharpness) terms scale as
`sum(diff(control, n=2)**2) / dt**3`. `smoothness` and `sharpness` set common
coefficients, with optional `u_smooth`, `v_smooth`, `u_sharp`, and `v_sharp`
overrides.

The implementation uses `jax.vmap`, `jax.value_and_grad`, `lax.scan`, and cached
`jax.jit` stage executables. Learning rates and sweep parameters are dynamic
batch inputs, avoiding recompilation where array shapes stay fixed. Cases with
zero effective `u_sharp` and `v_sharp` are compiled separately from active
sharpness cases, so their compiled objective contains no second-difference
operations.

## Stable split SQLite storage

Persistence is separated by meaning and linked only by the unique `run_id`:

- `results/results.sqlite3` contains physical/calculated outputs only;
- `results/results.parameters.sqlite3` contains exact config documents,
  resolved per-run settings, execution provenance, and optimizer-stage values.

Both databases use generic name/value and named-array records. Adding a config
field, optimizer, calculated scalar, or diagnostic therefore does not require a
column migration or deletion of earlier results. SQLite WAL mode permits
concurrent workers, and compressed NumPy blobs retain complete arrays.

For every initialization, the physical database stores `N`, `T`, `r_bg`,
`u_max`, `v_max`, full score/objective/penalty histories, initial/best/final
bounded controls, tolerance histories, best and final metrics, and the best
step. It also stores the maximum absolute first derivatives of the best `u` and
`v`. Maximum absolute second derivatives are stored only when the corresponding
sharpness coefficient is active.

Before calculation begins, the parameter database allocates the `run_id` and
stores the full original config plus its scalar resolved case, runtime settings,
config/batch/queue identifiers, seed, and initialization index. Thus every
physical record can always be traced to the exact method that produced it even
after the config schema or optimizer changes.

The optimizer does **not** stop every `block_size=500` steps. A recurring device
scan captures only the raw control snapshots needed for tolerance calculations,
without returning to Python. Python regains control only at a learning-rate
boundary, calculates available tolerance rows, commits the stage, and releases
its history arrays. This preserves crash-safe data without the old 500-step
optimization interruptions. Seed sensitivity is deliberately excluded because
it requires a separately chosen population.

A rerun never overwrites an earlier calculation. It allocates fresh `run_id`
values and appends another execution, while the embedded seeds still make the
numerical histories reproducible. If a process fails, its already committed
physical stages and complete methodology record remain available for diagnosis.

## Searching results

Exact values, sets, and inclusive numerical ranges are supported for identifiers,
metrics, or any saved config parameter:

```bash
python query.py --where config_name=example
python query.py --where u_max=25:100 --where best_max_abs_du_dt=:20
python query.py --where config_id=6509282327016992015 --where best_score=0.1:
python query.py --where batch_id=5494828753881022763 --limit 10
```

Use the read-only Python API when building analyses:

```python
from ofc.results import Results

results = Results("results/results.sqlite3")
rows = results.search(
    config_name="example",
    u_max=(25.0, 100.0),
    best_max_abs_du_dt=(None, 20.0),
)
run = results.get(rows[0]["run_id"])
history = run["history"]
best_controls = run["controls"]["best"]
tolerances = run["tolerances"]
```

Numerical range endpoints are inclusive; use `None` for an open endpoint.
Useful indexed fields include `config_name`, `config_file`, `config_id`,
`batch_id`, `queue_id`, `r_bg`, `u_max`, `N`, `best_score`, `best_objective`,
`best_penalty`, `best_step`, `best_max_abs_du_dt`, `best_max_abs_dv_dt`,
`best_max_abs_d2u_dt2`, and `best_max_abs_d2v_dt2`.

## Standard three-figure plotting

Running `plot.py` with no query selects the most recently stored completed
config and writes three separate PNG files. A selection that differs only by
initialization retains the existing convergence, yield-distribution, and
normalized-control view.

When exactly one configuration parameter varies, `plot.py` automatically
switches to parameter-sweep mode:

- Figures 1 and 3 show only the best-score initialization at each sweep value,
  with a consistent `viridis` colour scale.
- Figure 2 uses the swept parameter as its x-axis and scatters every
  initialization, with the best-score result at each value connected above it.
- Geometrically spaced positive sweeps use a logarithmic x-axis. Signed
  geometric sweeps such as positive and negative `r_bg` use a symmetric-log
  axis.
- Unstable best traces remain visible with reduced opacity in Figures 1 and 3;
  stability does not change the Figure 2 scatter styling.

```bash
python plot.py
```

Plotting never runs calculations and the plotting functions never open the
database. The thin adapter can instead retrieve any explicit selection and pass
the complete mappings as arguments:

```bash
python plot.py --where config_name=example --output-dir figures/example
python plot.py --run-id 1 --run-id 2 --output-dir figures/runs_1_2
python plot.py --where config_id=123 --format png --format pdf
python plot.py --where config_name=multi_sweep --sweep-parameter u_max
```

If a selection varies more than one configuration parameter, filter it down to
one sweep or provide `--sweep-parameter NAME`; this avoids silently assigning a
misleading Figure 2 x-axis.

The pure functions `plot_convergence()`, `plot_yield_distribution()`,
`plot_controls()`, and `plot_standard_figures()` remain independently callable
with already-retrieved run mappings. Pass `sweep_parameter="u_max"` to
`plot_standard_figures()` or `save_standard_figures()` when selecting one axis
from multidimensional sweep data.

## Repository layout

```text
run_config/                 immutable YAML configs and the sole config maker
src/ofc/
  config.py                 schema, validation, Cartesian sweep planning
  initialization.py         all random Fourier initialization choices
  physics.py                inelastic dimensionless equations only
  optimization.py           batched JIT Adam stages
  analysis.py               tolerances and best-control derivatives
  storage.py                split transactional physical/methodology writer
  results.py                read-only query/retrieval API
  runner.py                 the only package orchestration layer
  plotting.py               pure plotting from supplied result mappings
run.py                      thin local runner entry point
query.py                    independent read-only search entry point
plot.py                     independent plotting adapter
slurm/run.slurm             reusable queue launcher
tests/                      unit, golden-physics, integration, and boundary tests
```

## Verification

```bash
pytest -q
```

The suite includes independently recalculated signed-background golden values,
Fourier reproducibility/grid invariance, Cartesian sweep/batch ordering, JIT
derivatives, stage-level database commits, cross-database search/range queries,
append-only exact rerun reproduction, and architecture-boundary checks.

The supplied Slurm launchers use portable CPU JAX because the cluster exposes
no GPU resources. On a laptop with a compatible JAX CUDA installation, retain
`device: auto` or set `device: gpu` in the config maker; local runs honor that
setting because they do not have `SLURM_JOB_ID` in their environment.

For a fast test that never opens or appends to SQLite, run
[`sandbox/fast_cheap_test.ipynb`](sandbox/fast_cheap_test.ipynb). It reads the
scalar [`sandbox/cheap_test.yaml`](sandbox/cheap_test.yaml), runs its configured
optimizations entirely in memory, and displays the standard three figures
inside the notebook.
