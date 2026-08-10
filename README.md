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

For a copyable end-to-end notebook workflow, start from
[`scripts/templates/boilerplate_run.ipynb`](scripts/templates/boilerplate_run.ipynb). It exposes every
current config default, can run locally or submit the generated YAML through
the generic CPU-only [`slurm/run_config.slurm`](slurm/run_config.slurm), queries
the complete matching execution, and keeps Figures 1–3 in separate documented
cells with optional high-resolution saving.

Run notebooks are grouped by experiment family: shared-cap work is under
`scripts/n100/multi_cap/`, while cap-specific continuations and parameter
sweeps are under `scripts/n100/u40/`, `scripts/n100/u160/`, and
`scripts/n100/u1280/`. Notebook filenames describe the particular sweep rather
than repeating the directory hierarchy.

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

The default production target is the canonical pair `results/results.sqlite3`
and `results/results.parameters.sqlite3`. Deliberately distributed or
replaceable long-running experiments may use named scratch pairs so a laptop,
the `bar` GPU, and Slurm jobs can progress independently without transferring a
multi-gigabyte live database. Such isolation is an orchestration choice, not a
SQLite requirement; scratch pairs must retain both files and their immutable
config provenance.

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
`execution_device` in its methodology records. The notebook adapter's direct
bar-GPU launchers also set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so member
sharding reduces actual VRAM use instead of every JAX process reserving most of
the card up front.

### RTX 4060 laptop worker

The underexplored-cap campaign is partitioned between `bar` and an 8 GB RTX
4060 Laptop GPU. From a clean Linux or WSL2 clone, the laptop user runs only:

```bash
bash scripts/local/run_rtx4060.sh
```

The launcher verifies `origin/main`, installs an isolated CUDA-enabled JAX
environment, validates GPU discovery, and resumes the committed laptop
manifest. It needs no downloaded input database because its assigned lanes
start from fresh Fourier seeds. See [`scripts/local/README.md`](scripts/local/README.md)
for the assigned caps and the result bundle returned to the server.

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
search chooses the step size instead. This path uses bound-aware L-BFGS-B
directly in normalized feasible controls, avoiding sigmoid/tanh saturation.
`optimizer: peak_refinement` also works directly in normalized feasible
controls. It normalizes the projected-gradient direction and uses a separate
Armijo backtracking line search for every batch member. A step is accepted only
when it sufficiently improves the regularized score; otherwise the member
stays at its incumbent controls and its next trial step is reduced. Its
settings use the `peak_*` prefix. Adam and L-BFGS retain their separate
`adam_*` and `lbfgs_*` settings.

Stability is configured independently of the schedule:

```yaml
parameters:
  block_size: 500
  J_tol: 1.0e-5
  u_tol: 1.0e-4
  v_tol: 1.0e-4
  projected_gradient_tol: 1.0e-4
  projected_gradient_alpha: 1.0
runtime:
  auto_halt: true
```

Set any tolerance to `null` to remove that criterion and its device-side
calculation. The projected-gradient mapping is evaluated in normalized feasible
coordinates (`u` in `[0,1]`, `v` in `[-1,1]`) and reported as
`||G_alpha|| / sqrt(2P)`. When enabled, its value is stored both at the final
stability checkpoint and at the actual stored best step as
`best_projected_gradient_rms`.

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
  perturbed: true
  perturbation_levels: [0.0005, 0.001, 0.0025, 0.005, 0.01]
```

`runtime.initialisations` remains the number of new Fourier starts. If this
query matches five runs and perturbation is enabled, each stored control creates
one start at every configured perturbation level. With the five default levels,
that gives 25 queried starts after the new Fourier starts for every current
sweep case. Set `perturbed: false` to append each selected stored control once
without perturbing it. `status: complete` is applied automatically unless
`where` specifies another status. String ranges use the same `MIN:MAX`, `:MAX`,
and `MIN:` notation as `query.py`.

Queried controls are resampled over normalized time when `N` changes. Their
perturbations are reproducible continuous five-mode Fourier curves applied in
bounded-control space, with each level interpreted as an RMS fraction of
`u_max` and `v_max`. The perturbed controls are clipped to `0 <= u <= u_max`
and `-v_max <= v <= v_max` before the inverse sigmoid/tanh maps. Each resulting
run stores `initialization_source: query`, its `source_run_id`, selected
`source_control_kind`, perturbation level/index, and deterministic perturbation
seed; Fourier runs store `initialization_source: fourier`.
No-match queries fail before optimization rather than silently falling back to
random starts. Use `query: null` to disable stored-control initializations.

To continue Adam itself rather than cold-starting from a stored control, select
an unperturbed best or final checkpoint and opt in explicitly:

```yaml
query:
  where: {run_id: 12345}
  limit: 1
  control_kind: final
  perturbed: false
  resume_optimizer: true
```

This restores the raw controls, first and second moments, and the per-run Adam
bias-correction counter. It is available only for Adam checkpoints created
after this storage extension. Set `resume_optimizer: false` explicitly when the
old controls should use a fresh Adam count and zero moments; `true` means an
exact optimizer resume.

For a case-by-case restart sweep, set `match_parameters` and a finite `limit`.
The runner groups the rows selected by `where`, selects the best rows whose
stored values exactly match each current scalar case, and uses `fallback_where`
only for cases with no match. For example:

```yaml
runtime:
  initialisations: 0
  max_cases_per_batch: 10
  max_initialisations_per_batch: 25
  max_batch_elapsed_seconds: 18000
query:
  where: {queue_id: 700843, batch_index: 0, status: running}
  limit: 1
  order_by: best_score
  descending: true
  control_kind: best
  perturbed: false
  match_parameters: [adam_learning_rate, adam_beta1, adam_beta2, smoothness, sharpness]
  fallback_where: {queue_id: 696482, status: complete, u_max: 40.0}
```

`runtime.initialisations: 0` is valid when the query supplies at least one
stored start per case. `max_cases_per_batch` shards each compilation-compatible
case group into batches no larger than the requested count while retaining one
seed across the shards. `max_initialisations_per_batch` independently shards
the complete per-case population (Fourier starts first, then queried starts)
without dropping or regenerating any member. This controls peak GPU memory for
long histories while preserving the numerical sweep and initialization set.
`max_batch_elapsed_seconds` checks wall time between bounded execution chunks,
stores the partial controls and diagnostics with `termination_reason:
time_limit`, and then continues to the next batch. `max_elapsed_seconds` caps a
whole config; `distribute_max_elapsed_across_batches: true` gives each batch an
equal share of that total (for example, three cap batches share twelve hours as
four hours each).

For long detached GPU chains, `ofc.resilient_queue` launches every config in a
separate subprocess. A failed or killed config is closed as failed and the next
queued config still starts; its combined log reports the current config, saved
step, halted count, and elapsed time.

`u_isbound`, `v_isbound`, and `block_size` are intentionally scalar because
they change compiled or diagnostic structure. The maker validates all values
before writing YAML and refuses to overwrite an existing named config,
preserving the file required to reproduce earlier runs. `default_parameters()`
and `default_runtime()` return new mappings on every call.

## Physical parameter conversion

The optimization remains dimensionless, while `ofc.physical` provides the
laboratory interpretation as a separate pure calculation layer. Its equations
are

```text
r_bg = a_bg sqrt(m / (hbar t_star))
tau = T / t_star
u_max = Gamma_max / gamma
v_max = nu_max / gamma
```

Each `solve_*` function takes exactly one `None` and returns that unknown. For
example, the physical optical-width cap can be recovered from a linewidth and
dimensionless cap:

```python
from ofc.physical import solve_optical_width, solve_time_scale

Gamma_max = solve_optical_width(
    gamma=2 * 3.141592653589793 * 15_000,
    u_max=40,
    Gamma_max=None,
)
tau_range = solve_time_scale(
    T=[1e-6, 1e-5],
    t_star=1e-5,
    tau=None,
)  # (0.1, 1.0)
```

The same inversion is available through `solve_background_scale()` for any of
`a_bg`, `m`, `t_star`, and `r_bg`, and through `solve_detuning_scale()` for any
of `gamma`, `nu_max`, and `v_max`. A scalar calculation returns a `float`. A
two-element list or tuple denotes a closed physical range and produces an
ordered `(lower, upper)` tuple using interval arithmetic. `None` denotes the
single unknown; it is not an open range endpoint.

To convert one complete laboratory setup at once:

```python
from math import pi

from ofc.physical import physical_to_dimensionless

parameters = physical_to_dimensionless(
    a_bg=-7.4e-11,       # m
    m=1.46e-25,          # kg
    t_star=1e-5,         # s
    T=[1e-6, 1e-5],      # s
    gamma=2 * pi * 15e3, # use one rate convention consistently
    Gamma_max=2 * pi * 600e3,
    nu_max=2 * pi * 15e6,
)
# {'r_bg': ..., 't_interval': (0.1, 1.0), 'u_max': 40.0, 'v_max': 1000.0}
```

The returned names match `ResolvedConfig`, with `t_interval` representing
`tau`. Scalar output can be passed directly to `make_config()`. Range tuples
match `Results.search()` range filters; convert a chosen set of range points to
a list when constructing a parameter sweep. The default `hbar` is in SI, so
`a_bg`, `m`, and the time variables must use metres, kilograms, and seconds.
The optical ratios accept Hz or angular frequency as long as `gamma`,
`Gamma_max`, and `nu_max` all use the same convention.

For repeated work with one physical system, instantiate an immutable
`AtomConfiguration` rather than resupplying its fixed values:

```python
from math import pi

from ofc.physical import AtomConfiguration

sr88 = AtomConfiguration(
    a_bg=-7.4e-11,        # m
    gamma=2 * pi * 15e3, # rad/s
    g_2=(1e21) ** 2,      # m^-6; n^2 for this noninteracting Bose example
    m=1.46e-25,           # kg
    t_star=1e-5,          # s; configured short-time frame
)

parameters = sr88.dimensionless_parameters(
    Gamma_max=2 * pi * 600e3,
    nu_max=2 * pi * 15e6,
    T=[1e-6, 1e-5],
)
```

The object exposes `l_star`, `r_bg`, and `short_time_interval` properties plus
`solve_time_scale()`, `solve_optical_width()`, `solve_detuning_scale()`, and
`solve_background_scale()` methods. Calling `solve_time_scale()` or
`dimensionless_parameters()` without `T` uses `T=t_star`. A supplied `T`, or a
physical `T` recovered from `tau`, must be no larger than the configured
`t_star`; otherwise the object raises an error instead of silently applying the
short-time model outside its declared frame.

### Molecular-yield conversion

The optimized molecular objective omits the initial pair-density factor. Under
the loss prefactor implemented in `Physics.molecular_objective()`, its physical
product-density conversion is

```text
n_mol = g_2(0) l_star^3 J_mol,
l_star = sqrt(hbar t_star / m).
```

This follows by substituting `a=l_star*a_tilde`,
`eta=l_star*eta_tilde`, and `t=t_star*tau` into the dimensional loss integral,
then using `t_star=m*l_star^2/hbar`. It is also dimensionally consistent:
`g_2(0)` has units `L^-6`, so `g_2(0) l_star^3 J_mol` has density units `L^-3`.

```python
n_mol = sr88.molecular_density(dimensionless_yield=0.25)  # m^-3

# The standalone form accepts scalar or two-endpoint range inputs:
from ofc.physical import molecular_density
n_mol_range = molecular_density(
    [0.2, 0.3],
    g_2=sr88.g_2,
    l_star=sr88.l_star,
)
```

Here `g_2(0)` is the unnormalized equal-position pair density used in the
Qi--Shi--Zhai contact relation, not dimensionless normalized coherence
`g^(2)(0)`. For their noninteracting initial states it is `n^2` for spinless
bosons and `n_up*n_down` for two-component fermions. The conversion exactly
restores the normalization used by the current code, but the report's existing
counting-convention caveat remains: the physical derivation must still confirm
whether the adopted prefactor counts product molecules, lost pairs, or depleted
atoms before the result is presented as an absolute measured molecule density.

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
configs and batches launched together remain searchable as a group. One
configuration execution is identified by `(config_document_id, queue_id)`:
this keeps all Slurm-array batches from one submission together while
distinguishing multiple configs submitted under the same queue ID. Its stored
UTC start time provides a stable newest-to-oldest ordering.

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

Persistence is separated by meaning and joined by the unique `run_id`:

- `results/results.sqlite3` contains physical/calculated outputs only;
- `results/results.parameters.sqlite3` contains exact config documents,
  resolved per-run settings, execution provenance, and optimizer-stage values.

The project has one default canonical pair, while explicitly isolated scratch
work may use another pair. Both databases use generic name/value and named-array
records. Adding a config field, optimizer, calculated scalar, or diagnostic
therefore does not require a column migration or deletion of earlier results.
SQLite WAL mode permits concurrent readers and workers. SQLite still serializes
each individual writer, but the runner uses short `BEGIN IMMEDIATE`
transactions with a long busy timeout, so several processes can safely share a
pair. Scratch pairs primarily reduce write contention, failure blast radius,
transfer size, and accidental cross-experiment queries; they are not needed
merely to allow another process to read.

For every initialization, the physical database stores `N`, `T`, `r_bg`,
`u_max`, `v_max`, full score/objective/penalty histories, and both raw and
bounded controls at the initial, best, and final states. Adam runs additionally
store the first and second moments at the best and final states; together with
the stored raw controls and best/final step counts, these are exactly resumable
optimizer states. Only the final stability values are retained (not every
intermediate checkpoint), together with best/final metrics and the best step.
The database stores the maximum absolute first
derivatives of the best `u` and `v`. Maximum absolute second derivatives are
stored only when the corresponding sharpness coefficient is active. Its stable
`runs` index also duplicates the
durable provenance keys `config_id`, `config_document_id`, `batch_id`,
`execution_id`, `queue_id`, and `batch_index`, together with status and UTC
start/completion times. These fields keep physical results independently
orderable and attributable even if the flexible parameter schema later changes.

Before calculation begins, the parameter database allocates the `run_id` and
stores the full original config plus its scalar resolved case, runtime settings,
config/batch/queue identifiers, seed, and initialization index. Fourier starts
also store the exact per-control offset and sampled sine/cosine coefficient
arrays. Thus every physical record can be traced to the exact starting curve
and method that produced it even after the config schema or optimizer changes.

The optimizer does **not** return to Python every `block_size` steps. A compiled
device loop calculates only the enabled block diagnostics in place: relative
score movement, normalized-control RMS movement, and projected-gradient mapping
RMS. A `null` tolerance statically removes that calculation. With
`runtime.auto_halt: true`, all members of a batch must pass every enabled
criterion for three consecutive blocks; the device loop then exits early.
Otherwise it ends at the schedule boundary, where Python commits the stage.
Expensive optimizers can set `runtime.max_steps_per_chunk` so elapsed-time
checks and durable writes occur more frequently than the default 10,000-step
host chunk.
At every non-final learning-rate boundary, members that individually reached
three consecutive passing blocks are committed and sliced out of the optimizer
state and parameter arrays before the next compiled stage. The run log prints
the newly halted and remaining member counts. Only the final diagnostic values
for each member are written. Seed sensitivity is deliberately
excluded because it requires a separately chosen population.

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
python query.py --config-run 1                         # latest config execution
python query.py --config-run 2                         # second latest
python query.py --config-run 1 --where u_max=:50 --best
python query.py --where u_max=:50 --order-by best_score --descending --limit 1
```

`--best` is shorthand for ordering by `best_score` from largest to smallest and
returning one run. `--config-run N` ranks matching configuration executions by
their stored UTC start time, where 1 is newest. Filters are applied before the
rank, so `--where config_name=example --config-run 2` means the second most
recent execution of that particular config.

Use the read-only Python API when building analyses:

```python
from ofc.results import Results

results = Results("results/results.sqlite3")
rows = results.search(
    config_name="example",
    u_max=(25.0, 100.0),
    best_max_abs_du_dt=(None, 20.0),
)
latest = results.search_config_run(rank=1, status="complete")
second_latest = results.search_config_run(rank=2, status="complete")
best_below_50 = results.search(
    status="complete",
    u_max=(None, 50.0),
    order_by="best_score",
    descending=True,
    limit=1,
)
run = results.get(rows[0]["run_id"])
history = run["history"]
best_controls = run["controls"]["best"]
best_raw = run["controls"]["best_raw"]
final_adam_state = run["adam_states"]["final"]
tolerances = run["tolerances"]
```

The legacy control keys `initial`, `best`, and `final` remain the bounded
physical controls. Their raw optimizer-space counterparts use the
`initial_raw`, `best_raw`, and `final_raw` keys. `Results.adam_state(run_id,
"best")` and `Results.adam_state(run_id, "final")` return `raw`, `count`,
`first_moment`, and `second_moment`. Runs created before this storage extension
continue to load normally but do not have raw controls or Adam states.

Numerical range endpoints are inclusive; use `None` for an open endpoint.
Useful indexed fields include `config_name`, `config_file`, `config_id`,
`config_document_id`, `batch_id`, `execution_id`, `queue_id`, `started_utc`,
`completed_utc`, `r_bg`, `u_max`, `N`, `best_score`, `best_objective`,
`best_penalty`, `best_step`, `best_max_abs_du_dt`, `best_max_abs_dv_dt`,
`best_max_abs_d2u_dt2`, and `best_max_abs_d2v_dt2`.

## Unified standard plotting

Running `plot.py` with no query selects the most recently stored completed
config and writes one unified PNG summary. If no configuration parameter
varies, all selected initializations are displayed in one summary rectangle.

When exactly one configuration parameter varies, `plot.py` automatically
creates one row per parameter value. Every row contains:

- the selected runs' regularized-score histories, with the median and 10th–90th
  percentile spread calculated only from those displayed runs;
- dashed vertical markers at actual learning-rate changes;
- a right-hand molecular-objective strip sharing the history panel's vertical
  scale, with the seed spread $S_J$ shown above it; and
- the raw optimized $u$ and $\nu$ controls below.

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
python plot.py --config-run 2 --output-dir figures/second_latest
```

If a selection varies more than one configuration parameter, filter it down to
one sweep or provide `--sweep-parameter NAME`.

The pure functions `plot_convergence()`, `plot_yield_distribution()`, and
`plot_controls()` accept an optional `sweep` specification and otherwise
construct the numbered initialization sweep. `plot_standard_summary()`,
`plot_standard_figures()`, and `save_standard_figures()` infer the standard
selection automatically. Pass `sweep_parameter="u_max"` when selecting one
axis from multidimensional data. The legacy individual plotting functions
remain available for bespoke analysis.

The scatter and line plotting functions accept independent keyword-only
`log_base_x` and `log_base_y` settings. Their default is `None`, which leaves
the normal or inferred axis scale unchanged; pass a base such as `10` to make
that axis logarithmic. Without a supplied range, strictly positive data use
ordinary log spacing and data containing negative values use signed
symmetric-log spacing, so the same powers repeat in the negative direction.
`base_x` and `base_y` independently control the major tick labels: `"axis"`
writes powers of the axis base, `None` writes actual values such as `100`, and
a numeric base writes powers of that chosen base. The Figure 2 x-axis defaults
to `None` so grouped values are always written directly. `x_multiplier` and
`y_multiplier` default to `1` and scale the corresponding logarithmic lattice:
with base 10, multiplier 1 gives `1, 10, 100, ...`, multiplier 2 gives
`2, 20, 200, ...`, and multiplier 0.1 gives `0.1, 1, 10, ...`; signed axes add
zero and mirror these values below it. Multi-panel plots accept either one
`y_multiplier` for every panel or a separate value per panel. On zero-containing
log axes, the interval from zero to the multiplier is linear and occupies the
same display distance as the local logarithmic interval from the multiplier to
twice the multiplier (about 0.301 of a decade for base 10); the scale remains
truly logarithmic beyond that point. Plot axes retain
small, unlabeled minor
ticks at quarter-unit subdivisions within every decade (for example `1.25`,
`1.5`, ... `9.75`, then `12.5`, `15`, ... `97.5` for base 10). This is four
times the numeric subdivision density of integer-only marks. The pattern resets
each decade; whole-number subdivisions and marks following larger logarithmic
gaps are slightly longer. Minor grid lines remain
suppressed. If a positive axis extends below its labelled multiplier lattice,
the lower powers remain visible as longer unlabeled minor marks, so that region
does not lose its scale cues. On a logarithmic axis, `x_range` and `y_range`
are absolute clipping envelopes rather than forced limits. The effective zero
cutoff is the larger of the requested minimum and the smallest plotted nonzero
magnitude; all values at or below that cutoff occupy the zero position. The
upper limit advances to the next numbered major tick at or above the largest
plotted magnitude, without exceeding the requested maximum. If that hard cap
falls between regular logarithmic ticks, the cap itself becomes a numbered
major tick. The same tick-aligned upper boundary is used when no range is
supplied. Negative data use the same cutoff and maximum by absolute magnitude.
Consequently a requested `[1e-3, 1000]` range displays `1e-3` as zero and data
ending at `70` through the numbered `100` tick. Linear-axis ranges remain exact.
Sweep
colourbars have no minor ticks. Figure 2 shows the actual
grouped x values, retains sweep colours, and omits its redundant colourbar. Its
upper panel overlays a box-free percentile glyph at every sweep value: a
neutral-grey vertical connector joins 10th-, median-, and 90th-percentile lines.
The outer lines extend well beyond the initialization point cloud and the median
is slightly longer. Best-score points remain highlighted without a connecting line;
its lower panel reports the relative seed sensitivity
`S_J = (J_90 - J_10) / abs(J_50)`. Its tolerance references come directly
from every distinct `J_tol` in the queried database rows; a normal single-config
query produces one line, while a mixed query displays every stored tolerance.
References below a selected log cutoff remain visible on its zero boundary.
`seed_sensitivity_log_base_y`, `seed_sensitivity_base_y`,
`seed_sensitivity_y_multiplier`, and `seed_sensitivity_y_range` control the
lower panel independently of the objective panel. Figure 2 has no sensitivity
colourbar. `line_alpha` controls percentile-glyph opacity, while
`point_size` controls initialization markers. Every plot
accepts `x_range`, `y_range`, `x_label`, and `y_label`; labels accept
Matplotlib LaTeX math text.
Figures have no built-in title. The controls x-axis retains its intermediate
tick marks but labels only `0.0` and the final time (including a trailing `.0`).
Figure 1 keeps a very narrow physical gutter between its three panels while
preserving equal pixel spacing for equivalent vertical tick intervals.
Shared figures render at 600 DPI and PNG exports use 600 DPI; PDF output remains
resolution-independent vector artwork.

`plot_sweep_run_summaries()` provides the compact per-sweep view used by the
analysis notebooks. Each sweep value gets one regularized-score convergence
panel and two raw-control panels. It omits the duplicated objective scatter and
reports best and median score, score-stability counts, and control-stability
counts and thresholds for that value.

```python
figure, axis = plot_yield_distribution(
    runs,
    log_base_y=10,
    base_y=None,
    point_size=12,
)
```

## Repository layout

```text
run_config/                 immutable YAML configs and the sole config maker
src/ofc/
  config.py                 schema, validation, Cartesian sweep planning
  initialization.py         all random Fourier initialization choices
  physics.py                inelastic dimensionless equations only
  optimization.py           batched JIT optimizers and device-side stability
  analysis.py               post-processing and best-control derivatives
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
