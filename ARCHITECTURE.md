# Architecture contract

This repository uses ports-and-adapters boundaries so future refactors do not
recreate the coupling in the reference project. These rules are intentional and
are checked by `tests/test_architecture.py`.

```text
make_config.py ──YAML──▶ run.py / run.slurm ──arguments──▶ runner
                                                  │
                         ┌────────────────────────┼──────────────────────┐
                         ▼                        ▼                      ▼
                  numerical modules        analysis arrays       storage records
                         │                                               │
                         └──────── return values only ───────────────────┘
                                                        ┌────────────────┴───────────────┐
                                                        ▼                                ▼
                                                physical results                 run parameters
                                                results.sqlite3          results.parameters.sqlite3
                                                        └────────────────┬───────────────┘
                                                                         ▼
query.py ──filters──▶ Results ──plain mappings/arrays──▶ plot.py ──▶ plotting
```

The arrows are arguments, return values, YAML, or SQLite records—not hidden
imports or shared mutable objects.

- Config creation validates settings and writes immutable YAML. It cannot run a
  calculation, read results, or plot.
- `run.py` and `slurm/run.slurm` are composition roots. They resolve arguments
  and call the runner; they contain no physics, optimization, analysis, SQL, or
  plotting logic.
- `physical.py`, `physics.py`, `initialization.py`, `optimization.py`, and
  `analysis.py` are pure numerical layers. They accept arrays/scalars and never
  touch files. `physical.py` is the laboratory-to-dimensionless interpretation
  boundary; it returns ordinary values that can be passed into config creation
  or result queries.
- `optimization.py` owns the compiled block-stability state and early-exit
  condition. Optional tolerance flags are static compilation features, so a
  disabled diagnostic has no device-loop calculation. The runner receives only
  terminal diagnostic values for persistence and never interrupts a stage at a
  block checkpoint.
- Stored-control queries remain split across the same boundaries: `config.py`
  validates the declarative query, the runner retrieves matching controls and
  assigns reproducible seeds, and `initialization.py` performs resampling plus
  bounded-space perturbation/clipping without reading the database itself.
- `storage.py` is a write adapter. It accepts records and commits physical
  outputs separately from flexible config/methodology records; `run_id` is the
  only link. It does not know how values were calculated.
- `results.py` is read-only. It searches and returns values but cannot launch a
  run or plot them.
- `plotting.py` accepts already-retrieved mappings. It cannot query a database or
  launch calculations. `plot_cli.py` is the thin adapter that explicitly passes
  retrieved mappings to it.
- The package `__init__.py` performs no eager process imports. Import the exact
  module a process needs.

If new functionality needs several boundaries, put orchestration in a thin CLI
adapter or the runner and keep the underlying operation in its owning module.
