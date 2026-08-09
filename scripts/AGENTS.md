# Mandatory notebook style

These instructions apply to every notebook created or edited under `scripts/`.

1. Use `scripts/templates/boilerplate_run.ipynb` as the source of truth for every run notebook wherever possible. Copy its cell order, headings, guards, naming, and public adapter calls.
2. Do not add, remove, reorder, merge, or redesign boilerplate sections without the user's explicit permission in the current conversation. A request for a new run or analysis is not permission to change the notebook format.
3. Keep code cells declarative and easy for the user to edit. They may contain experiment descriptions, names, flags, parameter mappings, query filters, plot settings, and a call to the public notebook adapter.
4. Never define functions or classes in a run notebook. Do not put subprocess, Slurm, database, filesystem, validation, device-detection, batching, or plotting mechanics in notebook cells.
5. Put reusable behavior in `src/ofc/notebook_workflow.py` or the relevant package module, add tests there, and expose only editable arguments in the notebook.
6. Bespoke notebooks are allowed only when the boilerplate genuinely cannot represent the task and the user explicitly approves the deviation first. If uncertain, stop and ask.

Allowed experiment-specific edits are limited to prose describing the run and the exposed values: `run_name`, `Activated` flags, config parameters, runtime parameters, initialization query, execution resources, database query arguments, and plot/output arguments.

For every stored-run initialization query, expose and deliberately set
`resume_optimizer`: `false` resets the optimizer at the selected controls;
`true` restores the stored Adam count and moments and therefore requires exact,
unperturbed `best` or `final` controls. Never choose between reset and resume
implicitly when creating a notebook.
