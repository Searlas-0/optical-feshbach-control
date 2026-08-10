# Reports

Store written analysis and publication-oriented documents here. Reports should
reference source-labelled artefacts from `results/` and `figures/` instead of
creating data beside the document.

## Rebuilding the showcase

From the repository root, regenerate the conceptual two-channel schematic and
the current numerical evidence before compiling the document:

```bash
python reports/scripts/generate_figure1.py
python reports/scripts/generate_showcase_results.py
latexmk -cd -pdf -shell-escape reports/main.tex
```

The numerical generator resolves the completed baseline and regularisation
sweeps to their configuration-document and queue identifiers. It also reads the
latest saved targeted-refinement iterate at each selected cap. Unfinished
iterates remain explicitly marked and must not be described as converged
optima. The exact source identifiers, run states, values, and generation time
are written to `figures/showcase_results_metadata.json`; matching LaTeX macros
are written to `figures/showcase_results_macros.tex`.

The showcase axes are dimensionalised in
`scripts/generate_showcase_results.py` with the stated (t_\star), background
scattering length, atomic mass, linewidth, and initial pair density. The JSON
records both the untouched dimensionless objective and every conversion
constant. In this codebase, `g_2(0)` is an unnormalised equal-position pair
density in `m^-6`, not the dimensionless correlation function commonly written
as (g^{(2)}(0)); consequently the displayed molecular density remains a proxy
until the product-counting convention is fixed.
