# Fast in-memory test

`fast_cheap_test.ipynb` loads `cheap_test.yaml`, performs the configured Fourier
optimizations entirely in memory, and displays the standard convergence,
yield-distribution, and control figures inline.

The notebook imports `in_memory_runner.py`, which intentionally has no imports
from `ofc.runner`, `ofc.storage`, or `ofc.results`. The config's database value
is an unused sentinel, and the notebook verifies its file fingerprint is
unchanged across the run.

Start Jupyter from the repository root and open the notebook:

```bash
conda activate optical-feshbach-control
jupyter lab sandbox/fast_cheap_test.ipynb
```

Edit only `cheap_test.yaml` for quick experiments. It is deliberately a scalar
config; production sweeps continue to use `run.py` and SQLite.
