# accelerated-bess-optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewjeffallen/accelerated-bess-optimization/blob/main/bess_cuopt_benchmark.ipynb)

GPU-accelerated BESS optimization test & benchmark. A battery price-arbitrage LP for
**ERCOT North Hub (`HB_NORTH`)**, modeled once in Pyomo and solved two ways:

| | Solver | Hardware |
|---|---|---|
| Baseline | Pyomo + HiGHS (`appsi_highs`) | CPU |
| Accelerated | Pyomo + NVIDIA cuOpt (`cuopt`) | GPU |

## Run it in Google Colab

1. Click the **Open in Colab** badge above (or open `bess_cuopt_benchmark.ipynb` via `File -> Open notebook -> GitHub`).
2. `Runtime -> Change runtime type -> GPU` (T4 is fine).
3. Run all cells. The notebook installs dependencies and clones this repo.

If you restart the session after installing, re-run from the top.

## Run it locally (CPU only)

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv sync                      # add --extra ercot for real-price downloads
uv run python benchmark.py --cpu-only --days 1 7 30 90
```

On a Linux machine with an NVIDIA GPU (CUDA 12 or 13, Python 3.11+), sync with the matching
cuOpt extra and drop `--cpu-only`:

```bash
uv sync --extra cuda12       # CUDA 12.x   (or: uv sync --extra cuda13)
uv run python benchmark.py --days 1 7 30 90 365 730
```

## The model

Perfect-foresight energy arbitrage for one battery (default 100 MW / 200 MWh, 92% one-way
efficiency, $2/MWh throughput cost). Variables per interval: charge, discharge, state of charge.
Constraints: SOC balance, power/SOC bounds, and end-of-horizon SOC >= start SOC. Pure LP, no binaries.

Size grows linearly with the horizon: `3 * intervals` variables and `intervals + 1` constraints
(about 105k variables for a year of 15-minute data).

## Price data

- `synthetic` (default): ERCOT-like shape with solar dip, evening peak, spikes and negative prices.
  **Not real data.** It exists so the benchmark runs anywhere with a controllable size.
- `ercot`: real `HB_NORTH` settlement point prices through the
  [`gridstatus`](https://github.com/gridstatus/gridstatus) package (`--source ercot --start ... --end ...`).
  Falls back to synthetic with a warning if the download fails. This path depends on
  ERCOT/gridstatus availability and hasn't been verified in every environment.
- `csv`: your own file (`--source csv --csv prices.csv --time-col timestamp --price-col price`).

## What to expect

HiGHS is very good on LPs of this shape and will usually win on small and mid-sized
problems. cuOpt's GPU method (PDLP) targets large LPs, so the interesting question is where, or whether,
the curves cross as the horizon grows. GPU first-order methods also stop at a tolerance, so the
benchmark reports the objective gap between solvers. Compare timings only when that gap is small.

Timing covers `SolverFactory.solve(...)` (including Pyomo-to-solver transfer). Pyomo model-build
time is the same for both and is reported separately.

## Files

- `bess_opt.py`: price loaders, Pyomo model, solve helpers
- `benchmark.py`: sweep, summary table, plots, CLI
- `bess_cuopt_benchmark.ipynb`: Colab notebook
- `pyproject.toml` / `uv.lock`: dependencies. cuOpt is an optional extra (`cuda12` / `cuda13`) because it needs a GPU and Linux.

## Notes

- cuOpt's Pyomo plugin requires a recent Pyomo (`>=6.10`).
- Extra cuOpt solver settings can be passed via `GPU_OPTIONS` in the notebook (forwarded to cuOpt's
  `SolverSettings.set_parameter`).
